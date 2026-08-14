#!/usr/bin/env bash
# Video worker entrypoint.
#
# Same mount-s3 + NVMe + warm + supervision pattern as image worker, plus
# video-specific SageAttention env. /opt/cache is bind-mounted from the host
# (formatted by user-data in compute.ts onto the included NVMe instance store).
set -euo pipefail

echo "comfy-video-worker starting on $(hostname)"
echo "fleet=${FLEET:-video} queue=${QUEUE_URL:-unset}"

export TQDM_DISABLE=1
export TRANSFORMERS_VERBOSITY=warning
export DIFFUSERS_VERBOSITY=warning
# SageAttention: prefer Ampere/Ada-optimized kernels. Sage is enabled at the
# ComfyUI process via --use-sage-attention (set in worker.py for fleet=video).
export SAGE_ATTENTION_BACKEND=auto

CACHE_ROOT=/opt/cache
MODELS_BUCKET=${MODELS_BUCKET:-}
COMFY_MODELS=/opt/comfy/models

mkdir -p "$CACHE_ROOT/mount-s3"
echo "cache root: $(df -h "$CACHE_ROOT" | tail -1 | awk '{print $1 " " $2}')"

declare -a MOUNTS=()
if [ -z "$MODELS_BUCKET" ]; then
    echo "MODELS_BUCKET unset; skipping S3 mounts"
else
    declare -A TYPES
    while IFS='=' read -r key val; do
        key=${key#[}; key=${key%]}
        TYPES[$key]=$val
    done < <(cd /opt/worker && python3 -m model_types --bash)

    # Per-mount cache budget is WEIGHTED, not an even split. This fleet used to
    # divide the NVMe evenly, which was fine when no video weights existed. The
    # SCAIL-2 pipeline breaks that: a single 15.3 GiB diffusion model plus a
    # 10.6 GiB umt5 text encoder both exceed the ~7.5 GiB an even split yields
    # across our catalog types, so every job would re-stream them from S3 and
    # thrash the LRU. Weight the two mounts that hold the hot-set accordingly.
    # Each cap = weight/total_weight * (NVMe - reserve), so the sum stays bounded
    # below the NVMe regardless of instance/disk size.
    cache_weight() {
        case "$1" in
            diffusion_models) echo 45 ;;  # SCAIL-2 14B fp8_scaled ~15.3 GiB
            text_encoders)    echo 25 ;;  # umt5-xxl-enc-bf16 ~10.6 GiB
            lora)             echo 5 ;;   # lightx2v distill + character LoRAs
            detection)        echo 5 ;;   # vitpose-l-wholebody + yolov10m ONNX
            clip_vision)      echo 3 ;;   # CLIP-ViT-H ~3.7 GiB
            vae)              echo 2 ;;
            nlf)              echo 2 ;;
            *)                echo 1 ;;   # unused on the video fleet
        esac
    }
    NVME_MIB=$(df -m --output=avail "$CACHE_ROOT" | tail -1 | tr -d ' ')
    RESERVE_MIB=10240
    AVAIL_MIB=$(( NVME_MIB - RESERVE_MIB ))
    if [ "$AVAIL_MIB" -lt 1024 ]; then AVAIL_MIB=1024; fi
    TOTAL_WEIGHT=0
    for s3_type in "${!TYPES[@]}"; do
        TOTAL_WEIGHT=$(( TOTAL_WEIGHT + $(cache_weight "$s3_type") ))
    done
    echo "NVMe ${NVME_MIB} MiB, ${AVAIL_MIB} MiB cacheable across weight ${TOTAL_WEIGHT}"

    for s3_type in "${!TYPES[@]}"; do
        comfy_dir=${TYPES[$s3_type]}
        mount_at="$COMFY_MODELS/$comfy_dir"
        cache_dir="$CACHE_ROOT/mount-s3/$s3_type"
        cache_mb=$(( AVAIL_MIB * $(cache_weight "$s3_type") / TOTAL_WEIGHT ))
        if [ "$cache_mb" -lt 512 ]; then cache_mb=512; fi
        mkdir -p "$mount_at" "$cache_dir"
        if mountpoint -q "$mount_at"; then
            MOUNTS+=("$mount_at")
            continue
        fi
        find "$mount_at" -mindepth 1 -maxdepth 1 -delete 2>/dev/null || true
        if mount-s3 \
                --prefix "$s3_type/" \
                --cache "$cache_dir" \
                --max-cache-size "$cache_mb" \
                --allow-other \
                --metadata-ttl 60 \
                --read-only \
                "$MODELS_BUCKET" "$mount_at"; then
            echo "  mounted s3://$MODELS_BUCKET/$s3_type/ at $mount_at (cache ${cache_mb} MiB)"
            MOUNTS+=("$mount_at")
        else
            echo "  WARN: mount-s3 failed for $s3_type (continuing)"
        fi
    done
fi

# Mount barrier (see image entrypoint for rationale).
for m in "${MOUNTS[@]}"; do
    for i in {1..20}; do
        if mountpoint -q "$m"; then break; fi
        sleep 0.25
    done
    mountpoint -q "$m" || echo "  WARN: $m never became a mountpoint"
done
echo "  ${#MOUNTS[@]} mounts visible"

if [ -n "$MODELS_BUCKET" ] && [ "${#MOUNTS[@]}" -gt 0 ]; then
    (cd /opt/worker && python3 -u -m warm_pinned 2>&1 | sed 's/^/warm: /' &)
fi

# Sentinel: kill the container if any mount-s3 daemon dies.
if [ "${#MOUNTS[@]}" -gt 0 ]; then
(
    while sleep 30; do
        for m in "${MOUNTS[@]}"; do
            if ! mountpoint -q "$m"; then
                echo "FATAL: mount $m disappeared; killing worker" >&2
                kill -TERM 1
                exit 0
            fi
        done
    done
) &
fi

cd /opt/worker
exec python3 -u worker.py
