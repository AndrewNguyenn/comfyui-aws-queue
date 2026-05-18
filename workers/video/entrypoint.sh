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
    done < <(cd /opt/worker && python -m model_types --bash)

    NVME_MIB=$(df -m --output=avail "$CACHE_ROOT" | tail -1 | tr -d ' ')
    NUM_MOUNTS=${#TYPES[@]}
    RESERVE_MIB=10240
    CACHE_PER_MOUNT_MB=$(( (NVME_MIB - RESERVE_MIB) / NUM_MOUNTS ))
    if [ "$CACHE_PER_MOUNT_MB" -lt 512 ]; then CACHE_PER_MOUNT_MB=512; fi
    echo "NVMe ${NVME_MIB} MiB / ${NUM_MOUNTS} mounts → ${CACHE_PER_MOUNT_MB} MiB per mount"

    for s3_type in "${!TYPES[@]}"; do
        comfy_dir=${TYPES[$s3_type]}
        mount_at="$COMFY_MODELS/$comfy_dir"
        cache_dir="$CACHE_ROOT/mount-s3/$s3_type"
        mkdir -p "$mount_at" "$cache_dir"
        if mountpoint -q "$mount_at"; then
            MOUNTS+=("$mount_at")
            continue
        fi
        find "$mount_at" -mindepth 1 -maxdepth 1 -delete 2>/dev/null || true
        if mount-s3 \
                --prefix "$s3_type/" \
                --cache "$cache_dir" \
                --max-cache-size "$CACHE_PER_MOUNT_MB" \
                --allow-other \
                --metadata-ttl 60 \
                --read-only \
                "$MODELS_BUCKET" "$mount_at"; then
            echo "  mounted s3://$MODELS_BUCKET/$s3_type/ at $mount_at"
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
    (cd /opt/worker && python -u -m warm_pinned 2>&1 | sed 's/^/warm: /' &)
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
exec python -u worker.py
