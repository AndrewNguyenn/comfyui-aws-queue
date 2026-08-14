#!/usr/bin/env bash
# Image worker entrypoint.
#
# Sequence:
#   1. Mount each model-type S3 prefix at /opt/comfy/models/<type>/ via
#      Mountpoint for Amazon S3. ComfyUI's folder_paths.get_filename_list()
#      then sees every catalog file as a local file; first read streams from
#      S3 and lands in the NVMe cache, subsequent reads hit local disk.
#   2. Wait for every mount to be visible (FUSE mounts are async — without
#      this barrier ComfyUI may scan the dirs before mount-s3 populates them).
#   3. Background pre-warm: stream pinned models so they're hot before the
#      first matching job lands.
#   4. Background sentinel: watch every mount-s3 daemon; if any dies, kill
#      the worker process so ECS replaces the container.
#   5. Hand off to worker.py.
#
# /opt/cache (the mount-s3 disk cache root) is bind-mounted from the host;
# user-data in compute.ts formats it onto the NVMe instance store.
set -euo pipefail

echo "comfy-image-worker starting on $(hostname)"
echo "fleet=${FLEET:-image} queue=${QUEUE_URL:-unset}"

# Reduce noise from common chatty tools (v3 N1 — log spend control).
export TQDM_DISABLE=1
export TRANSFORMERS_VERBOSITY=warning
export DIFFUSERS_VERBOSITY=warning

CACHE_ROOT=/opt/cache
MODELS_BUCKET=${MODELS_BUCKET:-}
COMFY_MODELS=/opt/comfy/models

mkdir -p "$CACHE_ROOT/mount-s3"
echo "cache root: $(df -h "$CACHE_ROOT" | tail -1 | awk '{print $1 " " $2}')"

# Compute the per-mount disk cache budget so the sum across all mounts can't
# exceed the available NVMe. NVMe minus a 10 GB OS/container reserve, split by
# per-type WEIGHTS (checkpoint-heavy) — see cache_weight() below. Computed after
# TYPES is loaded so we know the weight total.

# ----- 1. mount-s3 per model type -----
# We mount each catalog type at its ComfyUI directory name. S3 layout is
# s3://comfy-models/<catalog-type>/<filename>; ComfyUI expects
# /opt/comfy/models/<comfy-type-dir>/<filename>.
#
# --read-only: mount-s3 has limited write semantics, we never write here.
# --max-cache-size: per-mount cap; sum bounded below NVMe size (see above).
declare -a MOUNTS=()  # list of "$mount_at" paths for the barrier + sentinel
if [ -z "$MODELS_BUCKET" ]; then
    echo "MODELS_BUCKET unset; skipping S3 mounts (worker will see empty model dirs)"
else
    # Catalog type → ComfyUI dir mapping comes from workers/shared/model_types.py.
    # Eval'd into a bash assoc array so we don't have two lists to keep in sync.
    declare -A TYPES
    while IFS='=' read -r key val; do
        key=${key#[}; key=${key%]}
        TYPES[$key]=$val
    done < <(cd /opt/worker && python3 -m model_types --bash)

    # Per-mount cache budget is WEIGHTED, not an even split. The image fleet is
    # checkpoint-heavy (SDXL), so the checkpoints mount gets the lion's share to
    # hold the pinned hot-set (multiple ~7 GB checkpoints), while mounts this
    # fleet never uses (diffusion_models / unet / text_encoders — Flux/video-class)
    # get a small floor. Each cap = weight/total_weight * (NVMe - reserve), so the
    # sum stays bounded below the NVMe regardless of instance/disk size, and
    # mount-s3 LRU-evicts within each per-mount cap. (The video fleet uses the same
    # scheme with diffusion_models/text_encoders weighted up for SCAIL-2; see
    # workers/video/entrypoint.sh.)
    cache_weight() {
        case "$1" in
            checkpoint) echo 45 ;;   # the pinned hot-set lives here (~90 GiB on a 250 GB NVMe)
            lora)       echo 25 ;;
            controlnet) echo 8 ;;
            upscale)    echo 5 ;;
            ipadapter)  echo 5 ;;
            vae)        echo 2 ;;
            *)          echo 1 ;;     # light / unused on the image fleet
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
        # ComfyUI ships placeholder files (`put_checkpoints_here` etc.) in each
        # model dir. FUSE refuses to mount over a non-empty directory.
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

# ----- 1b. mount-s3 for prompt wildcards -----
# Wildcards aren't a models/ type — they're prompt .txt files. Impact-Pack
# loads them from its default custom_wildcards path (<node>/custom_wildcards —
# the node dir name comes from baked_nodes.txt), so mount the wildcards/ S3
# prefix straight there; downloaded packs are then picked up with no
# Impact-Pack config. Impact-Pack scans this dir once at startup, so a pack
# downloaded after a worker is running needs a worker rotation to be seen.
# Tiny cache — these are text files.
if [ -n "$MODELS_BUCKET" ]; then
    WC_AT=/opt/comfy/custom_nodes/comfyui-impact-pack/custom_wildcards
    WC_CACHE="$CACHE_ROOT/mount-s3/wildcards"
    mkdir -p "$WC_AT" "$WC_CACHE"
    if mountpoint -q "$WC_AT"; then
        MOUNTS+=("$WC_AT")
    else
        find "$WC_AT" -mindepth 1 -maxdepth 1 -delete 2>/dev/null || true
        if mount-s3 \
                --prefix wildcards/ \
                --cache "$WC_CACHE" \
                --max-cache-size 256 \
                --allow-other \
                --metadata-ttl 60 \
                --read-only \
                "$MODELS_BUCKET" "$WC_AT"; then
            echo "  mounted s3://$MODELS_BUCKET/wildcards/ at $WC_AT"
            MOUNTS+=("$WC_AT")
        else
            echo "  WARN: mount-s3 failed for wildcards (continuing)"
        fi
    fi
fi

# ----- 2. Barrier: wait until every mount is visible -----
# mount-s3 daemonizes when it succeeds, but FUSE may need a moment for the
# mountpoint to be visible to other processes. Without this, ComfyUI's
# startup folder_paths scan can race the mounts and cache an empty list.
for m in "${MOUNTS[@]}"; do
    for i in {1..20}; do
        if mountpoint -q "$m"; then break; fi
        sleep 0.25
    done
    mountpoint -q "$m" || echo "  WARN: $m never became a mountpoint"
done
echo "  ${#MOUNTS[@]} mounts visible"

# ----- 3. Background pre-warm of pinned models -----
# After the barrier so warm_pinned doesn't see "not visible in mount" for
# freshly-mounted paths.
if [ -n "$MODELS_BUCKET" ] && [ "${#MOUNTS[@]}" -gt 0 ]; then
    (cd /opt/worker && python3 -u -m warm_pinned 2>&1 | sed 's/^/warm: /' &)
fi

# ----- 4. Sentinel: kill the container if any mount-s3 daemon dies -----
# If a mount-s3 process crashes, the mountpoint becomes "Transport endpoint is
# not connected" and ComfyUI gets EIO on every read from that type. Faster to
# fail the worker and let ECS replace it than to try to recover in-place.
if [ "${#MOUNTS[@]}" -gt 0 ]; then
(
    while sleep 30; do
        for m in "${MOUNTS[@]}"; do
            if ! mountpoint -q "$m"; then
                echo "FATAL: mount $m disappeared; killing worker" >&2
                # PID 1 is the entrypoint -> worker.py exec; SIGTERM lets it
                # drain the current SQS message before ECS replaces us.
                kill -TERM 1
                exit 0
            fi
        done
    done
) &
fi

# ----- 5. Hand off to the main worker loop -----
cd /opt/worker
exec python3 -u worker.py
