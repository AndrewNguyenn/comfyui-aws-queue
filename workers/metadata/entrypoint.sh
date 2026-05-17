#!/usr/bin/env bash
# Metadata-instance entrypoint.
#
# Boots ComfyUI in CPU mode, publishes object_info + extensions to
# dispatcher (so the editor's node menu and Manager UI work), then sits
# idle. Does NOT poll SQS — job processing stays on the GPU workers.
#
# Restarts ComfyUI if it crashes (simple supervisor loop).
set -euo pipefail

echo "comfy-metadata starting on $(hostname)"
echo "  FLEET=${FLEET:-metadata}"
echo "  DISPATCHER_API_URL=${DISPATCHER_API_URL:-unset}"
echo "  FRONTEND_BUCKET=${FRONTEND_BUCKET:-unset}"

export TQDM_DISABLE=1
export PYTHONUNBUFFERED=1

# Set ComfyUI-Manager security_level=weak so /customnode/install accepts
# requests from non-loopback callers (our dispatcher Lambda proxy). The
# Cognito authorizer at API GW is the real gate; Manager's built-in check
# would otherwise block every install with HTTP 403.
mkdir -p /opt/comfy/user/__manager
if [ ! -f /opt/comfy/user/__manager/config.ini ] || \
   ! grep -q '^security_level = weak$' /opt/comfy/user/__manager/config.ini; then
  cat > /opt/comfy/user/__manager/config.ini <<'INI'
[default]
preview_method = none
git_exe =
use_uv = True
channel_url = https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main
share_option = all
bypass_ssl = False
file_logging = True
component_policy = workflow
update_policy = stable-comfyui
windows_selector_event_loop_policy = False
model_download_by_agent = False
downgrade_blacklist =
security_level = weak
always_lazy_install = False
network_mode = public
db_mode = cache
INI
  echo "  wrote Manager config (security_level=weak)"
fi

# Sync custom nodes from the S3 manifest BEFORE starting ComfyUI so the node
# defs surface in /object_info on first publish. Best-effort: a failing node
# logs and we continue with the rest. See workers/shared/manifest_installer.py.
echo "  syncing custom nodes from manifest..."
( cd /opt/worker && python -m manifest_installer 2>&1 ) || \
  echo "  WARN: manifest_installer exited non-zero (continuing)"

# Start ComfyUI in the background, CPU mode, listening on 0.0.0.0 so that
# external traffic DNAT'd through docker's iptables PREROUTING rules
# (host:8188 -> container 172.17.0.2:8188) is actually accepted. With
# --listen 127.0.0.1 ComfyUI only accepts container loopback, breaking
# API GW HTTP_PROXY proxying.
cd /opt/comfy
python main.py --listen 0.0.0.0 --port 8188 --cpu &
COMFY_PID=$!
echo "  comfy pid=$COMFY_PID"

# Wait for /system_stats to respond (means ComfyUI is ready).
for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then
    echo "  comfy ready"
    break
  fi
  sleep 2
done

# Publish object_info + extensions once.
# We use the workers/shared/ modules directly since we already imported them.
cd /opt/worker
python - <<'PY'
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
from comfy_client import ComfyClient
from object_info_publisher import publish_object_info
from extensions_publisher import publish_extensions
import os

fleet = os.environ.get("FLEET", "image")  # Use 'image' so the merged /object_info works
client = ComfyClient("http://127.0.0.1:8188")
try:
    oi = client.fetch_object_info()
    print(f"  fetched object_info: {len(oi)} node classes")
    publish_object_info(fleet, oi)
    print("  ✓ object_info published")
except Exception as e:
    print(f"  ✗ object_info publish failed: {e!r}")

try:
    ok = publish_extensions(fleet)
    print(f"  extensions publish: {'✓' if ok else '✗'}")
except Exception as e:
    print(f"  ✗ extensions publish failed: {e!r}")
PY

echo "  initial publish complete; metadata server now idling"
echo "  (re-publish on container restart; nothing else to do)"

# Keep the foreground process alive so the container doesn't exit.
# If ComfyUI dies, exit so docker restarts us.
wait "$COMFY_PID"
echo "  comfy exited (rc=$?); container will restart via docker --restart policy"
