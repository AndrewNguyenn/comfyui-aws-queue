"""Unit tests for the cost kill-switch handler.

Run: python3 services/killswitch/test_handler.py
boto3 is stubbed (and the clients replaced with mocks) so it needs no AWS.
"""
import importlib.util
import os
import sys
import types
from unittest.mock import MagicMock

sys.modules.setdefault("boto3", types.SimpleNamespace(client=lambda *a, **k: MagicMock()))

# Read at import time by the handler — must be set BEFORE loading it.
os.environ["ASG_NAMES"] = "comfy-image-asg,comfy-video-asg"
os.environ["ECS_CLUSTER"] = "comfyui-aws-queue-cluster"
os.environ["ECS_SERVICES"] = "comfy-image,comfy-video"

_spec = importlib.util.spec_from_file_location(
    "ks", os.path.join(os.path.dirname(__file__), "handler.py"))
ks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ks)


def _fresh_mocks():
    ks.asg = MagicMock()
    ks.ecs = MagicMock()
    ks.asg.describe_auto_scaling_groups.return_value = {
        "AutoScalingGroups": [{"MinSize": 0, "MaxSize": 6, "DesiredCapacity": 5,
                               "Instances": [{"InstanceId": "i-busy"}, {"InstanceId": "i-idle"}]}]
    }


def test_dry_run_changes_nothing():
    _fresh_mocks()
    out = ks.lambda_handler({"dry_run": True}, None)
    assert out["killswitch"] == "DRY_RUN"
    ks.asg.update_auto_scaling_group.assert_not_called()
    ks.ecs.update_service.assert_not_called()
    # still describes both ASGs to report current state
    assert ks.asg.describe_auto_scaling_groups.call_count == 2


def test_fire_zeros_all_asgs_and_services():
    _fresh_mocks()
    out = ks.lambda_handler({"Records": [{"Sns": {"Message": "budget breached"}}]}, None)
    assert out["killswitch"] == "FIRED"
    asg_calls = ks.asg.update_auto_scaling_group.call_args_list
    assert len(asg_calls) == 2
    for c in asg_calls:
        assert c.kwargs["MinSize"] == 0
        assert c.kwargs["MaxSize"] == 0
        assert c.kwargs["DesiredCapacity"] == 0
    assert {c.kwargs["AutoScalingGroupName"] for c in asg_calls} == {
        "comfy-image-asg", "comfy-video-asg"}
    # Scale-in protection is cleared on every instance BEFORE the group is
    # zeroed, or a protected mid-render worker outlives the budget breach.
    prot_calls = ks.asg.set_instance_protection.call_args_list
    assert len(prot_calls) == 2
    for c in prot_calls:
        assert c.kwargs["ProtectedFromScaleIn"] is False
        assert sorted(c.kwargs["InstanceIds"]) == ["i-busy", "i-idle"]
    names = [c[0] for c in ks.asg.mock_calls]
    assert names.index("set_instance_protection") < names.index("update_auto_scaling_group")
    svc_calls = ks.ecs.update_service.call_args_list
    assert len(svc_calls) == 2
    for c in svc_calls:
        assert c.kwargs["cluster"] == "comfyui-aws-queue-cluster"
        assert c.kwargs["desiredCount"] == 0
    assert {c.kwargs["service"] for c in svc_calls} == {"comfy-image", "comfy-video"}


def test_env_dry_run_forces_safe_even_without_event_flag():
    _fresh_mocks()
    os.environ["DRY_RUN"] = "1"
    try:
        out = ks.lambda_handler({}, None)  # no event flag, env forces dry run
        assert out["killswitch"] == "DRY_RUN"
        ks.asg.update_auto_scaling_group.assert_not_called()
        ks.ecs.update_service.assert_not_called()
    finally:
        del os.environ["DRY_RUN"]


def test_asg_update_failure_is_isolated_not_fatal():
    _fresh_mocks()
    ks.asg.update_auto_scaling_group.side_effect = [Exception("boom"), None]
    out = ks.lambda_handler({}, None)  # real fire
    assert out["killswitch"] == "FIRED"
    # one error recorded, the other ASG + both services still attempted
    errs = [a for a in out["actions"] if "error" in a]
    assert len(errs) == 1
    assert ks.ecs.update_service.call_count == 2


if __name__ == "__main__":
    import traceback
    fails = 0
    for _n, _f in sorted(globals().items()):
        if _n.startswith("test_") and callable(_f):
            try:
                _f()
                print("ok  ", _n)
            except Exception:  # noqa: BLE001
                fails += 1
                print("FAIL", _n)
                traceback.print_exc()
    print(f"\n{'PASS' if not fails else 'FAIL'} ({fails} failed)")
    sys.exit(1 if fails else 0)
