"""Cost kill-switch — hard-stops all GPU compute when the monthly budget is
breached.

Wired to the `comfy-killswitch` SNS topic, which the AWS Budget's 100% ACTUAL
notification publishes to (see infra/lib/stacks/monitoring.ts). ANY SNS delivery
is treated as "fire" — the message body is not inspected.

What it does, for both fleets (image + video):
  1. ASG -> MinSize=0, MaxSize=0, DesiredCapacity=0. **MaxSize=0 is the only
     durable hard-stop**: the ECS capacity provider's managed scaling drives the
     ASG's DesiredCapacity but is hard-capped by MaxSize, so it cannot launch or
     relaunch instances. Neither the per-fleet scalers nor App Auto Scaling touch
     the ASG min/max, so this holds.
  2. ECS service -> desiredCount=0. This is only a transient courtesy that stops
     tasks being scheduled in the moment — it is NOT belt-and-suspenders: the
     comfy-fleet-scaler Lambda (every 60s, both fleets) re-raises desiredCount
     above 0 within ~a minute if a fleet's queue is non-empty.
     That resurrected "want" is harmless — MaxSize=0 means it can never be
     satisfied, so no instance launches (the steady state is just the service
     wanting tasks the capacity provider can't place; no cost).
Running instances drain via the ASG lifecycle hook and terminate; in-flight jobs
on them are lost (that's the point of an emergency stop). To suspend the scaler
churn entirely (cleaner but not required for the hold), a future version could
also disable the comfy-fleet-scaler EventBridge rule on fire.

SAFE TESTING: invoke directly with {"dry_run": true} (or set env DRY_RUN=1) to
log what it WOULD change (incl. each ASG's current min/max/desired) WITHOUT
touching anything. An SNS-triggered invocation has no dry_run flag, so a real
budget breach always fires for real.

RECOVERY (after a real fire): the ASGs are left at min/max/desired=0, which is
DRIFT from CloudFormation. To bring compute back, either redeploy the compute
stack (`cd infra && npx cdk deploy ComfyComputeStack -c useGoldenAmi=true` —
resets the ASG bounds from config) or manually restore, e.g.
`aws autoscaling update-auto-scaling-group --auto-scaling-group-name \
comfy-image-asg --min-size 0 --max-size 6 --desired-capacity 0` (video max 3),
then let the ECS services scale back up.
"""
import json
import os

import boto3

asg = boto3.client("autoscaling")
ecs = boto3.client("ecs")

# Comma-separated literals supplied by CDK (deterministic names).
ASG_NAMES = [s for s in os.environ.get("ASG_NAMES", "").split(",") if s]
ECS_CLUSTER = os.environ.get("ECS_CLUSTER", "")
ECS_SERVICES = [s for s in os.environ.get("ECS_SERVICES", "").split(",") if s]


def _dry_run(event) -> bool:
    if str(os.environ.get("DRY_RUN", "")).strip().lower() in ("1", "true", "yes"):
        return True
    return bool(isinstance(event, dict) and event.get("dry_run"))


def lambda_handler(event, _context):
    dry = _dry_run(event)
    actions: list = []

    # 1. Hard-stop both ASGs (min=max=desired=0).
    for name in ASG_NAMES:
        before = None
        try:
            groups = asg.describe_auto_scaling_groups(
                AutoScalingGroupNames=[name]
            ).get("AutoScalingGroups", [])
            if groups:
                g = groups[0]
                before = {"min": g["MinSize"], "max": g["MaxSize"],
                          "desired": g["DesiredCapacity"]}
        except Exception as e:  # noqa: BLE001
            before = f"describe-failed: {e!r}"
        if not dry:
            try:
                asg.update_auto_scaling_group(
                    AutoScalingGroupName=name,
                    MinSize=0, MaxSize=0, DesiredCapacity=0,
                )
            except Exception as e:  # noqa: BLE001
                actions.append({"asg": name, "error": repr(e)})
                continue
        actions.append({"asg": name, "before": before, "set": "min=max=desired=0"})

    # 2. Zero the ECS services so the capacity provider stops requesting tasks.
    for svc in ECS_SERVICES:
        if not dry:
            try:
                ecs.update_service(cluster=ECS_CLUSTER, service=svc, desiredCount=0)
            except Exception as e:  # noqa: BLE001
                actions.append({"service": svc, "error": repr(e)})
                continue
        actions.append({"service": svc, "set": "desiredCount=0"})

    result = {
        "killswitch": "DRY_RUN" if dry else "FIRED",
        "cluster": ECS_CLUSTER,
        "actions": actions,
    }
    print(json.dumps(result))
    return result
