"""Publish the golden AMI id to SSM when an EC2 Image Builder bake completes.

Triggered by the EventBridge rule on "EC2 Image Builder Image State Change" →
status AVAILABLE (see infra/lib/stacks/imagebuilder.ts). Replaces Image Builder's
native ssmParameterConfigurations distribution write, which the AWS-managed
service-linked role (AWSServiceRoleForImageBuilder) cannot perform — it has no
ssm:PutParameter permission and can't be granted one. This Lambda writes the
parameter under its own role, which we control.

The image-fleet launch template reads the parameter via
MachineImage.fromSsmParameter (deploy-time), so a fresh AMI only lands on the
next `cdk deploy` — never surprise-rotating a running worker.
"""

import os

import boto3

ssm = boto3.client("ssm")
imagebuilder = boto3.client("imagebuilder")

PARAM = os.environ["GOLDEN_AMI_PARAM"]
RECIPE_NAME = os.environ.get("RECIPE_NAME", "comfy-image-golden")


def lambda_handler(event, _context):
    # The "EC2 Image Builder Image State Change" event carries the build-version
    # ARN at the TOP LEVEL (event["resources"][0]) — NOT inside detail (detail
    # only holds {state:{status}}). Verified against AWS's reference handler.
    resources = event.get("resources") or []
    arn = resources[0] if resources else None
    if not arn:
        print(f"no image arn in event, skipping: {event!r}")
        return

    image = imagebuilder.get_image(imageBuildVersionArn=arn).get("image", {})

    # Guard: only publish for OUR recipe (the rule is account-wide for imagebuilder).
    # Exact match (not substring) so a future `comfy-image-golden-*` recipe can't
    # hijack the param.
    recipe_name = (image.get("imageRecipe", {}) or {}).get("name", "")
    if recipe_name != RECIPE_NAME:
        print(f"not the golden recipe (recipe={recipe_name!r}, arn={arn}); skipping")
        return

    amis = (image.get("outputResources", {}) or {}).get("amis", [])
    if not amis:
        print(f"no AMIs in outputResources for {arn}; skipping")
        return

    ami_id = amis[0]["image"]
    ssm.put_parameter(Name=PARAM, Type="String", Value=ami_id, Overwrite=True)
    print(f"published golden AMI {ami_id} -> {PARAM}")
