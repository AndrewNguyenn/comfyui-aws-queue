"""Model affinity: which of a few received jobs a worker should run.

Switching checkpoint/unet costs an offload and a reload, so a worker pulls a
small batch and runs the one for the model it already has loaded. The pick is
pure (no AWS) so it can be tested.

Rules, in order:
  - nothing loaded yet, or no job here for the loaded model: the oldest job
    (a free worker never sits on work just because it is another model's);
  - the worker has run RUN_MAX jobs in a row on its model while another
    model's job is waiting here: the oldest of those (bounds how long the
    minority model can wait, whatever the backlog);
  - otherwise the oldest job for the loaded model.

A job whose model is unknown ('') runs anywhere and never changes the loaded
model. Affinity is on the checkpoint/unet only: LoRA patches are cheap and
their sets vary per batch, so keying on them would fragment the match.
"""
from __future__ import annotations

RUN_MAX = 10


def pick(models: list[str], sent: list[int], loaded: str | None, run: int) -> int:
    """Index of the job to take. `models[i]` is job i's model, `sent[i]` its
    SQS SentTimestamp, `loaded` the worker's model (None before its first job),
    `run` how many jobs in a row it has run on `loaded`."""
    order = sorted(range(len(models)), key=lambda i: sent[i])
    match = [i for i in order if models[i] in ("", loaded)]
    other = [i for i in order if i not in match]
    if loaded is None or not match:
        return order[0]
    if other and run >= RUN_MAX:
        return other[0]
    return match[0]
