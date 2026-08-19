"""Celery application — Redis as broker (and result backend).

Run:  celery -A worker.celery_app.celery worker --loglevel=info
"""
import os
import sys
from pathlib import Path

# Two path problems to solve:
#  1. Celery's CLI imports this app with the CWD only TEMPORARILY on sys.path
#     (cwd_in_path), so lazy imports inside tasks (e.g. `from agent.graph
#     import run_agent`) fail in the MAIN process after boot.
#  2. On macOS/Python 3.13 the prefork pool SPAWNS fresh interpreters (not
#     fork), which rebuild sys.path from scratch — in-memory pinning done here
#     does not reach them. PYTHONPATH is read at interpreter startup, so
#     exporting it is the only fix that reaches spawned pool children.
# Insert UNCONDITIONALLY: at import time Celery's cwd_in_path has temporarily
# added the CWD (same string), so a "not in sys.path" guard would skip the
# insert — and Celery then removes its copy on exit, leaving no root at all.
# With two copies present, Celery's cleanup removes one and ours survives.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _PROJECT_ROOT)
if _PROJECT_ROOT not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    os.environ["PYTHONPATH"] = (
        _PROJECT_ROOT + os.pathsep + os.environ["PYTHONPATH"]
        if os.environ.get("PYTHONPATH")
        else _PROJECT_ROOT
    )

from celery import Celery

from core.config import settings

celery = Celery(
    "egov_rag",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["worker.tasks"],
)

celery.conf.update(
    task_acks_late=True,             # ack after completion; survive worker crash
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,    # fair dispatch for slow LLM/ASR tasks
    task_track_started=True,
    task_time_limit=300,             # hard ceiling per task (seconds)
    task_soft_time_limit=270,        # soft limit -> graceful cleanup
    broker_connection_retry_on_startup=True,
)
