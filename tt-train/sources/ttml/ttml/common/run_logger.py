# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Metrics sink for training scripts: a CSV file and, optionally, a wandb run.

Usage::

    with RunLogger(csv_path="metrics.csv", columns=["train/loss", "test/acc"],
                   wandb_project="my-project", config=vars(args)) as logger:
        logger.log({"train/loss": 2.3}, step=1)
        logger.summary(n_params=5.4e6)

wandb settings that are not arguments here (entity, mode, dir, API key) come from the usual
``WANDB_*`` environment variables. On machines whose compute nodes have no internet access, set
``WANDB_MODE=offline`` and run ``wandb sync`` later from a node that has it.
"""

from __future__ import annotations

import csv
import os
import socket
from pathlib import Path
from typing import Any, Iterable, Mapping

# Recorded in the wandb config so a run can be traced back to its batch job and devices.
_ENV_KEYS = ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_CLUSTER_NAME", "TT_VISIBLE_DEVICES")


class RunLogger:
    """Writes each :meth:`log` call as one CSV row and one wandb step.

    Args:
        csv_path: CSV file to write, or ``None`` for no CSV. The file is line-buffered, so rows
            written before a crash or a Slurm timeout are kept.
        columns: CSV columns after ``step``. Keys that a row does not contain are left empty.
            Keys not in ``columns`` go to wandb only.
        wandb_project: Enables wandb when set. Fails here if ``wandb`` is not installed, so a
            long job does not run without the logging it was asked for.
        run_name: wandb run name; ``None`` lets wandb pick one.
        config: Hyperparameters stored with the wandb run.
        tags: wandb tags.
    """

    def __init__(
        self,
        csv_path: str | os.PathLike | None = None,
        columns: Iterable[str] = (),
        wandb_project: str | None = None,
        run_name: str | None = None,
        config: Mapping[str, Any] | None = None,
        tags: Iterable[str] | None = None,
    ) -> None:
        self._csv_file = None
        self._csv = None
        if csv_path is not None:
            path = Path(csv_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = open(path, "w", newline="", buffering=1)
            self._csv = csv.DictWriter(self._csv_file, fieldnames=["step", *columns], extrasaction="ignore")
            self._csv.writeheader()

        self._wandb_run = None
        if wandb_project is not None:
            import wandb

            run_config = {k: _jsonable(v) for k, v in (config or {}).items()}
            run_config["host"] = socket.gethostname()
            run_config.update({k.lower(): os.environ[k] for k in _ENV_KEYS if k in os.environ})
            self._wandb_run = wandb.init(
                project=wandb_project, name=run_name, config=run_config, tags=list(tags) if tags else None
            )

    def log(self, metrics: Mapping[str, Any], step: int) -> None:
        if self._csv is not None:
            self._csv.writerow({"step": step, **metrics})
        if self._wandb_run is not None:
            self._wandb_run.log(dict(metrics), step=step)

    def summary(self, **values: Any) -> None:
        """Set run-level values (shown in the wandb runs table). No-op without wandb."""
        if self._wandb_run is not None:
            self._wandb_run.summary.update(values)

    def close(self, exit_code: int = 0) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = self._csv = None
        if self._wandb_run is not None:
            self._wandb_run.finish(exit_code=exit_code)
            self._wandb_run = None

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(exit_code=0 if exc_type is None else 1)


def _jsonable(value: Any) -> Any:
    return str(value) if isinstance(value, os.PathLike) else value
