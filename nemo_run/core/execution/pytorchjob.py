# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Kubeflow Training-Operator **v1** ``PyTorchJob`` executor.

``KubeflowExecutor`` targets the Training Operator **v2** ``TrainJob``
(``trainer.kubeflow.org``), which expands a ``ClusterTrainingRuntime`` into a
JobSet. Many clusters — notably **NVIDIA Run:ai** — run the **v1** operator,
whose API is the self-contained ``kubeflow.org/v1`` ``PyTorchJob`` (Master/Worker
``pytorchReplicaSpecs`` with an embedded pod template, no runtime reference).

``PyTorchJobExecutor`` is a thin subclass that retargets only the version-specific
seams exposed by ``KubeflowExecutor`` — the CRD coordinates, the manifest body
(:meth:`get_job_body`), the status parse (:meth:`_parse_status`), the pod label
selector / rank label, and the distributed-launch env-var contract
(:meth:`macro_values`). Everything else (packaging / data-mover, log streaming,
pod ops, cert-rotation reload, and all the executor fields the CSP fabric plugins
write to) is inherited unchanged, so e.g. ``RunAIPlugin`` works against this
executor with no changes (it still ``isinstance``-matches ``KubeflowExecutor``).
"""

import copy
from dataclasses import dataclass
from typing import Any, ClassVar

from nemo_run.core.execution.kubeflow import KubeflowExecutor, KubeflowJobState


@dataclass(kw_only=True)
class PyTorchJobExecutor(KubeflowExecutor):
    """Executor for the Kubeflow Training-Operator v1 ``PyTorchJob``.

    Drop-in alternative to :class:`KubeflowExecutor` for clusters running the v1
    operator (e.g. Run:ai). Same configuration surface; the difference is the
    custom resource it submits.

    Distributed launch: when ``spec.nprocPerNode`` is set (this executor sets it
    from :meth:`nproc_per_node`), the v1 operator runs each replica under torch
    elastic and injects the same ``PET_*`` rendezvous env the v2 TrainJob uses
    (``PET_MASTER_ADDR`` / ``PET_MASTER_PORT`` / ``PET_NNODES`` /
    ``PET_NODE_RANK`` / ``PET_NPROC_PER_NODE``). It therefore inherits
    :meth:`KubeflowExecutor.macro_values` unchanged — verified against the
    cluster operator, which also sets the classic ``MASTER_ADDR`` / ``WORLD_SIZE``
    / ``RANK`` as a fallback.
    """

    # v1 PyTorchJob CRD coordinates (override the v2 TrainJob defaults).
    crd_group: ClassVar[str] = "kubeflow.org"
    crd_version: ClassVar[str] = "v1"
    crd_plural: ClassVar[str] = "pytorchjobs"
    crd_kind: ClassVar[str] = "PyTorchJob"

    # The v1 operator only injects env into a container named "pytorch".
    _CONTAINER_NAME: ClassVar[str] = "pytorch"

    def _pod_label_selector(self, job_name: str) -> str:
        """v1 PyTorchJob pods carry the operator's job-name label."""
        return f"training.kubeflow.org/job-name={job_name}"

    def _completion_index_label(self) -> str:
        """Best-effort node-rank label for early log routing.

        The v1 operator labels pods with their replica index; the authoritative
        global rank is still resolved from ``GROUP_RANK`` once workers are up
        (see ``KubeflowExecutor.fetch_logs``), so this only affects pre-rendezvous
        fallback routing.
        """
        return "training.kubeflow.org/replica-index"

    def _parse_status(self, job_status: dict) -> KubeflowJobState:
        """Map ``PyTorchJob.status.conditions[]`` to a ``KubeflowJobState``.

        v1 reports state via ``conditions`` (type in Created/Running/Succeeded/
        Failed/Restarting, with ``status == "True"`` for the active one), unlike
        the v2 TrainJob's ``jobsStatus[]`` counts.
        """
        conditions = job_status.get("conditions", []) or []
        active = {c.get("type") for c in conditions if c.get("status") == "True"}
        if "Failed" in active:
            return KubeflowJobState.FAILED
        if "Succeeded" in active:
            return KubeflowJobState.SUCCEEDED
        if "Running" in active or "Restarting" in active:
            return KubeflowJobState.RUNNING
        if "Created" in active:
            return KubeflowJobState.CREATED
        return KubeflowJobState.UNKNOWN

    def _replica_template(self, command: list[str], env: list[dict[str, Any]]) -> dict[str, Any]:
        """Build the shared pod template used by both Master and Worker replicas."""
        resources = self._build_resources()
        container: dict[str, Any] = {
            "name": self._CONTAINER_NAME,
            "image": self.image,
            "command": command,
            "env": env,
        }
        if resources:
            container["resources"] = resources
        if self.volume_mounts:
            container["volumeMounts"] = self.volume_mounts
        container.update(self.container_kwargs)

        pod_spec: dict[str, Any] = {"containers": [container]}
        if self.volumes:
            pod_spec["volumes"] = self.volumes
        if self.tolerations:
            pod_spec["tolerations"] = self.tolerations
        if self.affinity:
            pod_spec["affinity"] = self.affinity
        if self.image_pull_secrets:
            pod_spec["imagePullSecrets"] = [{"name": s} for s in self.image_pull_secrets]
        pod_spec.update(self.pod_spec_overrides)

        template: dict[str, Any] = {"spec": pod_spec}
        pod_meta: dict[str, Any] = {}
        if self.pod_labels:
            pod_meta["labels"] = self.pod_labels
        if self.pod_annotations:
            pod_meta["annotations"] = self.pod_annotations
        if pod_meta:
            template["metadata"] = pod_meta
        return template

    def get_job_body(self, name: str, command: list[str]) -> dict:
        """Build and return the v1 ``PyTorchJob`` CRD manifest dict."""
        env = [{"name": k, "value": v} for k, v in self.env_vars.items()] + self.env_list

        template = self._replica_template(command, env)

        def _replica_spec(replicas: int) -> dict[str, Any]:
            return {
                "replicas": replicas,
                "restartPolicy": self.restart_policy,
                "template": copy.deepcopy(template),
            }

        # Master is rank 0 (1 pod); the remaining nodes are Workers.
        replica_specs: dict[str, Any] = {"Master": _replica_spec(1)}
        if self.num_nodes > 1:
            replica_specs["Worker"] = _replica_spec(self.num_nodes - 1)

        # nprocPerNode makes the operator launch each replica under torch elastic
        # and export the PET_* rendezvous env (incl. PET_NPROC_PER_NODE) that the
        # inherited macro_values resolves; it is a string per the CRD schema.
        spec: dict[str, Any] = {
            "nprocPerNode": str(self.nproc_per_node()),
            "pytorchReplicaSpecs": replica_specs,
        }
        spec.update(self.spec_kwargs)

        metadata: dict[str, Any] = {"name": name, "namespace": self.namespace}
        if self.labels:
            metadata["labels"] = self.labels
        if self.annotations:
            metadata["annotations"] = self.annotations

        return {
            "apiVersion": f"{self.crd_group}/{self.crd_version}",
            "kind": self.crd_kind,
            "metadata": metadata,
            "spec": spec,
        }
