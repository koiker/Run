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

from unittest.mock import patch

import pytest

from nemo_run.core.execution.kubeflow import KubeflowExecutor, KubeflowJobState
from nemo_run.core.execution.pytorchjob import PyTorchJobExecutor


class TestPyTorchJobExecutor:
    @pytest.fixture
    def mock_k8s_clients(self):
        with (
            patch("nemo_run.core.execution.kubeflow.config.load_kube_config"),
            patch("nemo_run.core.execution.kubeflow.client.CustomObjectsApi") as mock_custom,
            patch("nemo_run.core.execution.kubeflow.client.CoreV1Api") as mock_core,
        ):
            yield mock_custom.return_value, mock_core.return_value

    @pytest.fixture
    def executor(self, mock_k8s_clients):
        return PyTorchJobExecutor(
            image="nvcr.io/nvidia/nemo:26.04",
            num_nodes=2,
            gpus_per_node=8,
            namespace="runai-bench",
            env_vars={"FOO": "bar"},
            # Fields the CSP fabric plugins (e.g. RunAIPlugin) write to:
            extra_resource_requests={"nvidia.com/r0-p0": "1"},
            extra_resource_limits={"nvidia.com/r0-p0": "1"},
            volumes=[{"name": "dshm", "emptyDir": {"medium": "Memory"}}],
            volume_mounts=[{"name": "dshm", "mountPath": "/dev/shm"}],
            pod_annotations={"k8s.v1.cni.cncf.io/networks": "rail0,rail1"},
            pod_labels={"runai/queue": "bench"},
            labels={"app": "bench"},
        )

    # ── CRD identity ──────────────────────────────────────────────────────────

    def test_crd_coordinates_are_v1_pytorchjob(self, executor):
        assert executor.crd_group == "kubeflow.org"
        assert executor.crd_version == "v1"
        assert executor.crd_plural == "pytorchjobs"
        assert executor.crd_kind == "PyTorchJob"

    def test_base_executor_unchanged_v2_defaults(self, mock_k8s_clients):
        """The refactor must leave the v2 TrainJob coordinates as the base default."""
        base = KubeflowExecutor(image="x:1")
        assert base.crd_group == "trainer.kubeflow.org"
        assert base.crd_version == "v1alpha1"
        assert base.crd_plural == "trainjobs"
        assert base.crd_kind == "TrainJob"

    # ── Manifest ──────────────────────────────────────────────────────────────

    def test_get_job_body_is_v1_pytorchjob(self, executor):
        body = executor.get_job_body("demo-job", ["bash", "launch.sh"])
        assert body["apiVersion"] == "kubeflow.org/v1"
        assert body["kind"] == "PyTorchJob"
        assert body["metadata"] == {
            "name": "demo-job",
            "namespace": "runai-bench",
            "labels": {"app": "bench"},
        }
        assert "runtimeRef" not in body["spec"]  # v2-only concept
        assert set(body["spec"]["pytorchReplicaSpecs"]) == {"Master", "Worker"}

    def test_replica_counts(self, executor):
        specs = executor.get_job_body("j", ["c"])["spec"]["pytorchReplicaSpecs"]
        assert specs["Master"]["replicas"] == 1
        assert specs["Worker"]["replicas"] == 1  # num_nodes (2) - 1
        assert specs["Master"]["restartPolicy"] == "OnFailure"

    def test_single_node_has_no_worker(self, mock_k8s_clients):
        e = PyTorchJobExecutor(image="x:1", num_nodes=1, gpus_per_node=8)
        specs = e.get_job_body("j", ["c"])["spec"]["pytorchReplicaSpecs"]
        assert set(specs) == {"Master"}

    def test_container_named_pytorch_with_command_and_env(self, executor):
        master = executor.get_job_body("j", ["bash", "launch.sh"])["spec"][
            "pytorchReplicaSpecs"
        ]["Master"]
        container = master["template"]["spec"]["containers"][0]
        assert container["name"] == "pytorch"
        assert container["image"] == "nvcr.io/nvidia/nemo:26.04"
        assert container["command"] == ["bash", "launch.sh"]
        env = {e["name"]: e["value"] for e in container["env"]}
        assert env["FOO"] == "bar"

    def test_nproc_per_node_set_on_spec(self, executor):
        # spec.nprocPerNode (string) makes the v1 operator export PET_NPROC_PER_NODE
        # so the inherited PET_* macro_values resolve (verified on-cluster).
        spec = executor.get_job_body("j", ["c"])["spec"]
        assert spec["nprocPerNode"] == "8"

    def test_extended_resources_and_gpu_in_container_resources(self, executor):
        container = executor.get_job_body("j", ["c"])["spec"]["pytorchReplicaSpecs"][
            "Master"
        ]["template"]["spec"]["containers"][0]
        res = container["resources"]
        assert res["limits"]["nvidia.com/gpu"] == "8"
        assert res["requests"]["nvidia.com/gpu"] == "8"
        # RunAIPlugin RoCE rails layered via extra_resource_requests/limits
        assert res["requests"]["nvidia.com/r0-p0"] == "1"
        assert res["limits"]["nvidia.com/r0-p0"] == "1"

    def test_pod_template_volumes_and_metadata(self, executor):
        tmpl = executor.get_job_body("j", ["c"])["spec"]["pytorchReplicaSpecs"]["Master"][
            "template"
        ]
        assert tmpl["spec"]["volumes"] == [{"name": "dshm", "emptyDir": {"medium": "Memory"}}]
        assert tmpl["spec"]["containers"][0]["volumeMounts"] == [
            {"name": "dshm", "mountPath": "/dev/shm"}
        ]
        # Multus annotation + Run:ai queue label land on the pod template metadata.
        assert tmpl["metadata"]["annotations"] == {"k8s.v1.cni.cncf.io/networks": "rail0,rail1"}
        assert tmpl["metadata"]["labels"] == {"runai/queue": "bench"}

    def test_master_and_worker_templates_are_independent(self, executor):
        specs = executor.get_job_body("j", ["c"])["spec"]["pytorchReplicaSpecs"]
        assert specs["Master"]["template"] is not specs["Worker"]["template"]

    # ── Distributed-launch contract ───────────────────────────────────────────

    def test_macro_values_inherited_pet_contract(self, executor):
        # With spec.nprocPerNode set, the v1 operator exports the same PET_* env
        # as the v2 TrainJob, so v1 inherits KubeflowExecutor.macro_values as-is.
        macros = executor.macro_values()
        assert macros.head_node_ip_var == "PET_MASTER_ADDR"
        assert macros.num_nodes_var == "PET_NNODES"
        assert macros.node_rank_var == "PET_NODE_RANK"
        assert macros.nproc_per_node_var == "PET_NPROC_PER_NODE"

    # ── Status / selectors ────────────────────────────────────────────────────

    def test_pod_label_selector(self, executor):
        assert executor._pod_label_selector("demo") == "training.kubeflow.org/job-name=demo"

    @pytest.mark.parametrize(
        "conditions,expected",
        [
            ([{"type": "Running", "status": "True"}], KubeflowJobState.RUNNING),
            ([{"type": "Succeeded", "status": "True"}], KubeflowJobState.SUCCEEDED),
            (
                [{"type": "Running", "status": "False"}, {"type": "Failed", "status": "True"}],
                KubeflowJobState.FAILED,
            ),
            ([{"type": "Created", "status": "True"}], KubeflowJobState.CREATED),
            ([], KubeflowJobState.UNKNOWN),
        ],
    )
    def test_parse_status_from_conditions(self, executor, conditions, expected):
        assert executor._parse_status({"conditions": conditions}) == expected
