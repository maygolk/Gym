# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Selected task settings must survive snapshots and reach every shared phase."""

import pytest
from fastapi import HTTPException

from environment_servers.harbor.session import resolve_env_templates
from environment_servers.harbor.tests.conftest import TASK_TOML, make_server, run_episode
from nemo_gym.data.tasks.harbor import load_task


SELECTION = {"GYM_AGENT": "claude_code_agent", "GYM_MODEL": "aws/anthropic/bedrock-claude-haiku-5-5"}
TOML = (
    TASK_TOML
    + """
[environment.env]
GYM_AGENT = "${GYM_AGENT:-codex_agent}"
GYM_MODEL = "${GYM_MODEL:-openai/openai/gpt-5.4-mini}"
[solution.env]
ORACLE_ONLY = "reference"
[verifier.env]
VERIFIER_ONLY = "judge"
"""
)


@pytest.mark.parametrize("implementation", ["test", "harbor_oracle_agent"])
async def test_selected_environment_is_used_by_sandbox_oracle_and_shared_verifier(
    tmp_path, monkeypatch, implementation
):
    # Both real-agent and oracle sessions inherit the selected sandbox settings;
    # phase-specific auth/settings add to them rather than resetting the harness.
    monkeypatch.setenv("GYM_AGENT", "codex_agent")
    monkeypatch.setattr(
        "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
    )
    server, original, sandbox, _ = make_server(
        tmp_path,
        monkeypatch,
        toml=TOML,
        solution=True,
        agent_implementation=implementation,
        task_env=SELECTION,
    )
    task = load_task(original.path, task_env=SELECTION)
    server.config.tasksets["ds"].tasks["hello"] = task.digest
    snapshot = server._snapshot_task(task.path, task.digest)
    assert snapshot.config.environment.env == SELECTION
    spec = server._sandbox_spec(snapshot, "/app")
    assert {key: spec.env[key] for key in SELECTION} == SELECTION
    assert spec.env | resolve_env_templates(snapshot.config.solution.env) == spec.env | {"ORACLE_ONLY": "reference"}
    verifier_env = server._verifier_env(snapshot.config.verifier)
    assert spec.env | verifier_env == spec.env | {"VERIFIER_ONLY": "judge"}
    response = await run_episode(server, task)
    assert response.failure is None
    assert response.result.reward == 1
    if implementation == "harbor_oracle_agent":
        assert server.server_client.seed_bodies()[0]["sandbox_access"]["env"] == {"ORACLE_ONLY": "reference"}


def test_snapshot_rejects_materialized_task_from_a_different_preset(tmp_path, monkeypatch):
    # The same folder with a different selected harness cannot accept cached task identity.
    server, original, _, _ = make_server(tmp_path, monkeypatch, toml=TOML, task_env=SELECTION)
    codex = load_task(original.path, task_env={"GYM_AGENT": "codex_agent", "GYM_MODEL": "mini"})
    with pytest.raises(HTTPException) as error:
        server._snapshot_task(original.path, codex.digest)
    assert error.value.status_code == 409
