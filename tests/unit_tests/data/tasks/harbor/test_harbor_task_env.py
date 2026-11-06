# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Run presets must select the same environment during preparation and seed."""

import pytest
import yaml

from nemo_gym.cli.eval import task_env_from_overrides
from nemo_gym.data.tasks.harbor.build_options import BuildOptions
from nemo_gym.data.tasks.harbor.errors import HarborTaskError
from nemo_gym.data.tasks.harbor.prepare import prepare_target
from nemo_gym.data.tasks.harbor.task import discover_tasks, load_task
from tests.unit_tests.data.tasks.harbor.test_harbor_tasks import HELLO_TOML, write_image_config, write_task


CODEX = {"GYM_AGENT": "codex_agent", "GYM_MODEL": "openai/openai/gpt-5.4-mini"}
CLAUDE = {"GYM_AGENT": "claude_code_agent", "GYM_MODEL": "aws/anthropic/bedrock-claude-haiku-5-5"}
TOML = (
    HELLO_TOML
    + """
[environment.env]
GYM_AGENT = "${GYM_AGENT:-codex_agent}"
GYM_MODEL = "${GYM_MODEL:-openai/openai/gpt-5.4-mini}"
TOKEN = "${TEST_RUNTIME_KEY}"
"""
)


def test_task_env_changes_effective_identity_without_editing_files(tmp_path, monkeypatch):
    # A preset wins over stale terminal selections and leaves unrelated auth templated.
    monkeypatch.setenv("GYM_AGENT", "stale-terminal-agent")
    folder = write_task(tmp_path / "hello", toml=TOML)
    write_image_config(tmp_path, "ubuntu:24.04")
    original = (folder / "task.toml").read_bytes()
    baseline = load_task(folder)
    codex = load_task(folder, task_env=CODEX)
    claude = load_task(folder, task_env=CLAUDE)
    assert codex.config.environment.env == CODEX | {"TOKEN": "${TEST_RUNTIME_KEY}"}
    assert claude.env["GYM_AGENT"] == "claude_code_agent"
    assert len({baseline.digest, codex.digest, claude.digest}) == 3
    assert baseline.folder_digest == codex.folder_digest == claude.folder_digest
    assert (folder / "task.toml").read_bytes() == original
    assert discover_tasks(folder, task_env=CLAUDE)[0].digest == claude.digest
    assert discover_tasks(tmp_path, task_env=CLAUDE)[0].digest == claude.digest
    assert load_task(folder, task_env={"UNDECLARED": "ignored"}).digest == baseline.digest
    assert load_task(folder, task_env=dict(reversed(list(CLAUDE.items())))).digest == claude.digest


def test_prepare_and_reloaded_task_use_same_selection(tmp_path, fake_registry, monkeypatch):
    # Preparation uses the same effective task as a later server snapshot load.
    monkeypatch.setenv("TEST_RUNTIME_KEY", "synthetic-runtime-key")
    folder = write_task(tmp_path / "dataset" / "hello", toml=TOML)
    write_image_config(folder.parent, "ubuntu:24.04")
    prepared = prepare_target(
        str(folder), output_root=tmp_path / "results", build=BuildOptions(enabled=False), task_env=CLAUDE
    )
    assert len(prepared.tasks) == 1
    assert prepared.tasks[0].digest == load_task(folder, task_env=CLAUDE).digest
    assert prepared.tasks[0].config.environment.env["GYM_MODEL"] == CLAUDE["GYM_MODEL"]


def test_compose_overrides_are_refused_instead_of_using_wrong_build_settings(tmp_path):
    # Runtime presets do not pretend to alter Compose's independently planned builds.
    folder = write_task(tmp_path / "hello", toml=TOML)
    (folder / "environment" / "docker-compose.yaml").write_text("services: {main: {image: ubuntu:24.04}}\n")
    with pytest.raises(HarborTaskError, match="without Compose"):
        load_task(folder, task_env=CLAUDE)


def test_cli_reads_nested_preset_without_resolving_unrelated_credentials(tmp_path, monkeypatch):
    # The child config can contain an unset key; only the task selection is needed here.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("UNSET_PRESET_TEST_KEY", raising=False)
    child = tmp_path / "shared.yaml"
    child.write_text("policy_api_key: ${oc.env:UNSET_PRESET_TEST_KEY}\n")
    preset = tmp_path / "preset.yaml"
    preset.write_text(
        yaml.safe_dump(
            {
                "config_paths": [str(child)],
                "harbor_environment_server": {"environment_servers": {"harbor": {"task_env": CLAUDE}}},
            }
        )
    )
    assert task_env_from_overrides([f"+config_paths=[{preset}]"]) == CLAUDE
    assert task_env_from_overrides([]) == {}
    preset.write_text("harbor_environment_server: {environment_servers: {harbor: {task_env: {GYM_MODEL: 3}}}}\n")
    with pytest.raises(ValueError, match="mapping of variable names to strings"):
        task_env_from_overrides([f"+config_paths=[{preset}]"])
