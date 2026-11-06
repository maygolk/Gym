# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Check retained-session and oracle evidence using temporary files only."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from .test_gym_fm_02_verifier import evidence_reader
from .test_gym_verifier import verifier, version_output


def test_session_version_polling_keeps_completion_and_source():
    # A running version command and its poll must remain one completed execution.
    with TemporaryDirectory() as directory:
        path = Path(directory) / "session.jsonl"
        events = [
            {
                "type": "function_call",
                "name": "exec_command",
                "call_id": "start",
                "arguments": '{"cmd":"gym --version"}',
            },
            {
                "type": "function_call_output",
                "call_id": "start",
                "output": "Process running with session ID 42\nOutput:\n" + version_output(),
            },
            {"type": "function_call", "name": "write_stdin", "call_id": "poll", "arguments": '{"session_id":42}'},
            {"type": "function_call_output", "call_id": "poll", "output": "Process exited with code 0\nOutput:\n"},
        ]
        path.write_text("\n".join(json.dumps({"type": "response_item", "payload": item}) for item in events))
        trajectory = verifier.session_trajectory([path])
        candidates = list(verifier.find_version_checks(trajectory))
        assert len(candidates) == 1
        assert candidates[0]["evidence"]["exit_code"] == 0
        assert candidates[0]["evidence"]["observations"][0]["source_file"] == str(path)
        # An uncompleted poll cannot establish completion.
        trajectory["steps"][1].pop("observation")
        with pytest.raises(verifier.EvidenceError):
            list(verifier.find_version_checks(trajectory))


def test_oracle_header_does_not_hide_real_execution_log():
    # The native oracle exports message-only ATIF; the judge still needs its log.
    with TemporaryDirectory() as directory:
        root = Path(directory)
        agent = root / "agent"
        agent.mkdir()
        (agent / "trajectory.json").write_text(json.dumps({"session_id": "oracle", "steps": []}))
        (agent / "oracle.txt").write_text("Real reference execution output\n")
        (agent / "exit-code.txt").write_text("0\n")
        solution = root / "solve.sh"
        solution.write_text("gym eval run\n")
        reference = json.loads((Path(evidence_reader.__file__).with_name("reference.json")).read_text())
        reference["bundled_output_hashes"] = []
        evidence = evidence_reader.EvidenceFiles(
            reference,
            workspace=root,
            agent_root=agent,
            solution=solution,
            window={"container_started_at": 1, "verifier_started_at": 2},
        )
        paths = {str(item["path"]) for item in evidence.files.values() if item["kind"] == "oracle"}
        assert str(agent / "oracle.txt") in paths
        assert str(agent / "exit-code.txt") in paths
        assert str(solution) in paths
