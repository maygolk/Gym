# Source: https://gitlab-master.nvidia.com/hwolff/pare-harbor-tasks
# Revision: 7acf882c86a724a67075ae729608500e8a95e0a7
# Original file: tests/test_gym_fm_02_hybrid.py
# Source license: not declared in source file; contribution clearance pending
"""Check the LLM judge contract, file access and mocked requests without running Gym."""

import importlib.util
import json
import sys
import tomllib
import unittest
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError


TASK_TESTS = Path(__file__).resolve().parents[4] / "environments/pare/gym-fm-02/tests"
sys.path.insert(0, str(TASK_TESTS))
try:
    import hybrid
    import read_execution
finally:
    sys.path.pop(0)
SPEC = importlib.util.spec_from_file_location("fm02_hybrid_entry", TASK_TESTS / "verify.py")
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


class HybridVerifierTests(unittest.TestCase):
    def setUp(self):
        # Create a scored-zero output and public execution records in temporary storage.
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.workspace, self.agent = self.root / "workspace", self.root / "agent"
        self.workspace.mkdir()
        self.agent.mkdir()
        self.expected = json.loads((TASK_TESTS / "reference.json").read_text())
        self.expected["bundled_output_hashes"] = []
        task_config = tomllib.loads((TASK_TESTS.parent / "task.toml").read_text())
        self.verifier_env = task_config["verifier"]["env"]
        # Emulate the runner's default expansion without touching real credentials.
        self.gym_env = {"GYM_AGENT": "codex_agent", "GYM_MODEL": "openai/openai/gpt-5.4-mini"}
        gym_settings = patch.dict("os.environ", self.gym_env)
        gym_settings.start()
        self.addCleanup(gym_settings.stop)
        self.expected["gym_agent"] = {"harness": self.gym_env["GYM_AGENT"], "model": self.gym_env["GYM_MODEL"]}
        self.window = {"container_started_at": 1000, "verifier_started_at": 2000}
        self.row = {
            "responses_create_params": {"input": [{"role": "user", "content": self.expected["task_request"]}]},
            "reward": 0,
            "response": {
                "id": "response-test",
                "model": "gym-policy-model",
                "created_at": 1500,
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "A scored response."}],
                    }
                ],
            },
        }
        self.rollout = self.workspace / "saved-result"
        self.rollout.write_text(json.dumps(self.row) + "\n")
        self.trajectory = {
            "steps": [
                {
                    "step_id": 1,
                    "timestamp": "2026-10-23T00:00:00Z",
                    "tool_calls": [
                        {
                            "tool_call_id": "launch",
                            "function_name": "exec_command",
                            "arguments": {"cmd": "python run.py"},
                        }
                    ],
                    "observation": {
                        "results": [
                            {
                                "source_call_id": "launch",
                                "content": "Inner Codex ready; Gym collection finished; exit 0.",
                            }
                        ]
                    },
                    "reasoning_content": "private-test-placeholder",
                }
            ]
        }
        (self.agent / "trajectory.json").write_text(json.dumps(self.trajectory))

    def evidence(self):
        # Gather only the temporary files, with a fixed time window and no oracle script.
        return read_execution.EvidenceFiles(
            self.expected,
            workspace=self.workspace,
            agent_root=self.agent,
            solution=self.root / "missing-solution",
            window=self.window,
        )

    def judgment(self, evidence, status="supported"):
        # Build a fake answer citing the complete fixture result and trace that
        # fit in the initial bundle. Creating these IDs does not register delivery.
        output_id = evidence.candidates[0]["file_id"]
        trace_id = next(key for key, item in evidence.files.items() if item["kind"] == "public_trace")
        citations = [
            {
                "excerpt_id": read_execution.label_excerpt(
                    {"file_id": file_id, "offset": 0, "text": evidence.text(file_id)}
                )["excerpt_id"]
            }
            for file_id in (output_id, trace_id)
        ]
        return {
            "selected_result": {"file_id": output_id, "row_index": 0},
            "findings": {
                name: {
                    "status": status,
                    "explanation": "Supported by recorded evidence.",
                    "evidence": deepcopy(citations),
                }
                for name in self.expected["verification_contract"]["requirements"]
            },
        }

    def response(self, judgment):
        # Construct a fake Responses envelope without contacting a provider.
        return {
            "id": "response-test",
            "model": "azure/openai/gpt-6.1-terra",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": json.dumps(judgment)}],
                }
            ],
        }

    def run_review(self, evidence, requester, *, env=None, clock=None):
        # Run the review with fake model replies and temporary report files.
        kwargs = {} if clock is None else {"clock": clock}
        with patch.dict("os.environ", {**self.verifier_env, **(env or {})}, clear=True):
            return hybrid.review(
                evidence,
                report_dir=self.root / "reports",
                artifacts=self.root / "retained",
                requester=requester,
                **kwargs,
            )

    def request(self, opener, *, sleeper=None, clock=None, attempts=None):
        # Use a fake connection, key and clock to test transport without network or waiting.
        with (
            patch.dict("os.environ", {"VERIFIER_JUDGE_API_KEY": "fake-test-key"}, clear=True),
            patch.object(hybrid.random, "uniform", return_value=0),
        ):
            return hybrid.request_response(
                {"model": "azure/openai/gpt-6.1-terra", "input": []},
                deadline=80,
                attempts=[] if attempts is None else attempts,
                directory=self.root / "requests",
                opener=opener,
                sleeper=MagicMock() if sleeper is None else sleeper,
                clock=MagicMock(return_value=0) if clock is None else clock,
            )

    def stream(self, response):
        # Return an in-memory HTTP stream containing only synthetic data.
        stream = MagicMock()
        stream.__enter__.return_value.read.return_value = json.dumps(response).encode()
        return stream

    def test_grading_criteria_and_answer_schema_use_only_the_four_requirement_ids(self):
        # Check that the criteria, task specification and Python answer format
        # use the same four requirement IDs so their findings stay aligned.
        contract = self.expected["verification_contract"]
        self.assertEqual(
            set(contract["requirements"]), {"carlos_task", "gym_evaluation", "gym_agent", "saved_scored_result"}
        )
        task_spec = (TASK_TESTS.parent / "Task.md").read_text()
        for name in contract["requirements"]:
            self.assertIn(f"`{name}`", task_spec)
            self.assertIn(f"`{name}`", hybrid.GRADING_CRITERIA)
        schema = hybrid.response_schema(contract)["properties"]["findings"]
        self.assertEqual(set(schema["required"]), set(contract["requirements"]))
        self.assertFalse(schema["additionalProperties"])
        citation = schema["properties"]["gym_evaluation"]["properties"]["evidence"]["items"]
        self.assertEqual(citation["required"], ["excerpt_id"])
        self.assertEqual(set(citation["properties"]), {"excerpt_id"})

    def test_all_eight_saved_result_facts_are_preserved_as_evidence(self):
        # Check that collected evidence keeps all eight result facts and the original saved row.
        evidence = self.evidence()
        facts = evidence.candidates[0]["facts"]
        self.assertEqual(set(facts), set(self.expected["verification_contract"]["saved_result_evidence"]))
        for name in facts:
            self.assertIn(f"`{name}`", hybrid.GRADING_CRITERIA)
        self.assertTrue(all(facts.values()))
        self.assertEqual(evidence.text(evidence.candidates[0]["file_id"]), self.rollout.read_text())

    def test_invalid_saved_results_are_visible_instead_of_prefiltered(self):
        # Keep an out-of-range reward visible to the judge and flag its score as invalid.
        self.row["reward"] = 2
        self.rollout.write_text(json.dumps(self.row))
        evidence = self.evidence()
        self.assertEqual(len(evidence.candidates), 1)
        self.assertFalse(evidence.candidates[0]["facts"]["gym_reward_is_valid"])
        self.assertIn('"reward": 2', evidence.text(evidence.candidates[0]["file_id"]))

    def test_unfamiliar_commands_and_full_middle_output_reach_the_judge(self):
        # Do not require a Gym command pattern or discard a long output's middle.
        output = "start " + "x" * 20000 + " Actual Codex runtime proof " + "y" * 20000 + " end"
        self.trajectory["steps"][0]["observation"]["results"][0]["content"] = output
        (self.agent / "trajectory.json").write_text(json.dumps(self.trajectory))
        evidence = self.evidence()
        trace_id = next(key for key, item in evidence.files.items() if item["kind"] == "public_trace")
        initial = evidence.initial(12000)
        self.assertNotIn("Actual Codex runtime proof", json.dumps(initial))
        found = evidence.search(trace_id, "Actual Codex runtime proof", 0)
        self.assertTrue(found["found"])
        self.assertIn("Actual Codex runtime proof", found["text"])
        self.assertIn("python run.py", evidence.text(trace_id))

    def test_public_traces_keep_ids_and_exits_without_private_reasoning(self):
        # Preserve original command events rather than deriving statuses from unrelated records.
        events = [
            {"type": "thread.started", "thread_id": "thread-one"},
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "id": "item_1",
                    "command": "python run.py",
                    "exit_code": 0,
                    "aggregated_output": "Done.",
                },
            },
            {"type": "item.completed", "item": {"type": "reasoning", "text": "private-test-placeholder"}},
        ]
        (self.agent / "codex.txt").write_text("".join(json.dumps(event) + "\n" for event in events))
        evidence = self.evidence()
        traces = "\n".join(
            evidence.text(key) for key, value in evidence.files.items() if value["kind"] == "public_trace"
        )
        self.assertIn("item_1", traces)
        self.assertIn('"exit_code": 0', traces)
        self.assertNotIn("private-test-placeholder", traces)

    def test_small_searches_keep_complete_matches_and_separate_search_continuation(self):
        # Find both nearby matches without skipping the second one when paging search results.
        log = self.workspace / "server.log"
        log.write_text("x" * 5000 + "runtime-one " + "y" * 20 + "runtime-two " + "z" * 20000)
        evidence = self.evidence()
        file_id = next(key for key, item in evidence.files.items() if item["path"] == str(log))
        first = evidence.search(file_id, "runtime-", 0)
        self.assertEqual(len(first["text"]), 2000)
        self.assertIn("runtime-one", first["text"])
        second = evidence.search(file_id, "runtime-", first["next_search_offset"], limit=100)
        self.assertEqual(second["match_offset"], log.read_text().index("runtime-two"))
        self.assertIn("runtime-two", second["text"])
        self.assertEqual(len(second["text"]), 100)
        expanded = evidence.read(file_id, first["offset"], 16000)
        self.assertEqual(expanded["text"], log.read_text()[first["offset"] : first["offset"] + 16000])
        self.assertFalse(evidence.search(file_id, "absent", 0)["found"])
        with self.assertRaises(ValueError):
            evidence.search(file_id, "long-query", 0, limit=2)

    def test_overlap_feedback_tracks_only_sections_actually_returned(self):
        # Unsent text must not appear as evidence already seen by the judge.
        evidence = self.evidence()
        file_id = evidence.candidates[0]["file_id"]
        first = evidence.read(file_id, 0, 100)
        self.assertEqual(first["previously_returned"], [])
        evidence.record_returned(first)
        overlap = evidence.read(file_id, 50, 100)
        self.assertEqual(overlap["previously_returned"], [{"offset": 0, "end_offset": 100}])
        evidence.record_returned(overlap)
        self.assertEqual(evidence.returned_ranges[file_id], [(0, 150)])
        self.assertEqual(evidence.read(file_id, 200, 20)["previously_returned"], [])
        evidence.initial(12000)
        repeated = evidence.read(file_id, 0, 100)
        self.assertEqual(repeated["previously_returned"][0]["end_offset"], len(self.rollout.read_text()))

    def test_serialized_limits_preserve_search_matches_and_exact_source_offsets(self):
        # Escaping nested JSON must not let a tool result exceed its real message allowance.
        source = ('"\\\n' * 1000) + "actual-runtime" + ('"\\\n' * 1000)
        output = {
            "file_id": "test-log",
            "offset": 0,
            "text": source,
            "total_chars": len(source),
            "next_offset": len(source),
            "eof": True,
            "found": True,
            "match_offset": 3000,
            "match_end_offset": 3014,
            "next_search_offset": 3014,
        }
        limited = hybrid.bounded_tool_output("call-one", output, 2000)
        self.assertLessEqual(hybrid.serialized_chars(hybrid.tool_message("call-one", limited)), 2000)
        self.assertTrue(limited["budget_limited"])
        self.assertTrue(limited["match_in_text"])
        self.assertIn("actual-runtime", limited["text"])
        self.assertEqual(limited["text"], source[limited["offset"] : limited["next_offset"]])
        self.assertFalse(limited["eof"])
        self.assertEqual(limited["next_search_offset"], 3014)
        read = {
            key: value
            for key, value in output.items()
            if key not in {"found", "match_offset", "match_end_offset", "next_search_offset"}
        }
        read = hybrid.bounded_tool_output("call-two", read, 2000)
        self.assertEqual(read["offset"], 0)
        self.assertEqual(read["text"], source[: read["next_offset"]])

    def test_catalog_shortening_returns_the_next_unsent_file(self):
        # A smaller catalog page must not skip file IDs that did not fit.
        output = {
            "files": [{"id": f"file:{index}", "path": "x" * 100} for index in range(100, 200)],
            "offset": 100,
            "next_offset": 200,
            "eof": True,
        }
        limited = hybrid.bounded_tool_output("catalog-call", output, 2000)
        self.assertTrue(limited["budget_limited"])
        self.assertGreater(len(limited["files"]), 0)
        self.assertLess(len(limited["files"]), 100)
        self.assertEqual(limited["next_offset"], 100 + len(limited["files"]))
        self.assertFalse(limited["eof"])
        self.assertLessEqual(hybrid.serialized_chars(hybrid.tool_message("catalog-call", limited)), 2000)

    def test_missing_codex_and_malformed_optional_trajectory_allow_native_evidence(self):
        # An optional capture cannot invalidate the other recorded public actions.
        (self.agent / "trajectory.json").write_text("{broken")
        native = self.agent / "sessions/session.jsonl"
        native.parent.mkdir()
        native.write_text(
            json.dumps(
                {
                    "type": "response_item",
                    "timestamp": "2026-10-23T00:00:00Z",
                    "payload": {"type": "function_call_output", "call_id": "launch", "output": "Completed."},
                }
            )
        )
        evidence = self.evidence()
        self.assertTrue(
            any(
                "Completed." in evidence.text(key)
                for key, item in evidence.files.items()
                if item["kind"] == "public_trace"
            )
        )
        self.assertEqual(evidence.diagnostics[0]["reason"], "optional_trace_unreadable")

    def test_supporting_files_are_read_on_demand_and_cannot_execute_commands(self):
        # A judge can inspect a script, but its tools have no terminal or arbitrary path capability.
        script = self.workspace / "run.py"
        script.write_text('print("a public script")')
        evidence = self.evidence()
        file_id = next(key for key, value in evidence.files.items() if value["path"] == str(script))
        output = evidence.tool("read_evidence", {"file_id": file_id, "offset": 0, "limit": 100})
        self.assertEqual(output["text"], script.read_text())
        self.assertTrue(evidence.tool("exec_command", {"cmd": "anything"})["unavailable"])
        self.assertTrue(
            evidence.tool("read_evidence", {"file_id": "/etc/shadow", "offset": 0, "limit": 100})["unavailable"]
        )

    def test_no_credential_redaction_is_performed(self):
        # Preserve the user's requested recorded text without opening any environment file.
        script = self.workspace / "run.log"
        script.write_text("a harmless fake-test-key appears in public output")
        evidence = self.evidence()
        file_id = next(key for key, item in evidence.files.items() if item["path"] == str(script))
        with patch.dict("os.environ", {"VERIFIER_JUDGE_API_KEY": "fake-test-key"}):
            self.assertIn("fake-test-key", evidence.text(file_id))

    def test_supported_zero_receives_hybrid_reward_one(self):
        # A zero email score should still allow a supported workflow to pass.
        evidence = self.evidence()
        requester = MagicMock(return_value=self.response(self.judgment(evidence)))
        result = self.run_review(evidence, requester)
        self.assertEqual(result["hybrid_reward"], 1)
        self.assertEqual(result["gym_reward"], 0)
        self.assertEqual((self.root / "retained/rollout.jsonl").read_bytes(), self.rollout.read_bytes())
        body = requester.call_args.args[0]
        self.assertEqual(body["model"], "azure/openai/gpt-6.1-terra")
        self.assertEqual(body["reasoning"]["effort"], "high")
        self.assertIn(hybrid.JUDGE_PROMPT, body["instructions"])
        self.assertIn(hybrid.GRADING_CRITERIA, body["instructions"])
        self.assertEqual((self.root / "reports/rubric.txt").read_text(), hybrid.RUBRIC)
        self.assertEqual(
            {tool["name"] for tool in body["tools"]}, {"list_evidence", "read_evidence", "search_evidence"}
        )
        self.assertFalse(body["store"])

    def test_optional_metrics_copy_failure_does_not_change_a_supported_verdict(self):
        # Optional metrics must not add a passing requirement during artifact retention.
        self.rollout.with_name("saved-result_aggregate_metrics.json").write_text('{"reward": 0}')
        evidence = self.evidence()
        requester = MagicMock(return_value=self.response(self.judgment(evidence)))
        original_copy = hybrid.shutil.copyfile

        def copy(source, destination):
            # Fail only the final optional copy, after preserving the candidate's metrics.
            if Path(destination).parent == self.root / "retained":
                raise PermissionError("synthetic copy failure")
            return original_copy(source, destination)

        with patch.object(hybrid.shutil, "copyfile", side_effect=copy):
            result = self.run_review(evidence, requester)
        self.assertEqual(result["hybrid_reward"], 1)
        self.assertEqual(result["artifact_diagnostics"][0]["reason"], "optional_metrics_unreadable")

    def test_requirement_coverage_missing_citations_and_invented_results_are_rejected(self):
        # Reject extra requirements, missing or invented references, and nonexistent selected results.
        evidence = self.evidence()
        evidence.initial(12000)
        judgment = self.judgment(evidence)
        judgment["findings"]["new_requirement"] = deepcopy(judgment["findings"]["gym_evaluation"])
        with self.assertRaisesRegex(ValueError, "judge_requirements_mismatch"):
            hybrid.validate_judgment(judgment, evidence)
        judgment = self.judgment(evidence)
        judgment["findings"]["gym_agent"]["evidence"] = []
        with self.assertRaisesRegex(ValueError, "judge_citation_missing"):
            hybrid.validate_judgment(judgment, evidence)
        judgment = self.judgment(evidence)
        judgment["findings"]["gym_evaluation"]["evidence"][0]["excerpt_id"] = "invented-reference"
        with self.assertRaisesRegex(ValueError, "judge_citation_not_found"):
            hybrid.validate_judgment(judgment, evidence)
        judgment = self.judgment(evidence)
        judgment["selected_result"]["file_id"] = "missing-file"
        with self.assertRaisesRegex(ValueError, "judge_selected_result_not_found"):
            hybrid.validate_judgment(judgment, evidence)

    def test_only_delivered_excerpts_can_be_cited(self):
        # Merely finding source text must not make it citable. Only text actually
        # supplied in the initial bundle or a tool response can support a reference.
        log = self.workspace / "runtime.log"
        log.write_text("Actual harness runtime recorded here.")
        evidence = self.evidence()
        initial = evidence.initial(12000)
        for section in initial["contents"]:
            resolved = evidence.excerpts[section["excerpt_id"]]
            self.assertEqual(resolved["quote"], section["text"])
            self.assertEqual(resolved["file_id"], section["file_id"])
            self.assertEqual(resolved["offset"], 0)
            self.assertEqual(resolved["end_offset"], len(section["text"]))
        file_id = next(key for key, item in evidence.files.items() if item["path"] == str(log))
        section = evidence.read(file_id, 0, 1000)
        judgment = self.judgment(evidence)
        judgment["findings"]["gym_agent"]["evidence"] = [{"excerpt_id": section["excerpt_id"]}]
        with self.assertRaisesRegex(ValueError, "judge_citation_not_found"):
            hybrid.validate_judgment(judgment, evidence)
        evidence.record_returned(section)
        self.assertTrue(hybrid.validate_judgment(judgment, evidence)[0])
        # Citing a real runtime section alone does not cite the selected saved result.
        for finding in judgment["findings"].values():
            finding["evidence"] = [{"excerpt_id": section["excerpt_id"]}]
        with self.assertRaisesRegex(ValueError, "judge_selected_result_not_cited"):
            hybrid.validate_judgment(judgment, evidence)

    def test_shortened_and_repeated_reads_keep_their_own_citation_ranges(self):
        # A reference must identify the final shortened text, not the larger
        # requested page. Later overlapping reads must not replace an earlier one.
        log = self.workspace / "long.log"
        log.write_text('"\\\n' * 10000)
        evidence = self.evidence()
        file_id = next(key for key, item in evidence.files.items() if item["path"] == str(log))
        requested = evidence.read(file_id, 0, 16000)
        shortened = hybrid.bounded_tool_output("read-one", requested, 1500)
        evidence.record_returned(shortened)
        self.assertTrue(shortened["text"])
        self.assertNotEqual(requested["excerpt_id"], shortened["excerpt_id"])
        self.assertNotIn(requested["excerpt_id"], evidence.excerpts)
        resolved = deepcopy(evidence.excerpts[shortened["excerpt_id"]])
        self.assertEqual(resolved["quote"], log.read_text()[resolved["offset"] : resolved["end_offset"]])
        self.assertEqual(resolved["quote"], shortened["text"])
        self.assertLessEqual(hybrid.serialized_chars(hybrid.tool_message("read-one", shortened)), 1500)
        later = evidence.read(file_id, 10, 30)
        evidence.record_returned(later)
        self.assertEqual(evidence.excerpts[shortened["excerpt_id"]], resolved)
        self.assertNotEqual(later["excerpt_id"], shortened["excerpt_id"])
        self.assertEqual(evidence.read(file_id, 10, 30)["excerpt_id"], later["excerpt_id"])
        empty = evidence.read(file_id, len(log.read_text()), 100)
        self.assertNotIn("excerpt_id", empty)
        self.assertNotIn("excerpt_id", hybrid.bounded_tool_output("tiny", requested, 1))
        self.assertNotIn("excerpt_id", read_execution.label_excerpt({"file_id": file_id, "offset": 0, "text": " \n"}))

    def test_references_preserve_escaped_source_text_and_file_identity_in_reports(self):
        # Use escaped JSON and identical text in different files to check that
        # delivered IDs preserve the exact source text and its file in the report.
        sources = [r"\"mean/reward\": 1.0", r"CALL reply {\"id\": \"example\"}\nRESULT {\"output\":\"ok\"}"]
        for source in sources:
            with self.subTest(source=source):
                paths = [self.workspace / "runtime.log", self.workspace / "runtime-copy.log"]
                for path in paths:
                    path.write_text("prefix " + source + " suffix")
                evidence = self.evidence()
                file_ids = [
                    next(key for key, item in evidence.files.items() if item["path"] == str(path)) for path in paths
                ]
                bodies = []

                def requester(body, **kwargs):
                    # Cite only the exact sections returned by the real tool path.
                    bodies.append(deepcopy(body))
                    if len(bodies) == 1:
                        return {
                            "status": "completed",
                            "output": [
                                {
                                    "type": "function_call",
                                    "call_id": f"read-{index}",
                                    "name": "read_evidence",
                                    "arguments": json.dumps({"file_id": file_id, "offset": 7, "limit": len(source)}),
                                }
                                for index, file_id in enumerate(file_ids)
                            ],
                        }
                    sections = [
                        json.loads(item["output"])
                        for item in body["input"]
                        if item.get("type") == "function_call_output"
                    ]
                    judgment = self.judgment(evidence)
                    judgment["findings"]["gym_evaluation"]["evidence"].extend(
                        {"excerpt_id": section["excerpt_id"]} for section in sections
                    )
                    return self.response(judgment)

                result = self.run_review(evidence, requester)
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["hybrid_reward"], 1)
                sections = [
                    item["output"] for item in json.loads((self.root / "reports/file-access.json").read_text())
                ]
                self.assertEqual(len({section["excerpt_id"] for section in sections}), 2)
                for section, path in zip(sections, paths):
                    resolved = result["citations"][section["excerpt_id"]]
                    self.assertEqual(
                        resolved,
                        {
                            "file_id": section["file_id"],
                            "path": str(path),
                            "offset": 7,
                            "end_offset": 7 + len(source),
                            "quote": source,
                        },
                    )
                saved = json.loads((self.root / "reports/report.json").read_text())
                self.assertEqual(saved["citations"], result["citations"])
                answer = json.loads((self.root / "reports/answer.txt").read_text())
                self.assertEqual(set(answer["findings"]["gym_evaluation"]["evidence"][-1]), {"excerpt_id"})

    def test_negative_or_insufficient_judgments_finish_without_resampling(self):
        # For either Gym agent, keep Harbor reward 0 when the judge reports the
        # requested agent did not run or its execution is not supported by evidence.
        for harness in ("codex_agent", "claude_code_agent"):
            self.expected["gym_agent"]["harness"] = harness
            for status in ("contradicted", "insufficient_evidence"):
                with self.subTest(harness=harness, status=status):
                    evidence = self.evidence()
                    judgment = self.judgment(evidence)
                    judgment["findings"]["gym_agent"]["status"] = status
                    requester = MagicMock(return_value=self.response(judgment))
                    result = self.run_review(evidence, requester)
                    self.assertEqual(result["status"], "completed")
                    self.assertEqual(result["hybrid_reward"], 0)
                    requester.assert_called_once()

    def test_judge_can_request_additional_text_before_its_decision(self):
        # Replay tool calls and encrypted continuation data in memory, saving only public accesses.
        evidence = self.evidence()
        file_id = evidence.candidates[0]["file_id"]
        tool_reply = {
            "status": "completed",
            "output": [
                {"type": "reasoning", "id": "reasoning-test", "encrypted_content": "private-test-placeholder"},
                {
                    "type": "function_call",
                    "id": "tool-test",
                    "call_id": "read-one",
                    "name": "read_evidence",
                    "arguments": json.dumps({"file_id": file_id, "offset": 0, "limit": 1000}),
                },
            ],
        }
        bodies = []

        def requester(body, **kwargs):
            # Snapshot each synthetic request without opening a network connection.
            bodies.append(deepcopy(body))
            return tool_reply if len(bodies) == 1 else self.response(self.judgment(evidence))

        result = self.run_review(evidence, requester)
        self.assertEqual(result["hybrid_reward"], 1)
        self.assertEqual(len(bodies), 2)
        self.assertTrue(any(item.get("type") == "function_call_output" for item in bodies[1]["input"]))
        self.assertIn("private-test-placeholder", json.dumps(bodies[1]))
        for path in (self.root / "reports").glob("*"):
            self.assertNotIn("private-test-placeholder", path.read_text())
        self.assertEqual(json.loads((self.root / "reports/file-access.json").read_text())[0]["name"], "read_evidence")

    def test_parallel_large_reads_leave_room_for_a_final_assessment(self):
        # Answer every parallel call within the cap, including heavily escaped source text.
        log = self.workspace / "large.log"
        log.write_text('"\\\n' * 30000)
        evidence = self.evidence()
        file_id = next(key for key, item in evidence.files.items() if item["path"] == str(log))
        tool_reply = {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": f"read-{index}",
                    "name": "read_evidence",
                    "arguments": json.dumps({"file_id": file_id, "offset": index * 16000, "limit": 16000}),
                }
                for index in range(6)
            ],
        }
        bodies = []

        def requester(body, **kwargs):
            # Return an unresolved finding when the final decision still lacks execution proof.
            bodies.append(deepcopy(body))
            return tool_reply if len(bodies) == 1 else self.response(self.judgment(evidence, "insufficient_evidence"))

        result = self.run_review(evidence, requester, env={"VERIFIER_JUDGE_MAX_EVIDENCE_CHARS": "40000"})
        self.assertEqual(result["status"], "judge_error")
        self.assertIsNone(result["hybrid_reward"])
        self.assertEqual(result["reason"], "judge_review_budget_exhausted")
        self.assertIn("judgment", result)
        self.assertEqual(len(bodies), 2)
        self.assertEqual(bodies[1]["tool_choice"], "none")
        self.assertTrue(all(hybrid.serialized_chars(body) <= 40000 for body in bodies))
        returned = [item for item in bodies[1]["input"] if item.get("type") == "function_call_output"]
        self.assertEqual([item["call_id"] for item in returned], [f"read-{index}" for index in range(6)])
        outputs = [json.loads(item["output"]) for item in returned]
        self.assertTrue(any(item.get("budget_limited") for item in outputs))
        for output in outputs:
            if "text" in output:
                self.assertEqual(output["text"], log.read_text()[output["offset"] : output["next_offset"]])
                if output["text"]:
                    resolved = evidence.excerpts[output["excerpt_id"]]
                    self.assertEqual(resolved["quote"], output["text"])
                    self.assertEqual(resolved["end_offset"], output["next_offset"])
        self.assertEqual(result["review_budget"][-1]["phase"], "final_assessment")
        self.assertTrue(list((self.root / "retained").glob("*/rollout.jsonl")))

    def test_reading_deadline_reserves_time_for_the_same_judges_decision(self):
        # Check configured reserves, a budget above 90 seconds and the smaller-budget cap.
        # Late file reads must leave time for the same judge to return its final answer.
        for budget, reserve, reading_deadline in ((80, 15, 65), (100, 20, 80), (20, 15, 15)):
            with self.subTest(budget=budget, reserve=reserve):
                evidence = self.evidence()
                file_id = evidence.candidates[0]["file_id"]
                clock = MagicMock(return_value=0)
                bodies, deadlines = [], []

                def requester(body, **kwargs):
                    # Advance a fake clock rather than waiting or contacting a provider.
                    bodies.append(deepcopy(body))
                    deadlines.append(kwargs["deadline"])
                    if len(bodies) == 1:
                        clock.return_value = reading_deadline + 1
                        return {
                            "status": "completed",
                            "output": [
                                {
                                    "type": "function_call",
                                    "call_id": "late-read",
                                    "name": "read_evidence",
                                    "arguments": json.dumps({"file_id": file_id, "offset": 0, "limit": 1000}),
                                }
                            ],
                        }
                    return self.response(self.judgment(evidence))

                env = {"VERIFIER_JUDGE_TIME_BUDGET_SEC": str(budget), "VERIFIER_JUDGE_FINAL_RESERVE_SEC": str(reserve)}
                with patch.object(evidence, "tool", wraps=evidence.tool) as file_tool:
                    result = self.run_review(evidence, requester, env=env, clock=clock)
                self.assertEqual(result["hybrid_reward"], 1)
                self.assertEqual(deadlines, [reading_deadline, budget])
                self.assertEqual(bodies[1]["tool_choice"], "none")
                self.assertEqual(result["review_budget"][-1]["final_reason"], "time_reserved")
                file_tool.assert_not_called()
                withheld = next(item for item in bodies[1]["input"] if item.get("type") == "function_call_output")
                self.assertTrue(json.loads(withheld["output"])["budget_limited"])

    def test_invalid_configured_timing_stops_before_calling_the_judge(self):
        # Reject unusable timing values instead of making a request without a valid budget.
        for name in ("VERIFIER_JUDGE_TIME_BUDGET_SEC", "VERIFIER_JUDGE_FINAL_RESERVE_SEC"):
            for value in ("0", "-1", "nan", "inf"):
                with self.subTest(setting=name, value=value):
                    requester = MagicMock()
                    result = self.run_review(self.evidence(), requester, env={name: value})
                    self.assertEqual(result["status"], "judge_error")
                    self.assertEqual(result["reason"], "judge_budget_invalid")
                    requester.assert_not_called()

    def test_a_smaller_parallel_section_can_be_expanded_when_the_batch_leaves_space(self):
        # One shortened section must not end reading if other calls used little of their allowance.
        log = self.workspace / "large.log"
        log.write_text('"\\\n' * 10000)
        evidence = self.evidence()
        file_id = next(key for key, item in evidence.files.items() if item["path"] == str(log))
        bodies = []

        def requester(body, **kwargs):
            # Return one large read and one empty catalog page, leaving room for another reading turn.
            bodies.append(deepcopy(body))
            if len(bodies) == 1:
                return {
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "large-read",
                            "name": "read_evidence",
                            "arguments": json.dumps({"file_id": file_id, "offset": 0, "limit": 16000}),
                        },
                        {
                            "type": "function_call",
                            "call_id": "empty-page",
                            "name": "list_evidence",
                            "arguments": '{"offset": 100000}',
                        },
                    ],
                }
            return self.response(self.judgment(evidence))

        result = self.run_review(evidence, requester, env={"VERIFIER_JUDGE_MAX_EVIDENCE_CHARS": "60000"})
        self.assertEqual(result["hybrid_reward"], 1)
        self.assertEqual(len(bodies), 2)
        self.assertEqual(bodies[1]["tool_choice"], "auto")
        returned = next(
            item
            for item in bodies[1]["input"]
            if item.get("call_id") == "large-read" and item.get("type") == "function_call_output"
        )
        self.assertTrue(json.loads(returned["output"])["budget_limited"])
        self.assertIsNone(result["review_budget"][-1]["final_reason"])

    def test_the_last_turn_requests_a_decision_instead_of_another_file(self):
        # Sixteen requests include the final assessment, rather than ending after unanswered tool calls.
        evidence = self.evidence()
        bodies = []

        def requester(body, **kwargs):
            # Use small fake catalog calls to reach the turn boundary without exhausting other budgets.
            bodies.append(deepcopy(body))
            if body["tool_choice"] == "none":
                return self.response(self.judgment(evidence))
            return {
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": f"catalog-{len(bodies)}",
                        "name": "list_evidence",
                        "arguments": '{"offset": 0}',
                    }
                ],
            }

        result = self.run_review(evidence, requester, clock=MagicMock(return_value=0))
        self.assertEqual(result["hybrid_reward"], 1)
        self.assertEqual(len(bodies), 16)
        self.assertEqual(bodies[-1]["tool_choice"], "none")
        self.assertEqual(result["review_budget"][-1]["final_reason"], "last_request")

    def test_a_cited_contradiction_is_still_a_completed_nonpass_at_the_budget_boundary(self):
        # Preserve a judge-reported contradiction even when other findings are unresolved.
        evidence = self.evidence()
        clock = MagicMock(return_value=0)
        judgment = self.judgment(evidence, "insufficient_evidence")
        judgment["findings"]["gym_agent"]["status"] = "contradicted"
        bodies = []

        def requester(body, **kwargs):
            # Ask for a file at the reading deadline, then return one cited contradiction.
            bodies.append(deepcopy(body))
            if len(bodies) == 1:
                clock.return_value = 66
                return {
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "catalog",
                            "name": "list_evidence",
                            "arguments": '{"offset": 0}',
                        }
                    ],
                }
            return self.response(judgment)

        result = self.run_review(evidence, requester, clock=clock)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["hybrid_reward"], 0)
        self.assertEqual(result["review_budget"][-1]["final_reason"], "time_reserved")

    def test_provider_output_that_alone_exceeds_the_cap_remains_an_ungraded_error(self):
        # A continuation too large to replay cannot become a judgment about the agent's workflow.
        evidence = self.evidence()
        requester = MagicMock(
            return_value={
                "status": "completed",
                "output": [
                    {"type": "reasoning", "encrypted_content": "x" * 50000},
                    {
                        "type": "function_call",
                        "call_id": "catalog",
                        "name": "list_evidence",
                        "arguments": '{"offset": 0}',
                    },
                ],
            }
        )
        result = self.run_review(evidence, requester, env={"VERIFIER_JUDGE_MAX_EVIDENCE_CHARS": "40000"})
        self.assertEqual(result["status"], "judge_error")
        self.assertIsNone(result["hybrid_reward"])
        self.assertEqual(result["reason"], "judge_input_budget_exceeded")
        requester.assert_called_once()
        self.assertTrue(list((self.root / "retained").glob("*/rollout.jsonl")))

    def test_judge_service_failure_retains_outputs_without_a_pass(self):
        # Evidence already discovered is retained even when no judgment can be obtained.
        evidence = self.evidence()
        requester = MagicMock(side_effect=HTTPError("https://example.test", 503, "private-error", None, None))
        result = self.run_review(evidence, requester)
        self.assertEqual(result["status"], "judge_error")
        self.assertIsNone(result["hybrid_reward"])
        self.assertEqual(result["saved_results"][0]["gym_reward"], 0)
        self.assertTrue(list((self.root / "retained").glob("*/rollout.jsonl")))
        self.assertNotIn("private-error", json.dumps(result))

    def test_temporary_http_errors_retry_only_the_identical_request(self):
        # Recover from two temporary errors without changing the evidence or starting real waits.
        response = self.response(self.judgment(self.evidence()))
        opener = MagicMock(
            side_effect=[
                HTTPError("https://example.test", 503, "error", None, None),
                HTTPError("https://example.test", 503, "error", None, None),
                self.stream(response),
            ]
        )
        clock = MagicMock(return_value=0)
        sleeper = MagicMock(side_effect=lambda delay: setattr(clock, "return_value", clock.return_value + delay))
        attempts = []
        self.assertEqual(self.request(opener, sleeper=sleeper, clock=clock, attempts=attempts), response)
        self.assertEqual([call.args[0] for call in sleeper.call_args_list], [2, 4])
        self.assertEqual([call.kwargs["timeout"] for call in opener.call_args_list], [80, 78, 74])
        self.assertTrue(all(call.args[0].data == opener.call_args.args[0].data for call in opener.call_args_list))
        self.assertEqual([item["status"] for item in attempts], ["request_error", "request_error", "completed"])

    def test_transport_errors_can_recover_but_authentication_and_bad_answers_do_not_retry(self):
        # Retry a connection error; stop on HTTP 400/401/403/429 and incomplete replies.
        response = self.response(self.judgment(self.evidence()))
        opener = MagicMock(side_effect=[URLError("unavailable"), self.stream(response)])
        self.assertEqual(self.request(opener), response)
        for status in (400, 401, 403, 429):
            opener = MagicMock(side_effect=HTTPError("https://example.test", status, "error", None, None))
            with self.assertRaises(HTTPError):
                self.request(opener)
            opener.assert_called_once()
        opener = MagicMock(return_value=self.stream({"status": "incomplete"}))
        with self.assertRaisesRegex(ValueError, "judge_response_incomplete"):
            self.request(opener)
        opener.assert_called_once()

    def test_http_error_body_is_saved_without_retrying_a_bad_request(self):
        # Keep the server's rejected-request explanation in the saved diagnostics,
        # while preserving HTTP 400 as an error with no retry.
        for payload in (b'{"error":{"message":"Unsupported request field"}}', b"x" * 16001, b"\xff"):
            with self.subTest(size=len(payload)):
                error = HTTPError("https://example.test", 400, "error", None, BytesIO(payload))
                opener = MagicMock(side_effect=error)
                with self.assertRaises(HTTPError) as raised:
                    self.request(opener)
                self.assertIs(raised.exception, error)
                opener.assert_called_once()
                saved = json.loads((self.root / "requests/request-attempts.json").read_text())
                self.assertEqual(saved[0]["http_status"], 400)
                self.assertEqual(saved[0]["response_body"], payload[:16000].decode("utf-8", errors="replace"))
                self.assertEqual(saved[0]["response_body_truncated"], len(payload) > 16000)

    def test_failed_error_body_read_preserves_the_original_http_error(self):
        # Report an unreadable diagnostic body without replacing the HTTP failure
        # that explains why the judge request stopped.
        error = HTTPError("https://example.test", 400, "error", None, BytesIO())
        with patch.object(error, "read", side_effect=OSError("synthetic read failure")):
            with self.assertRaises(HTTPError) as raised:
                self.request(MagicMock(side_effect=error))
        self.assertIs(raised.exception, error)
        saved = json.loads((self.root / "requests/request-attempts.json").read_text())
        self.assertEqual(saved[0]["response_body_error"], "OSError")
        self.assertEqual(saved[0]["http_status"], 400)

    def test_retry_after_and_shared_deadline_are_preserved(self):
        # Honor the server's delay unless the remaining review budget cannot fit it.
        response = self.response(self.judgment(self.evidence()))
        opener = MagicMock(
            side_effect=[
                HTTPError("https://example.test", 503, "error", {"Retry-After": "7"}, None),
                self.stream(response),
            ]
        )
        sleeper = MagicMock()
        self.request(opener, sleeper=sleeper)
        sleeper.assert_called_once_with(7)
        attempts = []
        opener = MagicMock(side_effect=HTTPError("https://example.test", 503, "error", {"Retry-After": "90"}, None))
        with self.assertRaises(HTTPError):
            self.request(opener, attempts=attempts)
        self.assertEqual(attempts[0]["retry_skipped"], "time_budget_exhausted")

    def test_reference_loading_combines_generated_hashes_with_task_data(self):
        # The entry point must use the generated baseline while preserving the
        # task request and requirement IDs from reference.json.
        baseline = self.root / "bundled-output-hashes.json"
        hashes = ["a" * 64, "b" * 64]
        baseline.write_text(json.dumps({"bundled_output_hashes": hashes}))
        self.assertEqual(entry.load_reference(baseline), {**self.expected, "bundled_output_hashes": hashes})

    def test_reference_loading_rejects_missing_or_unusable_sample_hashes(self):
        # A missing or malformed baseline is a verifier error; it cannot silently
        # turn into an empty list and allow copied sample outputs through.
        baseline = self.root / "bundled-output-hashes.json"
        with self.assertRaises(FileNotFoundError):
            entry.load_reference(baseline)
        invalid = ["{broken", "null", "[]", "{}"]
        invalid.extend(
            json.dumps({"bundled_output_hashes": value}) for value in ([], "a" * 64, ["short"], ["g" * 64], [42])
        )
        for text in invalid:
            with self.subTest(baseline=text):
                baseline.write_text(text)
                with self.assertRaises(ValueError):
                    entry.load_reference(baseline)

    def test_requested_harness_and_model_reach_the_judge_and_saved_report(self):
        # Read the requested Gym agent and model from the environment, then check
        # the judge request and saved report so neither keeps a different default.
        baseline = self.root / "bundled-output-hashes.json"
        baseline.write_text(json.dumps({"bundled_output_hashes": ["a" * 64]}))
        for harness in ("codex_agent", "claude_code_agent"):
            with (
                self.subTest(harness=harness),
                patch.dict("os.environ", {"GYM_AGENT": harness, "GYM_MODEL": "requested-test-model"}),
            ):
                self.expected = entry.load_reference(baseline)
                requested = {"harness": harness, "model": "requested-test-model"}
                evidence = self.evidence()
                judgment = self.judgment(evidence)

                def requester(body, **kwargs):
                    # Check the selection in the judge request before returning
                    # a fake answer, so this test does not need a model call.
                    initial = json.loads(body["input"][0]["content"])
                    self.assertEqual(initial["gym_agent"], requested)
                    return self.response(judgment)

                result = self.run_review(evidence, requester)
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["gym_agent"], requested)
                report = json.loads((self.root / "reports/report.json").read_text())
                self.assertEqual(report["gym_agent"], requested)

    def test_invalid_gym_settings_are_configuration_errors(self):
        # Reject unknown harnesses and blank models instead of judging against
        # a fallback that differs from the user's request.
        baseline = self.root / "bundled-output-hashes.json"
        baseline.write_text(json.dumps({"bundled_output_hashes": ["a" * 64]}))
        for settings, reason in (
            ({"GYM_AGENT": "simple_agent"}, "gym_agent_harness_invalid"),
            ({"GYM_AGENT": ""}, "gym_agent_harness_invalid"),
            ({"GYM_MODEL": "  "}, "gym_agent_model_invalid"),
        ):
            with self.subTest(settings=settings), patch.dict("os.environ", settings):
                with self.assertRaisesRegex(ValueError, reason):
                    entry.load_reference(baseline)
        for missing in self.gym_env:
            with self.subTest(missing=missing), patch.dict("os.environ", {}, clear=True):
                settings = {key: value for key, value in self.gym_env.items() if key != missing}
                with patch.dict("os.environ", settings), self.assertRaises(KeyError):
                    entry.load_reference(baseline)

    def test_missing_sample_baseline_stops_before_calling_the_judge(self):
        # Report the missing fixture data before asking the model to assess a
        # workflow with an incomplete copied-sample check.
        logs = self.root / "entry"
        with (
            patch.dict("os.environ", {"VERIFIER_OUTPUT_DIR": str(logs)}),
            patch.object(entry, "load_reference", side_effect=FileNotFoundError),
            patch.object(entry, "EvidenceFiles") as collect,
            patch.object(entry, "review") as review,
            patch("builtins.print"),
        ):
            self.assertEqual(entry.main(), 1)
        report = json.loads((logs / "report.json").read_text())
        self.assertEqual(report["status"], "verifier_error")
        self.assertEqual(report["error_type"], "FileNotFoundError")
        collect.assert_not_called()
        review.assert_not_called()
        self.assertFalse((logs / "reward.txt").exists())
        self.assertFalse((logs / "reward.json").exists())

    def test_entry_point_publishes_rewards_only_after_completed_assessments(self):
        # Clear old scores before review, then publish only the completed verdict.
        # A valid Gym zero can accompany either Harbor workflow outcome.
        logs = self.root / "entry"
        for passes in (True, False):
            with self.subTest(passes=passes):
                logs.mkdir(exist_ok=True)
                (logs / "reward.txt").write_text("1\n")
                (logs / "reward.json").write_text('{"reward": 1}')
                report = {"status": "completed", "pare_success": passes, "hybrid_reward": int(passes), "gym_reward": 0}

                def review(*args, **kwargs):
                    # Neither reward format may exist while the judge is reviewing.
                    self.assertFalse((logs / "reward.txt").exists())
                    self.assertFalse((logs / "reward.json").exists())
                    self.assertEqual(json.loads((logs / "report.json").read_text())["status"], "pending")
                    return report

                with (
                    patch.dict("os.environ", {"VERIFIER_OUTPUT_DIR": str(logs)}),
                    patch.object(entry, "load_reference", return_value=self.expected),
                    patch.object(entry, "EvidenceFiles"),
                    patch.object(entry, "review", side_effect=review),
                    patch("builtins.print"),
                ):
                    self.assertEqual(entry.main(), 0)
                self.assertEqual((logs / "reward.txt").read_text(), "1\n" if passes else "0\n")
                self.assertFalse((logs / "reward.json").exists())
                self.assertEqual(json.loads((logs / "report.json").read_text()), report)

    def test_incomplete_or_errored_reviews_leave_no_reward_on_rerun(self):
        # Old text or JSON scores must not mask a review that has no valid grade.
        logs = self.root / "entry"
        for status in ("pending", "judge_error", "verifier_error"):
            with self.subTest(status=status):
                logs.mkdir(exist_ok=True)
                (logs / "reward.txt").write_text("1\n")
                (logs / "reward.json").write_text('{"reward": 1}')
                report = {"status": status, "pare_success": False, "hybrid_reward": None, "gym_reward": 0}
                with (
                    patch.dict("os.environ", {"VERIFIER_OUTPUT_DIR": str(logs)}),
                    patch.object(entry, "load_reference", return_value=self.expected),
                    patch.object(entry, "EvidenceFiles"),
                    patch.object(entry, "review", return_value=report),
                    patch("builtins.print"),
                ):
                    self.assertEqual(entry.main(), 1)
                self.assertFalse((logs / "reward.txt").exists())
                self.assertFalse((logs / "reward.json").exists())
                self.assertEqual(json.loads((logs / "report.json").read_text()), report)

    def test_unexpected_review_exception_is_reported_without_a_reward(self):
        # Keep the error diagnostic while letting Harbor report an ungraded attempt.
        logs = self.root / "entry"
        with (
            patch.dict("os.environ", {"VERIFIER_OUTPUT_DIR": str(logs)}),
            patch.object(entry, "load_reference", return_value=self.expected),
            patch.object(entry, "EvidenceFiles"),
            patch.object(entry, "review", side_effect=RuntimeError("synthetic review failure")),
            patch("builtins.print"),
        ):
            self.assertEqual(entry.main(), 1)
        report = json.loads((logs / "report.json").read_text())
        self.assertEqual(report["status"], "verifier_error")
        self.assertEqual(report["error_type"], "RuntimeError")
        self.assertFalse((logs / "reward.txt").exists())
        self.assertFalse((logs / "reward.json").exists())

    def test_invalid_completed_report_cannot_publish_a_reward(self):
        # A completed label alone cannot turn a missing or inconsistent grade
        # into a scored failure or pass.
        logs = self.root / "entry"
        for reward, passes in ((None, False), (True, True), (2, True), (0, True), (1, None)):
            with self.subTest(reward=reward, passes=passes):
                report = {"status": "completed", "pare_success": passes, "hybrid_reward": reward, "gym_reward": 0}
                with (
                    patch.dict("os.environ", {"VERIFIER_OUTPUT_DIR": str(logs)}),
                    patch.object(entry, "load_reference", return_value=self.expected),
                    patch.object(entry, "EvidenceFiles"),
                    patch.object(entry, "review", return_value=report),
                    patch("builtins.print"),
                ):
                    self.assertEqual(entry.main(), 1)
                self.assertFalse((logs / "reward.txt").exists())
                self.assertFalse((logs / "reward.json").exists())
                self.assertEqual(json.loads((logs / "report.json").read_text())["status"], "verifier_error")

    def test_judge_errors_retain_results_without_publishing_a_reward(self):
        # Exercise the actual review error path with fake requests so a provider
        # failure or timeout keeps evidence and the Gym score, without scoring it.
        logs = self.root / "entry"
        errors = [
            HTTPError("https://example.test", 503, "synthetic error", None, None),
            TimeoutError("synthetic timeout"),
            ValueError("judge_time_budget_exhausted"),
        ]
        for error in errors:
            with self.subTest(error=type(error).__name__):
                evidence = self.evidence()
                requester = MagicMock(side_effect=error)

                def review(*args, **kwargs):
                    # Redirect artifacts to the temporary test directory while
                    # using the real review logic and a fake provider failure.
                    self.assertFalse((logs / "reward.txt").exists())
                    self.assertFalse((logs / "reward.json").exists())
                    return self.run_review(evidence, requester)

                with (
                    patch.dict("os.environ", {"VERIFIER_OUTPUT_DIR": str(logs)}),
                    patch.object(entry, "load_reference", return_value=self.expected),
                    patch.object(entry, "EvidenceFiles", return_value=evidence),
                    patch.object(entry, "review", side_effect=review),
                    patch("builtins.print"),
                ):
                    self.assertEqual(entry.main(), 1)
                report = json.loads((logs / "report.json").read_text())
                self.assertEqual(report["status"], "judge_error")
                self.assertEqual(report["saved_results"][0]["gym_reward"], 0)
                self.assertTrue(list((self.root / "retained").glob("*/rollout.jsonl")))
                self.assertFalse((logs / "reward.txt").exists())
                self.assertFalse((logs / "reward.json").exists())

    def test_final_report_failure_cannot_publish_a_completed_reward(self):
        # A failure saving the final report must remain an error even after the
        # judge returned a supported assessment.
        logs = self.root / "entry"
        report = {"status": "completed", "pare_success": True, "hybrid_reward": 1, "gym_reward": 0}
        write_json = entry.write_json

        def save_report(path, value):
            # Allow the initial report but simulate failure saving the final one.
            if value["status"] == "completed":
                raise OSError("synthetic report failure")
            write_json(path, value)

        with (
            patch.dict("os.environ", {"VERIFIER_OUTPUT_DIR": str(logs)}),
            patch.object(entry, "load_reference", return_value=self.expected),
            patch.object(entry, "EvidenceFiles"),
            patch.object(entry, "review", return_value=report),
            patch.object(entry, "write_json", side_effect=save_report),
            patch("builtins.print"),
        ):
            with self.assertRaises(OSError):
                entry.main()
        self.assertFalse((logs / "reward.txt").exists())
        self.assertFalse((logs / "reward.json").exists())

    def test_reward_publication_failure_leaves_no_final_reward(self):
        # A failed rename must leave the score unpublished, rather than exposing
        # a partially written reward to Harbor.
        logs = self.root / "entry"
        report = {"status": "completed", "pare_success": True, "hybrid_reward": 1, "gym_reward": 0}
        with (
            patch.dict("os.environ", {"VERIFIER_OUTPUT_DIR": str(logs)}),
            patch.object(entry, "load_reference", return_value=self.expected),
            patch.object(entry, "EvidenceFiles"),
            patch.object(entry, "review", return_value=report),
            patch.object(Path, "replace", side_effect=OSError("synthetic publication failure")),
            patch("builtins.print"),
        ):
            with self.assertRaises(OSError):
                entry.main()
        self.assertFalse((logs / "reward.txt").exists())
        self.assertFalse((logs / "reward.json").exists())


if __name__ == "__main__":
    unittest.main()
