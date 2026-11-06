# Source: https://gitlab-master.nvidia.com/hwolff/pare-harbor-tasks
# Revision: 7acf882c86a724a67075ae729608500e8a95e0a7
# Original file: tests/test_gym_fm_02_verifier.py
# Source license: not declared in source file; contribution clearance pending
"""Check saved-result evidence supplied to the LLM judge, without running Gym."""

import hashlib
import importlib.util
import json
import os
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


TESTS_DIR = Path(__file__).resolve().parents[4] / "environments/pare/gym-fm-02/tests"
SPEC = importlib.util.spec_from_file_location("fm02_result_evidence", TESTS_DIR / "read_execution.py")
evidence_reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evidence_reader)
HASH_SPEC = importlib.util.spec_from_file_location(
    "fm02_sample_hashes", TESTS_DIR.parent / "environment/generate_sample_hashes.py"
)
sample_hashes = importlib.util.module_from_spec(HASH_SPEC)
HASH_SPEC.loader.exec_module(sample_hashes)


class SavedResultEvidenceTests(unittest.TestCase):
    def setUp(self):
        # Create one synthetic scored Carlos row inside a temporary workspace.
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.rollout = self.workspace / "rollout.jsonl"
        self.expected = json.loads((TESTS_DIR / "reference.json").read_text())
        self.expected["bundled_output_hashes"] = []
        self.expected["gym_agent"] = {"harness": "codex_agent", "model": "test-model"}
        self.window = {"container_started_at": 1000, "verifier_started_at": 2000}
        self.row = {
            "responses_create_params": {
                "input": [{"role": "user", "content": self.expected["task_request"]}],
            },
            "reward": 0,
            "response": {
                "id": "test-response",
                "model": "test-model",
                "created_at": 1500,
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "A nonblank test reply."}],
                    }
                ],
            },
        }
        self.save()

    def save(self, row=None, path=None):
        # Save a synthetic row without imposing a filename or directory convention.
        path = self.rollout if path is None else path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.row if row is None else row) + "\n")
        return path

    def facts(self, rows=None):
        # Calculate evidence facts only; the judge, not this helper, assigns Harbor's verdict.
        return evidence_reader.result_facts([self.row] if rows is None else rows, self.expected, self.window)

    def evidence(self):
        # Discover temporary workspace files without reading real agent records or an oracle.
        return evidence_reader.EvidenceFiles(
            self.expected,
            workspace=self.workspace,
            agent_root=self.root / "missing-agent",
            solution=self.root / "missing-solution",
            window=self.window,
        )

    def test_valid_zero_gym_score_supplies_all_eight_facts(self):
        # Zero email quality still supplies a valid scored-result fact to the judge.
        facts = self.facts()
        self.assertEqual(set(facts), set(self.expected["verification_contract"]["saved_result_evidence"]))
        self.assertTrue(all(facts.values()))
        self.assertEqual(self.evidence().candidates[0]["gym_reward"], 0)

    def test_rollout_requires_exactly_one_carlos_task(self):
        # Check that missing, wrong or multiple rows do not count as one Carlos result.
        wrong = deepcopy(self.row)
        wrong["responses_create_params"]["input"][0]["content"] = "A different email task."
        for rows in ([], [None], [wrong], [self.row, self.row]):
            with self.subTest(rows=rows):
                facts = self.facts(rows)
                self.assertFalse(facts["one_scored_carlos_rollout_saved"])
                self.assertEqual(len(facts), 1)

    def test_same_carlos_text_is_recognized_in_supported_input_layouts(self):
        # Gym accepts direct input text and user-message text parts. Each layout
        # must supply the same task fact when the actual Carlos text is unchanged.
        request = self.expected["task_request"]
        layouts = {
            "direct_text": request,
            "message_text": [{"role": "user", "content": request}],
            "one_text_part": [{"role": "user", "content": [{"type": "input_text", "text": request}]}],
            "split_text_parts": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": request[:20]},
                        {"type": "input_text", "text": request[20:]},
                    ],
                }
            ],
            "system_context": [
                {"role": "system", "content": "Use the Workplace tools."},
                {"role": "user", "content": [{"type": "input_text", "text": request}]},
            ],
        }
        for name, inputs in layouts.items():
            with self.subTest(layout=name):
                self.row["responses_create_params"]["input"] = inputs
                self.save()
                facts = self.evidence().candidates[0]["facts"]
                self.assertEqual(set(facts), set(self.expected["verification_contract"]["saved_result_evidence"]))
                self.assertTrue(all(facts.values()))

    def test_input_layout_support_preserves_exact_task_and_single_user_checks(self):
        # Recognizing text parts must not accept a changed task, extra user
        # messages or non-text content that the task comparison cannot interpret.
        request = self.expected["task_request"]
        invalid_inputs = {
            "wrong_direct_text": "A different email task.",
            "changed_whitespace": request + " ",
            "wrong_text_part": [
                {"role": "user", "content": [{"type": "input_text", "text": "A different email task."}]}
            ],
            "mixed_content": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": request},
                        {"type": "input_image", "image_url": "https://example.test/image.png"},
                    ],
                }
            ],
            "wrong_part_type": [{"role": "user", "content": [{"type": "output_text", "text": request}]}],
            "missing_text": [{"role": "user", "content": [{"type": "input_text"}]}],
            "nonstring_text": [{"role": "user", "content": [{"type": "input_text", "text": 1}]}],
            "malformed_part": [{"role": "user", "content": [request]}],
            "extra_user_message": [
                {"role": "user", "content": request},
                {"role": "user", "content": "Another request."},
            ],
            "assistant_only": [{"role": "assistant", "content": request}],
            "malformed_input": [None],
        }
        for name, inputs in invalid_inputs.items():
            with self.subTest(layout=name):
                self.row["responses_create_params"]["input"] = inputs
                self.assertFalse(self.facts()["one_scored_carlos_rollout_saved"])

    def test_gym_reward_requires_a_finite_number_between_zero_and_one(self):
        # Keep fractional scores and reject booleans, missing scores and invalid ranges.
        for reward in (0, 0.5, 1):
            with self.subTest(valid_reward=reward):
                self.row["reward"] = reward
                self.assertTrue(self.facts()["gym_reward_is_valid"])
        for reward in (-0.1, 1.1, True, False, None, "1", float("nan"), float("inf"), -float("inf")):
            with self.subTest(invalid_reward=reward):
                self.row["reward"] = reward
                self.assertFalse(self.facts()["gym_reward_is_valid"])

    def test_gym_verification_must_not_be_skipped(self):
        # Report skipped Gym scoring as false evidence rather than a separate verifier score.
        for skipped in (False, True):
            with self.subTest(skipped=skipped):
                self.row["verification_skipped"] = skipped
                self.assertEqual(self.facts()["gym_verification_not_skipped"], not skipped)

    def test_collection_failure_or_exclusion_is_reported(self):
        # Preserve evidence of collection failures and exclusion from scoring.
        baseline = {"failure_kind": None, "failure_reason": None, "mask_sample": 0}
        for field, value in (
            ("failure_kind", "collection_error"),
            ("failure_reason", "No response collected"),
            ("mask_sample", 1),
        ):
            with self.subTest(field=field):
                self.row.update(baseline)
                self.row[field] = value
                self.assertFalse(self.facts()["rollout_is_not_collection_failure"])

    def test_output_matching_a_bundled_sample_is_reported(self):
        # Changing response IDs and timestamps does not hide copied sample output.
        digest = hashlib.sha256(json.dumps(self.row["response"]["output"], sort_keys=True).encode()).hexdigest()
        self.expected["bundled_output_hashes"].append(digest)
        self.row["response"].update(id="different-id", created_at=1900)
        self.assertFalse(self.facts()["rollout_not_copied_from_bundled_samples"])

    def test_generated_hashes_detect_copies_with_changed_response_metadata(self):
        # Capture a sample, then check that changing its ID, time or reward does
        # not hide its copied output. A different output must remain distinguishable.
        gym = self.root / "gym"
        self.save(path=gym / "resources_servers/workplace_assistant/data/example_rollouts.jsonl")
        baseline = self.root / "fixture/bundled-output-hashes.json"
        sample_hashes.write_sample_hashes(gym, baseline)
        self.expected.update(json.loads(baseline.read_text()))
        self.row["response"].update(id="copied-with-new-id", created_at=1900)
        self.row["reward"] = 1
        self.assertFalse(self.facts()["rollout_not_copied_from_bundled_samples"])
        self.row["response"]["output"][0]["content"][0]["text"] = "A different model output."
        self.assertTrue(self.facts()["rollout_not_copied_from_bundled_samples"])

    def test_generated_hashes_remain_a_snapshot_until_samples_are_recaptured(self):
        # Later sample edits must not change the captured baseline. Rebuilding
        # from changed samples must produce fingerprints for the new outputs.
        gym = self.root / "gym"
        source = self.save(path=gym / "resources_servers/workplace_assistant/data/example_rollouts.jsonl")
        baseline = self.root / "fixture/bundled-output-hashes.json"
        sample_hashes.write_sample_hashes(gym, baseline)
        original = baseline.read_bytes()
        self.expected.update(json.loads(original))
        self.row["response"]["output"][0]["content"][0]["text"] = "A changed upstream sample."
        self.save(path=source)
        self.assertEqual(baseline.read_bytes(), original)
        self.assertTrue(self.facts()["rollout_not_copied_from_bundled_samples"])
        sample_hashes.write_sample_hashes(gym, baseline)
        self.assertNotEqual(baseline.read_bytes(), original)
        self.expected.update(json.loads(baseline.read_text()))
        self.assertFalse(self.facts()["rollout_not_copied_from_bundled_samples"])

    def test_duplicate_workplace_layouts_keep_each_output_fingerprint_once(self):
        # Gym can bundle identical samples in both Workplace directories. Keep
        # both distinct outputs while collapsing duplicate rows and copies.
        gym = self.root / "gym"
        source = self.save(path=gym / "resources_servers/workplace_assistant/data/example_rollouts.jsonl")
        other = deepcopy(self.row)
        other["response"]["output"][0]["content"][0]["text"] = "Another sample: café."
        source.write_text("\n".join(json.dumps(row) for row in (self.row, other, self.row)) + "\n")
        self.save(path=gym / "environments/workplace_assistant/data/example_rollouts.jsonl")
        baseline = self.root / "fixture/bundled-output-hashes.json"
        sample_hashes.write_sample_hashes(gym, baseline)
        self.expected.update(json.loads(baseline.read_text()))
        self.assertEqual(len(self.expected["bundled_output_hashes"]), 2)
        for row in (self.row, other):
            self.assertFalse(self.facts([row])["rollout_not_copied_from_bundled_samples"])

    def test_missing_empty_or_malformed_samples_cannot_generate_a_baseline(self):
        # Stop generation when the selected Gym fixture cannot supply usable
        # samples, rather than creating an empty list that disables copy detection.
        gym = self.root / "gym"
        baseline = self.root / "fixture/bundled-output-hashes.json"
        with self.assertRaises(FileNotFoundError):
            sample_hashes.write_sample_hashes(gym, baseline)
        source = gym / "resources_servers/workplace_assistant/data/example_rollouts.jsonl"
        source.parent.mkdir(parents=True)
        for text in ("", "\n ", "{broken", "{}", "null", '{"response":{"output":null}}'):
            with self.subTest(sample=text):
                source.write_text(text)
                with self.assertRaises((ValueError, KeyError, TypeError)):
                    sample_hashes.write_sample_hashes(gym, baseline)
                self.assertFalse(baseline.exists())

    def test_response_timestamp_keeps_the_existing_rounding_margins(self):
        # Keep the one-second lower and 120-second upper container-time margins.
        for created in (999, 1000, 1500, 2000, 2120):
            with self.subTest(valid_time=created):
                self.row["response"]["created_at"] = created
                self.assertTrue(self.facts()["response_belongs_to_attempt"])
        for created in (998, 2121, None, "1500", float("nan"), float("inf"), 10**400):
            with self.subTest(invalid_time=created):
                self.row["response"]["created_at"] = created
                self.assertFalse(self.facts()["response_belongs_to_attempt"])

    def test_model_output_requires_more_than_a_blank_placeholder(self):
        # Accept model text, refusals, reasoning summaries and calls; reject tool results alone.
        valid = [
            self.row["response"]["output"],
            [{"type": "message", "role": "assistant", "content": [{"type": "refusal", "refusal": "Declined."}]}],
            [{"type": "reasoning", "summary": [{"type": "summary_text", "text": "A public summary."}]}],
            [{"type": "function_call", "call_id": "call-one", "name": "email_search_emails"}],
        ]
        invalid = [
            [],
            [{"type": "message", "role": "assistant", "content": []}],
            [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": " "}]}],
            [{"type": "function_call_output", "call_id": "call-one", "output": "A tool result."}],
        ]
        for output, passes in [(value, True) for value in valid] + [(value, False) for value in invalid]:
            with self.subTest(output=output):
                self.row["response"]["output"] = output
                self.assertEqual(self.facts()["model_interaction_saved"], passes)
        for field in ("id", "model"):
            with self.subTest(missing=field):
                row = deepcopy(self.row)
                row["response"]["output"] = valid[0]
                row["response"].pop(field)
                self.assertFalse(self.facts([row])["model_interaction_saved"])

    def test_recorded_tool_calls_need_matching_results(self):
        # A real tool error can be paired, while a missing tool response remains false evidence.
        self.row["response"]["output"] = [
            {"type": "function_call", "call_id": "call-one", "name": "email_search_emails"}
        ]
        self.assertFalse(self.facts()["recorded_tool_calls_have_results"])
        self.row["response"]["output"].append(
            {"type": "function_call_output", "call_id": "call-one", "output": "A recorded tool error."}
        )
        self.assertTrue(self.facts()["recorded_tool_calls_have_results"])

    def test_invalid_newest_result_is_visible_without_preferring_a_higher_reward(self):
        # Present newest first, including invalid rows; let the judge choose a valid result.
        older = deepcopy(self.row)
        older["response"]["created_at"] = 1400
        older["reward"] = 1
        self.save(older, self.workspace / "older")
        newest = deepcopy(self.row)
        newest["response"]["created_at"] = 1800
        newest["reward"] = None
        self.save(newest, self.workspace / "newest")
        candidates = self.evidence().candidates
        self.assertEqual([item["created_at"] for item in candidates], [1800, 1500, 1400])
        self.assertFalse(candidates[0]["facts"]["gym_reward_is_valid"])
        self.assertEqual(candidates[1]["gym_reward"], 0)
        self.assertEqual(candidates[2]["gym_reward"], 1)

    def test_discovery_keeps_complete_results_among_corrupt_files(self):
        # Empty, binary and corrupt files cannot hide a separate complete result.
        (self.workspace / "empty").write_bytes(b"")
        (self.workspace / "binary").write_bytes(b"\0binary")
        (self.workspace / "corrupt").write_text("{unfinished")
        (self.workspace / "truncated").write_text(json.dumps(self.row) + "\n{unfinished")
        self.assertEqual(len(self.evidence().candidates), 1)

    def test_discovery_accepts_hidden_and_suffixless_files(self):
        # Keep the instructed workspace boundary without requiring a particular filename.
        self.rollout.unlink()
        for relative in ("results/custom", ".hidden/run.txt", "nested/out.data"):
            with self.subTest(path=relative):
                path = self.save(dict(reversed(list(self.row.items()))), self.workspace / relative)
                path.write_bytes(b"\n \n" + path.read_bytes())
                self.assertEqual(self.evidence().candidates[0]["path"], path)
                path.unlink()

    def test_candidate_order_uses_response_time_not_file_modification_time(self):
        # A recently copied old row cannot displace a newer response in the evidence list.
        newer = deepcopy(self.row)
        newer["response"]["created_at"] = 1800
        path = self.save(newer, self.workspace / "newer")
        os.utime(path, (10, 10))
        os.utime(self.rollout, (999999, 999999))
        self.assertEqual(self.evidence().candidates[0]["path"], path)

    def test_workspace_links_do_not_expand_into_unrelated_directories(self):
        # Internal links are usable, but an external directory cannot expand the search tree.
        self.save(path=self.root / "outside-result")
        (self.workspace / "internal").symlink_to(self.workspace, target_is_directory=True)
        (self.workspace / "external").symlink_to(self.root, target_is_directory=True)
        self.assertEqual(len(self.evidence().candidates), 1)

    def test_file_links_and_directory_cycles_do_not_duplicate_candidates(self):
        # Follow an ordinary file once even when several directory entries point to it.
        (self.workspace / "alias").symlink_to(self.rollout)
        os.link(self.rollout, self.workspace / "hardlink")
        (self.workspace / "cycle").symlink_to(self.workspace, target_is_directory=True)
        self.assertEqual(len(self.evidence().candidates), 1)

    def test_excluded_files_and_their_aliases_are_not_opened(self):
        # Synthetic excluded files cannot be opened through a differently named link.
        protected = self.save(path=self.workspace / "protected" / "record")
        secret = self.save(path=self.workspace / ".env")
        (self.workspace / "secret-alias").symlink_to(secret)
        original_open = Path.open

        def guarded_open(path, *args, **kwargs):
            # Fail if evidence discovery opens either excluded synthetic file.
            self.assertNotIn(path, (protected, secret))
            return original_open(path, *args, **kwargs)

        with (
            patch.object(evidence_reader, "EXCLUDED_PATHS", (*evidence_reader.EXCLUDED_PATHS, protected.parent)),
            patch.object(Path, "open", guarded_open),
        ):
            self.assertEqual(len(self.evidence().candidates), 1)
        self.assertFalse(evidence_reader._excluded(Path("/dev/shm/result")))
        self.assertTrue(evidence_reader._excluded(Path("/dev/shm/.env")))
        self.assertTrue(evidence_reader._excluded(Path("/logs/agent/sessions/outer-session.jsonl")))
        self.assertFalse(evidence_reader._excluded(Path("/logs/agent/sessions-backup/score")))

    def test_retention_keeps_the_original_row_and_optional_metrics(self):
        # Preserve discovered bytes for review without assigning a separate verifier reward.
        metrics = self.rollout.with_name("rollout_aggregate_metrics.json")
        metrics.write_text('{"reward": 0}')
        evidence = self.evidence()
        artifacts = self.root / "retained"
        evidence.retain(artifacts)
        digest = hashlib.sha256(self.rollout.read_bytes()).hexdigest()
        retained = artifacts / digest / "rollout.jsonl"
        self.assertEqual(retained.read_bytes(), self.rollout.read_bytes())
        self.assertEqual(retained.with_name("aggregate_metrics.json").read_bytes(), metrics.read_bytes())

    def test_container_window_reads_start_ticks_after_parenthesized_name(self):
        # Spaces and parentheses in PID 1's name must not shift its start-time field.
        # Supply the Linux clock constant on macOS too; clock_gettime is mocked.
        fields = ["S"] + ["0"] * 18 + ["50000"]
        for name in ("sleep", "a name (with parentheses)"):
            with (
                self.subTest(name=name),
                patch.object(Path, "read_text", return_value=f"1 ({name}) " + " ".join(fields)),
                patch.object(evidence_reader.os, "sysconf", return_value=100),
                patch.object(evidence_reader.time, "time", return_value=2000),
                patch.object(evidence_reader.time, "CLOCK_BOOTTIME", 0, create=True),
                patch.object(evidence_reader.time, "clock_gettime", return_value=1500),
            ):
                self.assertEqual(evidence_reader.container_window(), self.window)

    def test_missing_or_invalid_clock_leaves_timing_for_the_judge_to_resolve(self):
        # Missing timing does not invent a timestamp or automatically reject the workflow.
        # Supply the Linux clock constant on macOS too; clock_gettime is mocked.
        with (
            patch.object(Path, "read_text", side_effect=OSError),
            patch.object(evidence_reader.time, "time", return_value=2000),
        ):
            self.assertEqual(
                evidence_reader.container_window(), {"container_started_at": None, "verifier_started_at": 2000}
            )
        fields = ["S"] + ["0"] * 18 + ["50000"]
        for ticks, boot in ((0, 1500), (100, 100), (100, float("nan"))):
            with (
                self.subTest(ticks=ticks, boot=boot),
                patch.object(Path, "read_text", return_value="1 (sleep) " + " ".join(fields)),
                patch.object(evidence_reader.os, "sysconf", return_value=ticks),
                patch.object(evidence_reader.time, "time", return_value=2000),
                patch.object(evidence_reader.time, "CLOCK_BOOTTIME", 0, create=True),
                patch.object(evidence_reader.time, "clock_gettime", return_value=boot),
            ):
                self.assertEqual(
                    evidence_reader.container_window(), {"container_started_at": None, "verifier_started_at": 2000}
                )


if __name__ == "__main__":
    unittest.main()
