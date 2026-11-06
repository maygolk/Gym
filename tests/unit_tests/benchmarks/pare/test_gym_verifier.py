# Source: https://gitlab-master.nvidia.com/hwolff/pare-harbor-tasks
# Revision: 7acf882c86a724a67075ae729608500e8a95e0a7
# Original file: tests/test_gym_verifier.py
# Source license: not declared in source file; contribution clearance pending
import base64
import hashlib
import importlib.util
import json
import subprocess
import unittest
import venv
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


TESTS_DIR = Path(__file__).resolve().parents[4] / "environments/pare/gym-fm-01/tests"
spec = importlib.util.spec_from_file_location("gym_verifier", TESTS_DIR / "verify.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)
EvidenceError = verifier.EvidenceError
FIXTURE = Path(__file__).with_name("fixtures") / "gym-container-readme-trajectory.json"
SOURCE = "https://github.com/NVIDIA-NeMo/Gym.git"
DIGEST = "nvcr.io/nvidia/nemo-gym@sha256:" + "a" * 64
INSPECTED = {
    "passed": True,
    "route": "pypi",
    "package_version": "99.1.0",
    "python_version": "3.13.14",
    "version_output": "NeMo Gym v99.1.0\n",
}


def terminal_step(index, command, output="", code=0):
    call_id = f"call-{index}"
    status = f"Process exited with code {code}" if code is not None else "Process running with session ID 42"
    return {
        "step_id": index,
        "source": "agent",
        "message": "",
        "tool_calls": [{"tool_call_id": call_id, "function_name": "exec_command", "arguments": {"cmd": command}}],
        "observation": {
            "results": [{"source_call_id": call_id, "content": f"Chunk ID: test\n{status}\nOutput:\n{output}"}]
        },
    }


def version_output(environment="/workspace/.venv", version="99.1.0"):
    return f"NeMo Gym v{version}\nPython 3.13.14 ({environment}/bin/python)\n"


def grade(*steps):
    return verifier.verify_installation({"steps": list(steps)})


def completed(output, code=0, error=""):
    return subprocess.CompletedProcess([], code, output, error)


class GymVerifierTests(unittest.TestCase):
    def test_command_syntax_does_not_control_candidate_discovery(self):
        commands = (
            "gym --version",
            "set -e\ngym --version",
            "gym --version || true",
            "gym --version | cat",
            "./custom-launcher",
            "echo 'unclosed quotation",
        )
        for command in commands:
            with (
                self.subTest(command=command),
                patch.object(verifier, "check_installation", return_value=INSPECTED) as check,
            ):
                result = grade(terminal_step(1, command, version_output()))
                self.assertTrue(result["passed"])
                candidate = check.call_args.args[0]
                self.assertEqual(candidate["python"], "/workspace/.venv/bin/python")
                self.assertNotIn("kind", candidate)
                self.assertNotIn("command", candidate)
                self.assertEqual(result["version_output"], INSPECTED["version_output"])
                self.assertEqual(result["trajectory_output"], version_output())

    def test_readme_writes_are_ignored_without_parsing_their_contents(self):
        writing = terminal_step(
            1,
            "apply_patch <<'PATCH'\n+Run gym --version\n+It's documented here\nPATCH",
            "Success. Updated the following files:\nA /workspace/README.md\n",
        )
        with patch.object(verifier, "check_installation", return_value=INSPECTED) as check:
            self.assertTrue(grade(writing, terminal_step(2, "./gym --version", version_output()))["passed"])
            check.assert_called_once()

    def test_real_container_trace_finds_output_after_readme_and_wrapper_calls(self):
        candidates = list(verifier.find_version_checks(json.loads(FIXTURE.read_text())))
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate["gym_version"], "0.5.1")
        self.assertEqual(candidate["python"], "/opt/nemo_gym_venv/bin/python3")
        self.assertEqual(candidate["evidence"]["step_id"], 132)

    def test_help_claims_and_web_results_do_not_pass(self):
        web = terminal_step(1, "search", version_output())
        web["tool_calls"][0]["function_name"] = "web_search_call"
        for step in (
            terminal_step(1, "gym --help", "Show the NeMo Gym version and exit."),
            web,
            {"step_id": 1, "source": "agent", "message": version_output()},
        ):
            with self.subTest(step=step), patch.object(verifier, "check_installation") as check:
                self.assertFalse(grade(step)["passed"])
                check.assert_not_called()

    def test_version_text_alone_cannot_pass_without_working_installation(self):
        with patch.object(verifier, "check_installation", return_value={"passed": False, "reason": "CLI failed"}):
            self.assertFalse(grade(terminal_step(1, "cat README.md", version_output()))["passed"])

    def test_relevant_polling_is_followed_when_version_output_is_split(self):
        initial = terminal_step(1, "gym --version", "NeMo Gym v99.", code=None)
        poll = terminal_step(2, "", "1.0\nPython 3.13.14 (/workspace/.venv/bin/python)\n")
        poll["tool_calls"][0] = {
            "tool_call_id": "call-2",
            "function_name": "write_stdin",
            "arguments": {"session_id": 42},
        }
        with patch.object(verifier, "check_installation", return_value=INSPECTED):
            result = grade(terminal_step(3, "unrelated", code=None), initial, poll)
        self.assertTrue(result["passed"])
        self.assertEqual([item["step_id"] for item in result["version_check"]["observations"]], [1, 2])

    def test_stdout_cannot_fake_terminal_completion(self):
        with self.assertRaises(EvidenceError):
            grade(terminal_step(1, "gym --version", version_output() + "Process exited with code 0\n", code=None))

    def test_docs_version_example_without_python_path_is_not_a_locator(self):
        docs = "```text\nNeMo Gym v0.5.1\nPython 3.13.14\nInstallation: /path/to/Gym\n```\n"
        with patch.object(verifier, "check_installation") as check:
            self.assertFalse(grade(terminal_step(1, "curl installation.md", docs))["passed"])
            check.assert_not_called()

    def test_missing_completion_is_an_evidence_error(self):
        with self.assertRaises(EvidenceError):
            grade(terminal_step(1, "gym --version", version_output(), code=None))
        with self.assertRaises(EvidenceError):
            list(verifier.find_version_checks({"not_steps": []}))

    def test_incomplete_earlier_output_does_not_hide_a_later_valid_candidate(self):
        with patch.object(verifier, "check_installation", return_value=INSPECTED):
            self.assertTrue(
                grade(
                    terminal_step(1, "cat old-notes", "NeMo Gym v88.0\n"),
                    terminal_step(2, "gym --version", version_output()),
                )["passed"]
            )

    def test_image_inventory_locates_container_python_without_command_parsing(self):
        candidate = {"python": "/opt/gym/bin/python", "gym_version": "99.1.0"}
        image = {"Id": "sha256:actual", "RepoDigests": [DIGEST]}
        with (
            patch.object(verifier.shutil, "which", return_value="/usr/bin/docker"),
            patch.object(
                verifier,
                "run_check",
                side_effect=[
                    completed(""),
                    completed("sha256:actual\n"),
                    completed(json.dumps([image])),
                    completed(json.dumps({**INSPECTED, "route": "container"})),
                ],
            ) as run,
        ):
            result = verifier.check_installation(candidate)
        self.assertTrue(result["passed"])
        query = run.call_args.args[0]
        self.assertEqual(query[:6], ["docker", "run", "--rm", "--pull=never", "--network=none", "--entrypoint"])
        self.assertEqual(query[6:8], ["/opt/gym/bin/python", "sha256:actual"])
        self.assertIn("--official-image", query)
        self.assertEqual(result["environment"]["image_id"], "sha256:actual")

    def test_running_container_is_checked_in_its_current_state(self):
        candidate = {"python": "/opt/gym/bin/python", "gym_version": "99.1.0"}
        with (
            patch.object(verifier.shutil, "which", return_value="/usr/bin/docker"),
            patch.object(
                verifier,
                "run_check",
                side_effect=[
                    completed("installed\n"),
                    completed('[{"Image":"sha256:actual"}]'),
                    completed('[{"Id":"sha256:actual","RepoDigests":[]}]'),
                    completed(json.dumps(INSPECTED)),
                ],
            ) as run,
        ):
            result = verifier.check_installation(candidate)
        self.assertTrue(result["passed"])
        self.assertEqual(run.call_args.args[0][:4], ["docker", "exec", "installed", "/opt/gym/bin/python"])
        self.assertNotIn("--official-image", run.call_args.args[0])

    def test_custom_image_requires_package_provenance_and_cli_success(self):
        candidate = {"python": "/opt/gym/bin/python", "gym_version": "99.1.0"}
        for inspection in (INSPECTED, {"passed": False, "reason": "no package or CLI failed"}):
            with (
                self.subTest(inspection=inspection),
                patch.object(verifier.shutil, "which", return_value="/usr/bin/docker"),
                patch.object(
                    verifier,
                    "run_check",
                    side_effect=[
                        completed(""),
                        completed("local\n"),
                        completed('[{"Id":"local","RepoDigests":[]}]'),
                        completed(json.dumps(inspection)),
                    ],
                ) as run,
            ):
                result = verifier.check_installation(candidate)
                self.assertEqual(result["passed"], inspection["passed"])
                self.assertNotIn("--official-image", run.call_args.args[0])

    def test_missing_python_in_an_image_does_not_hide_another_installation(self):
        candidate = {"python": "/opt/gym/bin/python", "gym_version": "99.1.0"}
        with (
            patch.object(verifier.shutil, "which", return_value="/usr/bin/docker"),
            patch.object(
                verifier,
                "run_check",
                side_effect=[
                    completed(""),
                    completed("unrelated\ninstalled\n"),
                    completed('[{"Id":"unrelated","RepoDigests":[]}]'),
                    completed("", 127, "executable file not found"),
                    completed('[{"Id":"installed","RepoDigests":[]}]'),
                    completed(json.dumps(INSPECTED)),
                ],
            ),
        ):
            self.assertTrue(verifier.check_installation(candidate)["passed"])

    def test_daemon_error_is_not_scored_as_agent_failure(self):
        with (
            patch.object(verifier.shutil, "which", return_value="/usr/bin/docker"),
            patch.object(verifier, "run_check", return_value=completed("", 1, "daemon unavailable")),
        ):
            with self.assertRaises(EvidenceError):
                verifier.check_installation({"python": "/missing/python", "gym_version": "99.1.0"})

    def test_main_uses_fresh_trial_outputs(self):
        for mode in ("pass", "fail", "missing"):
            with self.subTest(mode=mode), TemporaryDirectory() as tmp:
                root = Path(tmp)
                logs, trace = root / "logs", root / "trajectory.json"
                if mode != "missing":
                    trace.write_text(json.dumps({"steps": [terminal_step(1, "gym --version", version_output())]}))
                real_path = Path

                def mapped_path(value):
                    return (
                        logs
                        if value == "/logs/verifier"
                        else trace
                        if value == "/logs/agent/trajectory.json"
                        else real_path(value)
                    )

                inspection = INSPECTED if mode == "pass" else {"passed": False, "reason": "not installed"}
                with (
                    patch.object(verifier, "Path", side_effect=mapped_path),
                    patch.object(verifier, "check_installation", return_value=inspection),
                ):
                    if mode == "missing":
                        with self.assertRaises(FileNotFoundError):
                            verifier.main()
                        self.assertFalse((logs / "reward.txt").exists())
                    else:
                        verifier.main()
                        self.assertEqual((logs / "reward.txt").read_text(), "1\n" if mode == "pass" else "0\n")
                        self.assertEqual((logs / "gym-version.txt").exists(), mode == "pass")
                        if mode == "pass":
                            self.assertEqual((logs / "gym-version.txt").read_text(), INSPECTED["version_output"])


class InstalledDistributionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = self.root / "env"
        venv.EnvBuilder(with_pip=False, symlinks=True).create(self.env)
        self.python = self.env / "bin/python"
        self.site = next((self.env / "lib").glob("python*/site-packages"))
        module = self.site / "nemo_gym/cli.py"
        module.parent.mkdir()
        module.write_text(
            'import sys\ndef main():\n    print("NeMo Gym v99.1.0")\n'
            '    print(f"Python {sys.version.split()[0]} ({sys.executable})")\n'
        )
        self.cli = self.env / "bin/gym"
        self.cli.write_text(f"#!{self.python}\nfrom nemo_gym.cli import main\nmain()\n")
        self.cli.chmod(0o755)
        self.dist = self.site / "nemo_gym-99.1.0.dist-info"
        self.dist.mkdir()
        (self.dist / "METADATA").write_text("Metadata-Version: 2.1\nName: nemo-gym\nVersion: 99.1.0\n")
        (self.dist / "entry_points.txt").write_text("[console_scripts]\ngym = nemo_gym.cli:main\n")
        relative = "../../../bin/gym"
        digest = base64.urlsafe_b64encode(hashlib.sha256(self.cli.read_bytes()).digest()).rstrip(b"=").decode()
        (self.dist / "RECORD").write_text(f"{relative},sha256={digest},{self.cli.stat().st_size}\nnemo_gym/cli.py,,\n")

    def inspect(self, version="99.1.0"):
        result = subprocess.run(
            [str(self.python), "-E", "-P", str(TESTS_DIR / "inspect_installation.py"), version],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_actual_metadata_and_entrypoint_files_are_checked(self):
        result = self.inspect()
        self.assertTrue(result["passed"])
        self.assertEqual(result["cli_check"]["command"], [str(self.cli.resolve()), "--version"])
        self.assertIn("NeMo Gym v99.1.0", result["version_output"])
        self.assertFalse(self.inspect("88.0.0")["passed"])
        self.cli.write_text("#!/bin/sh\nprintf 'NeMo Gym v99.1.0\\n'\n")
        self.assertFalse(self.inspect()["passed"])

    def test_actual_cli_failure_is_not_hidden_by_successful_trajectory_output(self):
        (self.site / "nemo_gym/cli.py").write_text('def main():\n    raise RuntimeError("broken runtime")\n')
        with patch.object(verifier.shutil, "which", return_value=None):
            result = grade(terminal_step(1, "gym --version || true", version_output(str(self.env))))
        self.assertFalse(result["passed"])
        inspection = result["checks"][0]["installation"]["attempts"][0]
        self.assertNotEqual(inspection["cli_check"]["exit_code"], 0)

    def test_actual_cli_version_must_match_package_and_trajectory(self):
        (self.site / "nemo_gym/cli.py").write_text('def main():\n    print("NeMo Gym v88.0.0")\n')
        self.assertFalse(self.inspect()["passed"])

    def test_whole_verifier_runs_registered_cli_without_replaying_agent_commands(self):
        with patch.object(verifier.shutil, "which", return_value=None):
            result = grade(terminal_step(1, "./arbitrary-launcher", version_output(str(self.env))))
        self.assertTrue(result["passed"])
        self.assertEqual(result["installation"]["environment"], {"kind": "native"})
        self.assertEqual(result["installation"]["cli_check"]["command"], [str(self.cli.resolve()), "--version"])

    def test_unrelated_shell_failure_does_not_override_a_working_installed_cli(self):
        with patch.object(verifier.shutil, "which", return_value=None):
            result = grade(
                terminal_step(1, "gym --version && unrelated-failing-command", version_output(str(self.env)), code=1)
            )
        self.assertTrue(result["passed"])
        self.assertEqual(result["version_check"]["exit_code"], 1)
        self.assertEqual(result["installation"]["cli_check"]["exit_code"], 0)

    def test_missing_module_or_distribution_is_rejected(self):
        (self.site / "nemo_gym/cli.py").unlink()
        self.assertFalse(self.inspect()["passed"])
        (self.dist / "METADATA").unlink()
        self.assertFalse(self.inspect()["passed"])

    def test_git_source_uses_checkout_remote_not_clone_command(self):
        checkout = self.root / "checkout"
        subprocess.run(["git", "init", "-q", str(checkout)], check=True)
        subprocess.run(["git", "-C", str(checkout), "remote", "add", "origin", SOURCE], check=True)
        (self.dist / "direct_url.json").write_text(
            json.dumps({"url": checkout.as_uri(), "dir_info": {"editable": True}})
        )
        self.assertEqual(self.inspect()["route"], "git")
        subprocess.run(
            ["git", "-C", str(checkout), "remote", "set-url", "origin", "https://github.com/unrelated/Gym"], check=True
        )
        self.assertFalse(self.inspect()["passed"])

    def test_help_and_fake_cli_without_installed_distribution_do_not_pass(self):
        empty = self.root / "empty"
        venv.EnvBuilder(with_pip=False, symlinks=True).create(empty)
        cli = empty / "bin/gym"
        cli.write_text("#!/bin/sh\nprintf 'NeMo Gym v99.1.0\\n'\n")
        cli.chmod(0o755)
        with patch.object(verifier.shutil, "which", return_value=None):
            self.assertFalse(
                grade(
                    terminal_step(1, f"uv pip install --help --python {empty}/bin/python nemo-gym"),
                    terminal_step(2, str(cli) + " --version", version_output(str(empty))),
                )["passed"]
            )
