# Source: https://gitlab-master.nvidia.com/hwolff/pare-harbor-tasks
# Revision: 7acf882c86a724a67075ae729608500e8a95e0a7
# Original file: tasks/gym-fm-01/tests/verify.py
# Source license: not declared in source file; contribution clearance pending
"""Locate Gym from Harbor terminal output, then check the installed CLI directly.

ATIF output supplies a version and Python path, not proof of shell semantics.
The installation helper checks package/source metadata and runs gym --version.
No agent command text is parsed or replayed.
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path


VERSION = re.compile(r"(?im)^\s*NeMo\s+Gym\s+v?(\d[^\s]*)\s*$")


class EvidenceError(RuntimeError):
    """The harness evidence or inspection infrastructure is unusable."""


def version_observations(steps, start, call_id):
    """Yield ATIF references, stdout chunks, and status for one relevant result.

    Follow the result's session polling if output arrives before completion.
    Status comes from the Codex terminal header, never from stdout itself.
    """
    call_ids, session = {call_id}, None
    for step in steps[start:]:
        if step.get("is_copied_context"):
            continue
        for call in step.get("tool_calls") or []:
            if (
                session is not None
                and call.get("function_name", "").split(".")[-1] == "write_stdin"
                and str((call.get("arguments") or {}).get("session_id")) == session
            ):
                call_ids.add(call["tool_call_id"])
        for index, result in enumerate((step.get("observation") or {}).get("results", [])):
            if result.get("source_call_id") not in call_ids:
                continue
            content = result.get("content")
            if not isinstance(content, str) or "\nOutput:\n" not in content:
                raise EvidenceError("Unsupported version observation; expected the Codex terminal envelope")
            header, _, output = content.partition("\nOutput:\n")
            completed = re.search(r"^Process exited with code (-?\d+)$", header, re.M)
            running = re.search(r"^Process running with session ID (\S+)$", header, re.M)
            if running:
                session = running[1]
            reference = {
                "step_id": step["step_id"],
                "result_index": index,
                "source_call_id": result["source_call_id"],
                **{key: step[key] for key in ("source_file", "source_line") if key in step},
            }
            yield reference, output, int(completed[1]) if completed else None
            if completed:
                return
    raise EvidenceError("Relevant version output has no recorded completion")


def find_version_checks(trajectory):
    """Yield version/Python locators from terminal output, regardless of command text.

    Read exec_command/write_stdin observations in ATIF directly. Only results with
    a Gym version line become candidates; follow their polling when necessary.
    Retain ATIF references. Installation and actual CLI success are checked later.
    """
    if not isinstance(trajectory, dict) or not isinstance(trajectory.get("steps"), list):
        raise EvidenceError("Expected an ATIF trajectory with a steps array")
    steps, problem = trajectory["steps"], None
    for index, step in enumerate(steps):
        if step.get("source") != "agent" or step.get("is_copied_context"):
            continue
        terminal_ids = {
            call["tool_call_id"]
            for call in step.get("tool_calls") or []
            if call.get("function_name", "").split(".")[-1] in {"exec_command", "write_stdin"}
        }
        for result in (step.get("observation") or {}).get("results", []):
            call_id = result.get("source_call_id")
            content = result.get("content", "")
            if call_id not in terminal_ids or not isinstance(content, str):
                continue
            _, separator, chunk = content.partition("\nOutput:\n")
            if not separator or VERSION.search(chunk) is None:
                continue
            output, references = "", []
            try:
                for reference, chunk, exit_code in version_observations(steps, index, call_id):
                    output += chunk
                    references.append(reference)
            except EvidenceError as error:
                problem = error
                continue
            version = VERSION.search(output)
            python = re.search(r"Python\s+[\d.]+\s+\((/[^\n)]+)\)", output)
            if python is None:
                # Downloaded docs can print example version output. Without an
                # interpreter path this is not a usable installation locator.
                continue
            yield {
                "python": python[1],
                "gym_version": version[1],
                "version_output": output,
                "evidence": {
                    "step_id": step["step_id"],
                    "tool_call_id": call_id,
                    "exit_code": exit_code,
                    "observations": references,
                },
            }
    if problem:
        raise problem


def run_check(command):
    """Execute a verifier-owned query without inheriting product/auth settings."""
    env = {key: os.environ[key] for key in ("PATH", "HOME", "DOCKER_HOST") if key in os.environ}
    return subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)


def inspect_environment(command, environment):
    """Run our package/CLI helper in one installed environment and read its JSON."""
    result = run_check(command)
    if result.returncode != 0:
        # Docker exec may write runtime errors to stdout rather than stderr.
        error = result.stdout + result.stderr
        if re.search(r"no such file or directory|executable file not found", error, re.I):
            return {
                "passed": False,
                "reason": "Reported Python is absent in this environment",
                "environment": environment,
            }
        raise EvidenceError("Installation query failed: " + error[-1000:])
    return {**json.loads(result.stdout), "environment": environment}


def check_installation(candidate):
    """Find a matching installation locally or in the attempt's private Docker state.

    Try the reported Python directly, then running containers and installed images.
    Query only that Python and expected Gym version. No launcher parsing, pulls,
    filesystem searches, or reinstall. The helper executes the installed CLI once.
    """
    python, version = candidate["python"], candidate["gym_version"]
    helper = Path(__file__).with_name("inspect_installation.py")
    attempts = []
    if Path(python).is_file():
        inspection = inspect_environment([python, "-E", "-P", str(helper), version], {"kind": "native"})
        if inspection["passed"]:
            return inspection
        attempts.append(inspection)
    if shutil.which("docker"):
        for kind, list_command in (
            ("container", ["docker", "container", "ls", "--quiet"]),
            ("image", ["docker", "image", "ls", "--quiet", "--no-trunc"]),
        ):
            listed = run_check(list_command)
            if listed.returncode != 0:
                raise EvidenceError("Private Docker inventory is unavailable: " + listed.stderr[-1000:])
            for target in dict.fromkeys(listed.stdout.splitlines()):
                inspected = run_check(["docker", kind, "inspect", target])
                if inspected.returncode != 0:
                    raise EvidenceError("Cannot inspect private Docker " + kind + ": " + inspected.stderr[-1000:])
                metadata = json.loads(inspected.stdout)[0]
                if kind == "container":
                    image = run_check(["docker", "image", "inspect", metadata["Image"]])
                    if image.returncode != 0:
                        raise EvidenceError("Cannot inspect the installed container image")
                    image_metadata = json.loads(image.stdout)[0]
                    command = ["docker", "exec", target, python]
                else:
                    image_metadata = metadata
                    command = [
                        "docker",
                        "run",
                        "--rm",
                        "--pull=never",
                        "--network=none",
                        "--entrypoint",
                        python,
                        image_metadata["Id"],
                    ]
                digests = [
                    item
                    for item in image_metadata.get("RepoDigests") or []
                    if re.fullmatch(r"nvcr\.io/nvidia/nemo-gym@sha256:[a-f0-9]{64}", item)
                ]
                command += ["-E", "-P", "-c", helper.read_text(), version]
                if digests:
                    command.append("--official-image")
                environment = {
                    "kind": kind,
                    "target": target,
                    "image_id": image_metadata["Id"],
                    "repo_digests": digests,
                }
                inspection = inspect_environment(command, environment)
                if inspection["passed"]:
                    return inspection
                attempts.append(inspection)
    return {"passed": False, "reason": "No matching installed Gym CLI passed its version check", "attempts": attempts}


def verify_installation(trajectory: dict | None, *, candidates: list[dict] | None = None) -> dict:
    """Require recorded version output plus an installed, working Gym CLI."""
    checks = []
    discovered = find_version_checks(trajectory) if candidates is None else candidates
    for candidate in discovered:
        inspection = check_installation(candidate)
        checks.append({"candidate": candidate, "installation": inspection})
        if inspection["passed"]:
            return {
                "passed": True,
                "gym_version": candidate["gym_version"],
                "python_version": inspection["python_version"],
                "installation": inspection,
                "version_check": candidate["evidence"],
                "trajectory_output": candidate["version_output"],
                "version_output": inspection["version_output"],
                "checks": checks,
            }
    return {"passed": False, "reason": "No recorded version output has a working installed Gym CLI", "checks": checks}


def session_trajectory(paths: list[Path]) -> dict:
    # Read the outer Codex sessions Gym retained. Keep call IDs and terminal
    # envelopes so the existing version/polling checks see real completion.
    steps = []
    for path in paths:
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            try:
                event = json.loads(line)
            except ValueError:
                continue
            payload = event.get("payload", {})
            if event.get("type") != "response_item" or not isinstance(payload, dict):
                continue
            kind = payload.get("type")
            if kind == "function_call":
                try:
                    arguments = payload.get("arguments") or {}
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                except ValueError:
                    continue
                steps.append(
                    {
                        "step_id": len(steps) + 1,
                        "source": "agent",
                        "source_file": str(path),
                        "source_line": line_number,
                        "tool_calls": [
                            {
                                "tool_call_id": payload.get("call_id"),
                                "function_name": payload.get("name", ""),
                                "arguments": arguments,
                            }
                        ],
                    }
                )
            elif kind == "function_call_output":
                # ATIF places results with their call; match by ID within this
                # session file rather than associate unrelated output by position.
                for step in reversed(steps):
                    if step["source_file"] == str(path) and step["tool_calls"][0]["tool_call_id"] == payload.get(
                        "call_id"
                    ):
                        step.setdefault("observation", {}).setdefault("results", []).append(
                            {"source_call_id": payload.get("call_id"), "content": payload.get("output", "")}
                        )
                        break
    return {"steps": steps}


def oracle_candidate(logs: Path) -> list[dict]:
    # The reference script records its actual exit status. Read its version
    # locator from the real oracle log; installation is still checked directly.
    output = (logs / "oracle.txt").read_text()
    exit_code = int((logs / "exit-code.txt").read_text().strip())
    version = VERSION.search(output)
    python = re.search(r"Python\s+[\d.]+\s+\((/[^\n)]+)\)", output)
    if version is None or python is None:
        return []
    return [
        {
            "python": python[1],
            "gym_version": version[1],
            "version_output": output,
            "evidence": {"kind": "oracle", "log": str(logs / "oracle.txt"), "exit_code": exit_code},
        }
    ]


def main():
    """Write evidence, fresh CLI output, and the completed trial's reward."""
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    try:
        trajectory = json.loads(Path("/logs/agent/trajectory.json").read_text())
        agent_logs = Path("/logs/agent")
        if (agent_logs / "oracle.txt").is_file():
            result = verify_installation(trajectory, candidates=oracle_candidate(agent_logs))
        else:
            sessions = sorted((agent_logs / "sessions").rglob("*.jsonl"))
            # Gym's ATIF export omits terminal completion headers. Prefer retained
            # sessions when present; Harbor ATIF with full headers still works.
            result = verify_installation(session_trajectory(sessions) if sessions else trajectory)
    except Exception as error:
        (logs / "evidence.json").write_text(json.dumps({"error": str(error)}, indent=2))
        raise
    (logs / "evidence.json").write_text(json.dumps(result, indent=2))
    if result["passed"]:
        (logs / "gym-version.txt").write_text(result["version_output"])
    (logs / "reward.txt").write_text("1\n" if result["passed"] else "0\n")
    print("Gym installation: PASS" if result["passed"] else "Gym installation: FAIL")


if __name__ == "__main__":
    main()
