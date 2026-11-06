<!--
Source: https://gitlab-master.nvidia.com/hwolff/pare-harbor-tasks
Revision: 7acf882c86a724a67075ae729608500e8a95e0a7
Original file: tasks/gym-fm-01/Task.md
Source license: not declared in source file; contribution clearance pending
-->
# GYM-FM-01 — Install Gym

This is the human-facing specification. Gym runs this Harbor-format task and
gives the outer coding agent `instruction.md` and the empty attempt workspace.
This file, the verifier, and the reference solution are not part of the starting
image.

## Identity and purpose

- Product: NeMo Gym; version unpinned. Lifecycle phase: install.
- Primary persona: P3-Agent — Agent Builder with no Post-Training Experience.
- JTBD: set up the evaluation or training framework so it is ready to evaluate,
  train, or build environments.
- Capability claim: a coding agent can install a usable NeMo Gym and reach a
  working local CLI.
- Situation: an empty attempt workspace, without Gym. A usable installation is
  this persona's first barrier.
- System under test: NeMo Gym installation and CLI. Actor: outer Codex.
- In scope: installation and a CLI that runs far enough to report its version.
- Out of scope: evaluation, model calls by the installed product, use cases,
  training, OpenShell, containers as a required path, and any particular release.

## Agent input and interaction

The single-turn prompt is:

> Install NeMo Gym in this workspace and confirm its command-line interface runs. Check the Gym version to confirm the install succeeded.

No documentation URL, release number, or policy paragraph is supplied. The agent
can consult public documentation through shell network access. The Gym Codex
harness disables its web-search tool; no documentation MCP is supplied.

The intended stopping point is a working local CLI and recorded version.
Follow-on activity is not a scored criterion; the verifier does not inspect the
trajectory for prohibited actions.

## Infrastructure and initial state

- Fixture: `gym-fm-01-empty`, implemented as a fresh Docker sandbox managed by
  Gym. The task uses one sandbox without a Compose file.
- `/workspace` starts empty. No Git repository, initial commit, Gym checkout,
  virtualenv, product configuration, skills, or documentation is seeded there.
- Generic PATH tools: Bash, Git, curl, Python with venv support, uv, and Docker.
  Ubuntu's Python is not a prescribed Gym runtime; the agent can install another
  Python. Gym installs the outer Codex harness separately inside the sandbox.
- CPU only, no GPUs or product inference services. Requested limits are 4 CPUs,
  16 GiB memory, and 64 GiB disk headroom. The Docker provider does not enforce a
  filesystem quota; the host needs sufficient free disk space and memory.
- The Dockerfile's startup script starts a private Docker daemon. Gym waits for
  the task's `docker info` health check before starting agent work.
- The FM01 runner preset supplies `--privileged` for nested Docker. No host Docker
  socket, repository, virtualenv, or existing cache is mounted. Private Docker
  storage lives in the disposable sandbox container layer and is removed with
  that container. No Gym images are pre-pulled. `task.toml` supplies
  `DOCKER_HOST=unix:///var/run/docker.sock`.
- Public egress supports documentation, package/git sources, Python downloads,
  and agent-selected container images. No connection for model calls by the
  installed Gym is configured.
- The outer Codex harness routes model requests through the runner's Model
  Server. The preset lets its shell tools inherit the task sandbox environment.
  The task declares only `DOCKER_HOST` and supplies no verifier or NGC registry
  credentials. This environment setup does not isolate harness files from task
  commands.

## Accepted outcome and verification

The developer can run Gym in the attempt environment. PyPI, git, and documented
container installation are acceptable. No particular release, pinning policy,
directory layout, wrapper, or agent-written receipt is required. Installation
files may be global within the isolated attempt; the host runner's installation
is not mounted into it.

**Pass iff completed execution output provides a Gym version and Python path,
and a matching installed Gym passes provenance checks and a fresh
`gym --version` invocation.**

`verify.py` reads `/logs/agent/trajectory.json`. For ordinary Codex runs it
prefers the native session files retained under `/logs/agent/sessions`, because
Gym's ATIF export omits the terminal completion headers used by this verifier.
The adapter pairs native tool calls and outputs by call ID within each file and
keeps source-file and line references. Without retained sessions, the verifier
uses ATIF terminal observations with the supported completion headers.

For reference-solution validation, the verifier reads real output from
`/logs/agent/oracle.txt` and the shell exit status recorded by `solve.sh` in
`/logs/agent/exit-code.txt`. This supports Gym's message-only oracle trajectory
without inventing terminal observations. The installation checks remain the
same for reference solutions and coding-agent runs.

Only terminal output supplies installation candidates; assistant claims and
web-tool results are ignored. A Gym version line makes an observation relevant;
matching polling supplies any remaining output and completion status. The
reported Python path and Gym version locate an installation to inspect. No
agent command text, installer spelling, shell syntax, heredoc, activation, or
launcher implementation is parsed.

Recorded output is a locator and retained evidence. Even printed documentation
can only suggest a candidate; a matching installed product must independently
pass the checks below.

The verifier first tries the reported Python locally. If needed, it queries the
attempt's private Docker inventory: running containers, then installed images.
Only that reported Python and expected Gym version are checked. This handles an
agent-written launcher such as `./gym` without interpreting its source. It does
not search the filesystem or pull images. Running containers retain their
current writable state; image checks use temporary pull-disabled,
network-disabled containers. Stopped containers with installation changes only
in their writable layer are not resumed or inspected.

`inspect_installation.py` runs with each candidate environment's Python and checks:

- The `nemo-gym` distribution exists and its version matches the recorded output.
- It declares a `gym` console entrypoint. Its script and Python module exist, and
  the script matches its installation hash when one is recorded.
- Installed source information satisfies the accepted PyPI/Git/container checks.
  Ordinary index installations use distribution identity; `direct_url.json` and
  a local checkout's actual `origin` remote corroborate direct/Git installations.
  Official Gym image repository digests corroborate packaged source; an image
  tag alone supplies no provenance. Other images must satisfy package-source
  checks.
- The helper executes the installed distribution's actual console script with
  `--version`, a 30-second timeout, and no configured inference credentials.
  It must exit zero and report the same Gym version.

There is no reinstall, arbitrary command replay, evaluation, model call, judge,
or prescribed release. The fresh CLI check supplies execution proof. The
historical overall exit status is retained but not scored: an unrelated command
in the same shell request could have failed. A deleted installation, forged
version-printing script without an accepted installed package, broken runtime,
or mismatched version does not pass.

Evidence includes the selected execution references and output, attempted
installation locations, package/source details, and the fresh CLI command/result
in `/logs/verifier/evidence.json`. `/logs/verifier/gym-version.txt` contains the
fresh verifier invocation's output. The selected recorded text is retained as
`trajectory_output`. Each trial starts fresh; no workspace receipt is required.

Completed decisions write reward 1 or 0. Missing/corrupt required evidence,
relevant output without recorded completion, an unavailable Docker daemon, and
unexpected inspection failures produce diagnostics without a reward file.
Output without a usable version/Python locator cannot establish a pass. Known
package/CLI absence, metadata mismatches, failed CLI execution, and CLI timeout
fail the candidate. See the dataset README for how Gym records
[task and verifier failures](../README.md#read-scores-and-errors).

These are consistency and readiness checks, not a supply-chain or adversarial
anti-tampering audit. Ordinary metadata does not cryptographically prove which
package index served the files. Finding a matching current installation does not
bind it to an exact historical image ID or launcher. Any release is accepted;
Python's version is recorded, not scored. The ordinary terminal reader supports
the Codex envelope with `Process ...` before `Output:`.

## Running and validating

Use the FM01 preset and commands in
[Evaluate the outer coding agent](../README.md#evaluate-the-outer-coding-agent).
The preset selects outer Codex with `openai/openai/gpt-6.1-sol` and high reasoning.
Use 1 attempt without automatic retries. Agent, build, and verifier timeouts are
30, 10, and 5 minutes respectively.

## Developer checks

Developer tests live under `tests/unit_tests/benchmarks/pare/`. They cover
recorded-output discovery, asynchronous completion, native-session adaptation,
installation metadata, source checks, and actual CLI success/failure in temporary
Python environments. Reference-solution validation and real agent evaluation
exercise the full task; inspect their saved execution and installation evidence
alongside the reward. The dataset README gives setup and validation commands.

The optional Docker smoke test uses a cached image with Python and venv support:

```bash
GYM_VERIFIER_TEST_IMAGE="your-cached-python-image:tag" uv run pytest \
  tests/unit_tests/benchmarks/pare/test_gym_verifier_docker.py -v
```

It creates a small synthetic distribution to exercise image and running-container
inspection, including rejection of a broken CLI and missing interpreter. It
cleans up its own image and container and does not download Gym or call a model.
This checks verifier plumbing, not installation of the published Gym image or
the task's complete nested-Docker lifecycle.
