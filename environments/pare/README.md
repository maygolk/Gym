# PARE Gym tasks

Gym runs these two Harbor-format tasks directly: it builds their images, starts
the sandbox, runs the coding agent, calls `tests/test.sh`, and saves the results.

| Task | Agent's goal | Scoring |
| --- | --- | --- |
| FM01 | Install Gym and confirm its CLI works. | An installed Gym must pass provenance checks and a fresh verifier-owned version command. |
| FM02 | Evaluate the installed Carlos email example with the requested inner harness and save its scored result. | All four workflow criteria must pass. The separate email-quality reward can be 0 or 1. |

Each task contains `task.toml`, `instruction.md`, `environment/`, `tests/` and
`solution/`. The three files under `config/` select the runner settings. See
[FM01's specification](gym-fm-01/Task.md), [FM02's guide](gym-fm-02/README.md) and
[source attribution](PROVENANCE.md) for details.

## Prerequisites

Run the commands from the Gym repository root. Use Linux amd64, a working local
Docker daemon and CLI, outbound network access, and uv 0.13.4 or newer.

```bash
uv python install 3.13.14
uv sync --locked --python 3.13.14 --extra dev --extra sandbox --extra telemetry --extra mlflow --extra wandb
```

Export `NGC_INFERENCE_API_KEY` through your usual credential setup. The presets
use it for outer Codex/Sol, FM02's inner model, and the Terra workflow judge.
Keep credentials and generated run configurations local.

FM01 requests 4 CPUs, 16 GiB memory and 64 GiB disk, plus space for builds and
caches. Its private Docker daemon needs the privileged sandbox setting in its
preset; the host Docker socket is not mounted. FM02 requests 2 CPUs, 4 GiB memory
and 10 GiB disk. Its image installs Gym v0.7.0 at commit
`1f8afc71d45fac14872d86fe300374afa2001e7a`, with Python 3.13.14. This product
version is independent of the host Gym runner.

## Docker requirements and limits

Run each task separately with its matching preset. The Docker provider cannot
run the sandbox-group networking required by Gym's Compose path, so these tasks
use their Dockerfiles directly. The presets supply different requirements:
FM01 needs privilege; FM02 needs a read-only fixture mount. One preset cannot run
the whole collection correctly.

Before **each FM02 validation or evaluation**, export a fresh volume name in the
same terminal:

```bash
export PARE_FM02_FIXTURE_VOLUME="pare-fm02-$(cat /proc/sys/kernel/random/uuid)"
```

Docker creates this named volume on sandbox startup and copies the image's
fixture files into it. The preset mounts it read-only at `/opt/gym-fixture`;
older Docker daemons reject read-only anonymous mounts. After Gym exits and its
sandboxes stop, remove this run's volume explicitly:

```bash
docker volume rm "$PARE_FM02_FIXTURE_VOLUME"
```

Volume naming and cleanup are manual. Use a fresh name for each invocation;
do not reuse another run's volume or prune unrelated volumes. If Docker reports
the volume is still in use, inspect that run's sandbox cleanup before retrying.

The Dockerfile's `GYM_COMMIT` and `PYTHON_VERSION` ARG defaults select FM02's
fixture versions. To change them, edit those defaults and rebuild; the runtime
presets do not override build arguments. Validate any changed fixture separately.

### Missing local task image

If Docker cannot find a generated `hb__<hash>` image, Gym may still have its
build record and skip rebuilding it. With no run or preparation in progress,
back up `environments/pare/built-images.json` and remove only the entry whose
image matches the missing tag. Rerun the same Gym command to rebuild it. Leave
`compose-images.json` intact; the rebuild refreshes that image's configuration.
`builds.force=true` does not clear this cache: it builds a task's Dockerfile over
a declared prebuilt image and can still reuse a matching build record.

## Validate the reference solutions

Validation builds the image, executes `solution/solve.sh` and runs the verifier.
FM02 validation uses inner-model and judge calls. Supply a fresh fixture-volume
name before its command and remove the volume afterward as described above.

```bash
NEMO_GYM_MAX_ROLLOUT_ATTEMPTS=1 uv run gym dataset validate environments/pare/gym-fm-01 \
  --sandbox docker --config environments/pare/config/gym-fm-01-codex.yaml --concurrency 1

NEMO_GYM_MAX_ROLLOUT_ATTEMPTS=1 uv run gym dataset validate environments/pare/gym-fm-02 \
  --sandbox docker --config environments/pare/config/gym-fm-02-codex.yaml --concurrency 1
```

For the Claude reference solution, use `gym-fm-02-claude.yaml` in the FM02 command.
Each reference solution should earn workflow reward 1. Check for skipped tasks,
masked rows and `failure_kind` errors as well as the score. Validation outputs
are under `results/harbor/validation/`.

For FM01, inspect `evidence.json` and `gym-version.txt` for a real installation
and a successful verifier-owned CLI invocation. For FM02, inspect `report.json`,
its citations and the scored Carlos result as described in
[the task guide](gym-fm-02/README.md#inspect-results-and-reports).

## Evaluate the outer coding agent

All three presets select outer Codex with Sol/high through Gym's
Responses-native Model Server. The implementation name `vllm_model` identifies
the adapter to Inference Hub; it does not start local vLLM inference. The
filenames for FM02 select its **inner** harness and model:

| Preset | Inner harness/model |
| --- | --- |
| `gym-fm-01-codex.yaml` | No inner model; the task installs Gym. |
| `gym-fm-02-codex.yaml` | Codex / `openai/openai/gpt-5.4-mini` |
| `gym-fm-02-claude.yaml` | Claude Code / `aws/anthropic/bedrock-claude-haiku-5-5` |

```bash
NEMO_GYM_MAX_ROLLOUT_ATTEMPTS=1 uv run gym eval run environments/pare/gym-fm-01 \
  --agent codex_agent --model-type vllm_model --sandbox docker \
  --config environments/pare/config/gym-fm-01-codex.yaml \
  --concurrency 1 --num-repeats 1

NEMO_GYM_MAX_ROLLOUT_ATTEMPTS=1 uv run gym eval run environments/pare/gym-fm-02 \
  --agent codex_agent --model-type vllm_model --sandbox docker \
  --config environments/pare/config/gym-fm-02-codex.yaml \
  --output results/harbor/gym-fm-02-codex/rollouts.jsonl \
  --concurrency 1 --num-repeats 1

NEMO_GYM_MAX_ROLLOUT_ATTEMPTS=1 uv run gym eval run environments/pare/gym-fm-02 \
  --agent codex_agent --model-type vllm_model --sandbox docker \
  --config environments/pare/config/gym-fm-02-claude.yaml \
  --output results/harbor/gym-fm-02-claude/rollouts.jsonl \
  --concurrency 1 --num-repeats 1
```

Run the FM02 commands separately, with a fresh fixture-volume name and cleanup
for each. `--num-repeats 1` selects one rollout per task;
`NEMO_GYM_MAX_ROLLOUT_ATTEMPTS=1` separately disables automatic rollout retries.
The task deadlines are 1800 seconds for FM01 and 1200 seconds for FM02.

The FM02 presets set `GYM_AGENT` and `GYM_MODEL` through `task_env`; no terminal
exports for those settings are needed. Overrides apply to declared task
environment entries at preparation and snapshot loading and enter the effective
task digest. Switching presets therefore refuses stale `--resume` rows. Use a
fresh run and separate output paths. This override supports tasks without
Compose files or Compose overlay files and does not alter build arguments.

## Read scores and errors

Start with `rollouts_run_summary.json` and `rollouts_coverage.json` beside the
rollout output. Follow the summary's paths to the verifier logs; changing
`--output` does not put every artifact in that directory.

A completed FM02 assessment writes workflow reward 1 or 0. Judge errors leave no
reward file. Gym represents a missing reward as numeric 0 with
`failure_kind=harbor:missing_reward`; it is not a completed negative assessment.
Timeouts and invalid rewards have their own failure kinds, and provider failures
may be masked. Read these fields and the verifier report before interpreting an
aggregate score. FM02's inner email reward remains separate from workflow reward.

## Developer checks

The verifier suites use temporary fixtures and mock judge requests. The Docker
smoke test is skipped unless `GYM_VERIFIER_TEST_IMAGE` is set. These suites do not
run real model evaluations:

```bash
uv run pytest tests/unit_tests/benchmarks/pare
uv run pytest tests/unit_tests/data/tasks/harbor/test_harbor_task_env.py \
  tests/unit_tests/data/tasks/harbor/test_harbor_tasks.py \
  environment_servers/harbor/tests/test_task_env.py \
  environment_servers/harbor/tests/test_app.py
uv run pytest responses_api_agents/codex_agent/tests/test_sandbox_sessions.py \
  responses_api_agents/codex_agent/tests/test_sandbox_runner.py
```

The suites cover task verification, retained-session/oracle evidence, runtime
preset selection, and preservation of the task's inner-model credential when
outer Codex connects to Gym's proxy. Real reference validation and agent runs
are separate checks. See [FM01's specification](gym-fm-01/Task.md#developer-checks)
for its optional Docker inspection tests.
