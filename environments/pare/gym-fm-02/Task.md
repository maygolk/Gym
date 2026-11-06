<!--
Source: https://gitlab-master.nvidia.com/hwolff/pare-harbor-tasks
Revision: 7acf882c86a724a67075ae729608500e8a95e0a7
Original file: tasks/gym-fm-02/Task.md
Source license: not declared in source file; contribution clearance pending
-->
# GYM-FM-02 — Evaluate the Workplace Assistant with a selected Gym agent

## Purpose and scope

This task evaluates whether an agent builder can use an installed NeMo Gym to
configure and run the bundled Workplace Assistant email example with Gym's
selected Codex or Claude Code harness, then retain a scored result for inspection.

- Case: GYM-FM-02, product use. GYM-FM-01 covers installation.
- Persona: Agent Builder with no Post-Training Experience, P3-Agent.
- JTBD: configure an agent, model and environment so they can be used together.
- Gym runs the outer coding agent. The selected inner Gym agent performs
  the email task.
- The task requires a Gym evaluation using the requested Gym agent and a
  newly generated scored Carlos rollout under `/workspace`. Gym's CLI and Python API are both
  acceptable. Gym's email reward may be zero.
- Installation, training, OpenShell and configuration/security grading are outside
  this task. The bundled simple agent does not satisfy the selected-harness requirement.

Gym gives the coding agent `instruction.md` and the installed environment.
This specification, the verifier and the reference solution are not part of its
starting workspace.

## Agent instruction

The current instruction is:

> NeMo Gym is already installed and I want to try its Workplace Assistant example, where the assistant replies to Carlos's last email about "Task Update on Develop prototype for report generation" with "Thanks for the update - I will get back to you tomorrow."
>
> I want to use the model named in `GYM_MODEL` through `OPENAI_BASE_URL` and my API key is in the `OPENAI_API_KEY` environment variable.
>
> Run the example using `GYM_AGENT` and save the result somewhere under `/workspace` so I can review it later.

The instruction leaves the agent to discover the configuration and how to run Gym.
It does not require a launcher script, a particular configuration file or a
particular adapter.

## Runtime selection

| Setting | Default | Meaning |
| --- | --- | --- |
| `GYM_AGENT` | `codex_agent` | Gym harness: `codex_agent` or `claude_code_agent`. |
| `GYM_MODEL` | `openai/openai/gpt-5.4-mini` | Requested model through the supplied Hub connection. |

Defaults live in `task.toml` under `environment.env`. Select a run preset
with `--config` to override those declared entries through the environment
server's `task_env` setting.
[`gym-fm-02-codex.yaml`](../config/gym-fm-02-codex.yaml) selects Gym's
Codex harness; [`gym-fm-02-claude.yaml`](../config/gym-fm-02-claude.yaml) selects
Gym's Claude Code harness with `aws/anthropic/bedrock-claude-haiku-5-5`.
Both use outer Codex; the filenames identify the inner Gym agent.
The selected sandbox environment reaches the coding agent, oracle and shared
verifier. The selection is runtime configuration, not a Docker build argument.
An unknown harness, missing setting or blank model is a configuration error;
it must not fall back to another harness or produce a workflow reward.
See [the dataset run guide](../README.md#evaluate-the-outer-coding-agent).

## Environment

Each attempt starts in a fresh Docker container with an empty `/workspace`.
Gym is already installed at `/opt/gym`; its bundled data, sample rollouts,
documentation and Git files are preserved. No evaluation project, running Gym
services or new scored result is prepared.

The default fixture is Gym v0.7.0, pinned to commit
`1f8afc71d45fac14872d86fe300374afa2001e7a`, with Python 3.13.14 and fixture
revision `upstream-1`. [environment/Dockerfile](environment/Dockerfile) installs
Gym's source without local patches. The reference solution uses Gym's
Responses-native mode.

The model connections and setup are documented in [README.md](README.md).
The outer agent's reasoning setting does not configure Gym's agent.
To change the installed Gym or Python version, edit the `GYM_COMMIT` or
`PYTHON_VERSION` ARG default in [environment/Dockerfile](environment/Dockerfile).
The runner presets select runtime settings; they do not override build ARGs.
Other versions need validation against the reference solution and verifier.

The task requests 2 CPUs, 4 GiB memory, 10 GiB storage and no GPU. Limits are
20 minutes for the coding agent, 2 minutes for verification and 30 minutes
for the environment build. Public network access supports documentation,
component dependencies and remote inference. Gym may install the selected
harness CLI and component dependencies when services start.

## Verification contract

GPT-6.1 Terra is the only workflow judge. [instruction.md](instruction.md) is the task request.
[tests/grading-criteria.md](tests/grading-criteria.md) contains 4 numbered criteria
and 8 saved-result evidence definitions. Result-selection and
completion conditions sit beside the criteria they support. Each criterion states
what must be true and describes supporting evidence.
[tests/judge-prompt.md](tests/judge-prompt.md) explains how to read and connect
execution records, handle uncertain or missing evidence and reference-solution
logs, apply the review boundaries, manage the review budget and return an assessment.
Workflow reward 1 requires all 4 requirements to pass; any failed requirement makes
the reward 0 for a completed assessment.

[tests/reference.json](tests/reference.json) holds the data Python uses: the exact
Carlos request, supported Gym harnesses, requirement and fact IDs, and
verification-contract version.
The contract version is `gym-fm-02-hybrid-5`. `verify.py` reads and validates the
runtime selection and adds it as `gym_agent`, with `harness` and `model` fields.
The judge receives this request context in its first evidence bundle, and both
the evidence and review report retain it. The model is connection context; it
does not add an independent upstream-model identity criterion.
Python uses the requirement IDs to build and validate the judge's
answer format. `EvidenceFiles.initial()` supplies the reference metadata together
with the evidence. `hybrid.py` loads `judge-prompt.md` and `grading-criteria.md`,
combines them for each judge request and saves the combined text as `rubric.txt`.

During the image build,
[environment/generate_sample_hashes.py](environment/generate_sample_hashes.py)
reads `workplace_assistant/data/example_rollouts.jsonl` under the installed Gym's
`resources_servers/` and `environments/` directories when present. It hashes each
row's `response.output` using SHA-256 over `json.dumps(output, sort_keys=True).encode()`
and saves unique fingerprints in `/opt/gym-fixture/bundled-output-hashes.json`.
Changing a sample's response ID, timestamp or reward does not change its output
fingerprint. The captured file stays separate from the bundled samples so later
sample edits do not automatically change the baseline. The FM02 presets mount
`/opt/gym-fixture` as a read-only named volume. Before each run, supply a fresh
UUID-based `PARE_FM02_FIXTURE_VOLUME`; remove it after Gym exits and its sandboxes
stop. Docker populates a fresh volume from the image's fixture files, so the
agent cannot overwrite the captured fingerprints through that mount. The runner
does not manage this volume's naming or cleanup. See the
[run guide](../README.md#validate-the-reference-solutions) for the commands.
Missing, empty or malformed samples stop the build.
`verify.py` loads and validates the generated file before
collecting evidence; a missing or malformed baseline is recorded as a verifier
error. It supplies the hashes to the existing saved-result comparison in
`read_execution.py`.

| Requirement ID | Source in the instruction | What the judge establishes |
| --- | --- | --- |
| `carlos_task` | The named Carlos email and requested reply | Gym evaluated the exact request and recorded real model output. Gym grades email quality; a valid zero score is acceptable. |
| `gym_evaluation` | “Run the example” | Gym actually ran the evaluation through its CLI or Python API. Shell launchers and Python subprocess wrappers are also acceptable. Configuration, a script or a saved result alone is insufficient. |
| `gym_agent` | “using GYM_AGENT” | This evaluation actually used the requested Gym harness. A different harness does not pass. The outer agent, configuration or component names alone are insufficient. |
| `saved_scored_result` | “save the result somewhere under /workspace so I can review it later.” | A genuine, fresh, inspectable scored Carlos result exists there and belongs to that same completed evaluation. |

The judge receives these saved-result facts alongside the original files:

| Fact | Evidence meaning |
| --- | --- |
| `one_scored_carlos_rollout_saved` | The selected file has exactly one row for the exact Carlos request. The reader accepts direct input text or one user message with a string or `input_text` parts, joined without a separator. |
| `gym_reward_is_valid` | The reward is a finite number from 0 to 1, not a boolean. |
| `gym_verification_not_skipped` | Gym verification was not skipped. |
| `rollout_is_not_collection_failure` | The row has no collection failure and is not excluded from scoring. |
| `model_interaction_saved` | A response ID, model label and a model tool call, nonblank assistant text, refusal or reasoning summary are recorded. Blank placeholders and tool results alone are insufficient. |
| `rollout_not_copied_from_bundled_samples` | The output does not match the known bundled sample fingerprints. |
| `response_belongs_to_attempt` | The response's `created_at` timestamp is within 1 second before container start through 120 seconds after verification begins. The judge also checks that it belongs to the selected evaluation; missing container timing can be resolved with other recorded timing. |
| `recorded_tool_calls_have_results` | Recorded model tool calls have paired results. Tool errors are allowed; tool use is not mandatory. |

These facts explain evidence for the 4 requirements; they are not a separate
verifier score. The helper stops at its first false fact. Unlisted facts mean
not reached, not passed. The judge can inspect the full result to resolve them.
These checks establish presence and structure. The judge uses the full result
and execution records to establish that a real model interaction occurred.
Candidates remain visible even when a fact is false. One valid result is enough;
failed retries, corrupt files and copies do not invalidate a separate valid run.
Candidates are presented newest first, without preferring a higher Gym reward.

The judge must connect the selected result to evidence that the requested Gym
agent ran and that the same evaluation completed successfully. It must preserve
known failures and conflicts, not borrow completion from another retry or invent
exit codes and timestamps.
Independent cleanup errors differ from an evaluation failure. A saved script is
a current snapshot; it does not establish what ran before a later edit.

No particular filename, suffix, `env.yaml`, shell syntax, adapter, metrics file,
`codex.txt`, native-session file or final terminal poll is required when other
evidence suffices. CLI and direct Python API evaluations are both acceptable,
as are separate startup and auto-serving workflows. Extra connection probes
can be legitimate. The real oracle log and known reference script may support
a reference-solution attempt without an outer agent trace; synthetic execution
evidence is not acceptable.

Configuration/security grading and independent upstream-model identity checks
are outside this verifier's scope. A constructed model label does
not establish upstream identity. Direct answers, copied samples and fabricated
rollouts do not satisfy the task. The evidence review is partial and cannot
prove that a file was never edited after execution.

## Verifier implementation and reports

| File | Responsibility |
| --- | --- |
| `environment/generate_sample_hashes.py` | Capture the installed Workplace sample output fingerprints during the Docker build. |
| `tests/test.sh` | Clear stale reward files and launch `tests/verify.py`, preserving its exit status. |
| `tests/verify.py` | Load reference data and sample fingerprints, gather evidence, save the report and publish a reward only for a completed assessment. |
| `tests/read_execution.py` | Discover workspace files, calculate saved-result facts and expose public records through read-only listing, reading and searching. |
| `tests/hybrid.py` | Ask GPT-6.1 Terra to assess the requirements, validate its answer and retain public diagnostics. |
| `tests/judge-prompt.md` | Store evidence-reading, execution-matching, reference-solution, review-budget and assessment instructions, loaded by `tests/hybrid.py`. |
| `tests/grading-criteria.md` | Store FM02's criteria, result conditions and evidence definitions, loaded by `tests/hybrid.py`. |
| `tests/reference.json` | Store the Carlos request, supported Gym harnesses, requirement and fact IDs, and verification-contract version used by Python. |

The reader supplies public tool actions, outputs, IDs and timestamps from
available outer-agent captures. It does not infer command syntax or reconstruct
execution relationships. The judge interprets that evidence and can inspect
workspace scripts, configuration and logs. Private outer-agent reasoning is
not included in the public trace view.

The first judge message contains the reference metadata, file catalog, possible-result
facts and complete result/execution files that fit its initial size allowance.
Files omitted from that first message remain accessible through 3 read-only
tools:

- `list_evidence`: list another page of available file IDs and paths.
- `read_evidence`: read a section of a file, up to 16,000 characters.
- `search_evidence`: find literal text in a file and return surrounding text.

`next_offset` continues reading after a returned section; `next_search_offset`
looks for another match. Tool responses identify shortened or withheld sections
when request limits apply. These notices describe verifier limits, not missing
source evidence. Later tool responses are saved in `file-access.json`, while
`evidence.json` remains the initial bundle.

Each nonblank text section has an `excerpt_id` identifying its file and exact
character range. The reader records the section only after budget shortening,
including complete files supplied initially. Earlier references remain valid
after later reads. The judge can request a smaller section for a precise citation.

Code validates the answer structure, all 4 requirement IDs, the selected
file/row and references to excerpts actually supplied to the judge. The judge
returns `excerpt_id` references; Python retrieves the exact text without asking
the judge to reproduce it. The report's `citations` maps each cited ID to its
source file, path, character offsets and original text in `quote`. Unknown or
undelivered references are errors. Offsets refer to the supplied evidence text;
execution records use the public trace view described above.
A valid reference establishes the source, not whether it supports the finding;
that interpretation remains the judge's responsibility. A completed nonpass
produces workflow reward 0. Judge or verifier
errors leave no reward file and exit unsuccessfully. Gym records a missing
reward as numeric 0 with `failure_kind=harbor:missing_reward`; this is different
from a completed negative assessment. The report records available error
details. Both entry points clear stale reward files before reviewing; the shell
does this before Python starts so an import error cannot preserve an old score.
The final report is saved before the reward is published through a temporary
file and rename. A valid inner Gym score of 0 can still produce workflow reward 1 when
all 4 requirements are supported.

Gym enforces `[verifier].timeout_sec` in `task.toml`, currently 120 seconds.
Under `[verifier.env]`, `VERIFIER_JUDGE_TIME_BUDGET_SEC` supplies the judge's
80-second budget, including retries, and `VERIFIER_JUDGE_FINAL_RESERVE_SEC`
reserves 15 seconds for its final assessment. Edit these declared settings in
`task.toml` when changing the judge budget. Both must be finite and positive;
the reserve is capped at one quarter of the review budget. Leave time within
Gym's overall timeout for evidence collection and report writing.

Each serialized request is limited to 300,000 characters by default. The review
allows up to 16 turns; each turn's HTTP request can retry up to 2 times for eligible
temporary HTTP/connection errors within the shared deadline. Authentication errors
and HTTP 429 are not retried, and usable judgments are not resampled.
Reaching a reading limit requests a final assessment using evidence already
received. A timeout or provider error can still end review without an assessment.
These limits do not create additional coding-agent requirements.

Gym records the host verifier folder in each trial's `paths.verifier_logs` in
`<output-stem>_run_summary.json`. Resolve relative paths from the summary's
directory; a custom rollout output can have verifier logs in the default task
result directory. Within that session folder, public outputs are:

- `reward.txt`: workflow reward 0 or 1, present only for a completed assessment.
- `report.json`: the assessment or verifier error, requested Gym agent/model,
  discovered candidate scores and the selected result's Gym score when available.
- `hybrid/`: contract, evidence catalog, rubric, answer, report, request
  diagnostics and file-access records when available.

Inside the sandbox, `/logs/artifacts/gym-fm-02-hybrid/<hash>/` retains discovered
result files and optional adjacent metrics before judging. A supported selected
result is also copied to the main hybrid artifact directory. The summary's
`paths.artifacts` identifies any collected host artifacts; null means no host
artifact path was recorded. See [the report guide](README.md#inspect-results-and-reports)
for the saved report files and artifact availability.

## Reference solution

[solution/solve.sh](solution/solve.sh)
creates its own Gym configuration in a fresh workspace. It uses the requested
Gym agent and model through Responses-native mode, starts services with
Gym's helper, evaluates the bundled Carlos row through `gym eval run --no-serve`,
retains the scored rollout and shuts services down. Its checks accept a valid
Gym zero and require response metadata and nonblank output. The judge uses the
real execution log to determine whether the evaluation succeeded. The script is
a reference implementation, not a required configuration template for the coding agent.

The harness block supplies Codex's `openai_api_key` and version or Claude Code's
`anthropic_api_key`, version and turn limit. Both use the same Gym model server,
Workplace resources and evaluation command. Claude Code reaches Hub through
Gym's Messages adapter. The oracle pins Claude Code `2.1.293` for Haiku 5.5.

See [reference validation](../README.md#validate-the-reference-solutions) for the
oracle command and [developer checks](../README.md#developer-checks)
for the local suites.

## Design and product references

The task follows the FM-02 case in the
[NeMo Gym PARE design document](https://docs.google.com/document/d/1eHmxF8ptQ34WOgyBoFNaNCJ4hE82YFASodxPS6SRrLI/edit?tab=t.8lz2fn8mpfps#heading=h.5nvwd9z7a5rf),
with the task requirements defined in `instruction.md` and the verification contract.
For this fixture, use Gym's versioned documentation and pinned source:

- [v0.7.0 CLI reference](https://docs.nvidia.com/nemo/gym/v0.7.0/reference/cli-commands/).
- [v0.7.0 quickstart](https://docs.nvidia.com/nemo/gym/v0.7.0/get-started/quickstart/).
- [Pinned Codex harness README](https://github.com/NVIDIA-NeMo/Gym/blob/1f8afc71d45fac14872d86fe300374afa2001e7a/responses_api_agents/codex_agent/README.md).
- [Pinned Claude Code harness README](https://github.com/NVIDIA-NeMo/Gym/blob/1f8afc71d45fac14872d86fe300374afa2001e7a/responses_api_agents/claude_code_agent/README.md).
- [v0.7.0 Messages adapter](https://docs.nvidia.com/nemo/gym/v0.7.0/model-server/anthropic-messages/).

The quickstart's simple-agent example does not replace this task's selected-harness
requirement. Observed workarounds and wrapper choices do not establish new task
requirements.
