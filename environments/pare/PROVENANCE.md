# Imported task provenance

Source repository: https://gitlab-master.nvidia.com/hwolff/pare-harbor-tasks

Exact source revision: `7acf882c86a724a67075ae729608500e8a95e0a7`.

Source `tasks/gym-fm-01/` and `tasks/gym-fm-02/` map respectively to this
dataset's `gym-fm-01/` and `gym-fm-02/`. Source `tests/test_gym_verifier.py`,
`tests/test_gym_verifier_docker.py`, `tests/test_gym_fm_02_verifier.py`,
`tests/test_gym_fm_02_hybrid.py` and
`tests/fixtures/gym-container-readme-trajectory.json` map to
`tests/unit_tests/benchmarks/pare/` with the same filenames/fixture path.
The dataset README describes the current Docker requirements and runtime limits.
The source FM02 files `tests/judge_prompt.md` and `tests/grading_criteria.md`
are named `tests/judge-prompt.md` and `tests/grading-criteria.md` here to follow
Gym's Markdown filename convention. Their original paths remain in their headers.
The source Compose files are omitted: each task needs only one sandbox, while
Gym's Compose execution requires networking capabilities absent from the Docker
provider. FM01's `DOCKER_HOST` setting is retained in its task environment.

FM02 files with SPDX notices declare Apache-2.0; those notices are retained.
FM01 and several source test/document/data files do not declare a license in
the file, and no repository LICENSE was found. Do not infer a license from
ownership or the destination repository. Before an upstream contribution,
confirm rights to contribute those files under Apache-2.0 and record that
clearance and applicable attribution in the PR description.

Copied executable/configuration source files carry their URL, revision,
original path and declared license status in their headers. Both instruction
files are preserved byte for byte, and JSON files remain valid data without
added comments. Their attribution is recorded here. The fixture follows the
same convention. These preservation choices follow the explicit requirement
to retain accepted task instructions and verification content.

Use this source URL, full revision, original-to-Gym path mapping and the
confirmed applicable license in the PR description, as required by Gym's
AGENTS.md. Clear licensing and add any required ATTRIBUTIONS.md entry before
submitting. Source-license clearance remains unresolved for the files identified
above.
