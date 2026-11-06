#!/usr/bin/env bash
# Source: https://gitlab-master.nvidia.com/hwolff/pare-harbor-tasks
# Revision: 7acf882c86a724a67075ae729608500e8a95e0a7
# Original file: tasks/gym-fm-01/solution/solve.sh
# Source license: not declared in source file; contribution clearance pending
set -euo pipefail
mkdir -p /logs/agent
# Keep the reference script's actual completion status beside its execution log.
trap 'printf "%s\n" "$?" > /logs/agent/exit-code.txt' EXIT
cd /workspace
# A reference path, not a prescribed agent procedure or Gym release.
uv venv --python 3.13
uv pip install --python .venv/bin/python nemo-gym
.venv/bin/gym --version
