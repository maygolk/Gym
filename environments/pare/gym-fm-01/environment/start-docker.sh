#!/usr/bin/env bash
# Source: https://gitlab-master.nvidia.com/hwolff/pare-harbor-tasks
# Revision: 7acf882c86a724a67075ae729608500e8a95e0a7
# Original file: tasks/gym-fm-01/environment/start-docker.sh
# Source license: not declared in source file; contribution clearance pending
set -euo pipefail

dockerd >/var/log/dockerd.log 2>&1 &
exec "$@"
