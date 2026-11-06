#!/usr/bin/env bash
# Source: https://gitlab-master.nvidia.com/hwolff/pare-harbor-tasks
# Revision: 7acf882c86a724a67075ae729608500e8a95e0a7
# Original file: tasks/gym-fm-02/tests/test.sh
# Source license: Apache-2.0 (source SPDX notice)
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Clear rewards from a previous invocation before Python starts, so an import
# or startup error cannot leave an old score for Harbor to read.
mkdir -p /logs/verifier
rm -f /logs/verifier/reward.txt /logs/verifier/reward.json
# Ask GPT-6.1 Terra to review the saved result and public execution records.
exec /opt/gym/.venv/bin/python /tests/verify.py
