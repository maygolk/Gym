# Source: https://gitlab-master.nvidia.com/hwolff/pare-harbor-tasks
# Revision: 7acf882c86a724a67075ae729608500e8a95e0a7
# Original file: tests/test_gym_verifier_docker.py
# Source license: not declared in source file; contribution clearance pending
"""Opt-in Docker smoke test, using a cached Python image and no downloads.

GYM_VERIFIER_TEST_IMAGE="your-cached-python-image:tag" uv run pytest \
    tests/unit_tests/benchmarks/pare/test_gym_verifier_docker.py -v

The small synthetic distribution tests verifier plumbing, not NeMo Gym itself.
Inventory is restricted to this test's own image/container; Docker operations
and the installation helper execute for real.
"""

import json
import os
import subprocess
import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from .test_gym_verifier import grade, terminal_step, verifier, version_output


INSTALL = """
import base64, hashlib
from pathlib import Path
import venv

root = Path('/opt/verifier-smoke')
venv.EnvBuilder(with_pip=False).create(root)
site = next((root / 'lib').glob('python*/site-packages'))
module = site / 'nemo_gym/cli.py'
module.parent.mkdir()
module.write_text('import sys\\ndef main():\\n    print("NeMo Gym v99.1.0")\\n    print(f"Python {sys.version.split()[0]} ({sys.executable})")\\n')
cli = root / 'bin/gym'
cli.write_text(f'#!{root}/bin/python\\nfrom nemo_gym.cli import main\\nmain()\\n')
cli.chmod(0o755)
dist = site / 'nemo_gym-99.1.0.dist-info'
dist.mkdir()
(dist / 'METADATA').write_text('Metadata-Version: 2.1\\nName: nemo-gym\\nVersion: 99.1.0\\n')
(dist / 'entry_points.txt').write_text('[console_scripts]\\ngym = nemo_gym.cli:main\\n')
digest = base64.urlsafe_b64encode(hashlib.sha256(cli.read_bytes()).digest()).rstrip(b'=').decode()
(dist / 'RECORD').write_text(f'../../../bin/gym,sha256={digest},{cli.stat().st_size}\\nnemo_gym/cli.py,,\\n')
"""


@unittest.skipUnless(os.environ.get("GYM_VERIFIER_TEST_IMAGE"), "Set GYM_VERIFIER_TEST_IMAGE to a cached Python image")
class DockerVerifierSmokeTests(unittest.TestCase):
    def test_real_image_and_running_container_checks(self):
        image = "pare-verifier-smoke:" + uuid.uuid4().hex
        run = verifier.run_check

        def docker(*args):
            result = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            return result.stdout.strip()

        # Require a cached base; never let the build trigger a large base pull.
        base = os.environ["GYM_VERIFIER_TEST_IMAGE"]
        docker("image", "inspect", base, "--format", "{{.Id}}")
        with TemporaryDirectory() as tmp:
            context = Path(tmp)
            (context / "install.py").write_text(INSTALL)
            (context / "Dockerfile").write_text(
                f"FROM {base}\nCOPY install.py /tmp/install.py\nRUN python3 /tmp/install.py\n"
            )
            self.addCleanup(subprocess.run, ["docker", "image", "rm", image], capture_output=True)
            docker("build", "--pull=false", "--network=none", "-q", "-t", image, tmp)
        output = (
            docker(
                "run",
                "--rm",
                "--pull=never",
                "--network=none",
                "--entrypoint",
                "/opt/verifier-smoke/bin/gym",
                image,
                "--version",
            )
            + "\n"
        )
        step = terminal_step(1, "./any-launcher --version", output)
        container = docker(
            "run",
            "-d",
            "--pull=never",
            "--network=none",
            "--entrypoint",
            "/opt/verifier-smoke/bin/python",
            image,
            "-c",
            "import time; time.sleep(180)",
        )
        self.addCleanup(subprocess.run, ["docker", "rm", "-f", container], capture_output=True)

        for kind in ("image", "container"):

            def scoped_inventory(command):
                if command[:4] == ["docker", "container", "ls", "--quiet"]:
                    return subprocess.CompletedProcess(command, 0, container + "\n" if kind == "container" else "", "")
                if command[:4] == ["docker", "image", "ls", "--quiet"]:
                    return subprocess.CompletedProcess(command, 0, image + "\n" if kind == "image" else "", "")
                return run(command)

            with self.subTest(kind=kind), patch.object(verifier, "run_check", side_effect=scoped_inventory):
                result = grade(step)
                self.assertTrue(result["passed"], json.dumps(result))
                self.assertEqual(result["installation"]["environment"]["kind"], kind)
                self.assertEqual(result["installation"]["cli_check"]["exit_code"], 0)
                if kind == "container":
                    docker(
                        "exec",
                        container,
                        "/opt/verifier-smoke/bin/python",
                        "-c",
                        "import nemo_gym.cli; from pathlib import Path; Path(nemo_gym.cli.__file__).write_text('def main(): raise RuntimeError(\"broken runtime\")\\n')",
                    )
                    self.assertFalse(grade(step)["passed"], "Historical success must not hide a broken CLI")
                    self.assertFalse(grade(terminal_step(2, "anything", version_output("/missing")))["passed"])
