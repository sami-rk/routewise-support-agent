"""The free-tier rule, tested where it matters: at process startup.

The spec requires that configuring a non-free model id makes the app refuse to
start, so these tests run the entry points as subprocesses with a bad
`LLM_MODELS` and check they fail rather than booting.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PYTHON = str(ROOT / ".venv" / "bin" / "python")
if not Path(PYTHON).exists():  # pragma: no cover - depends on the environment
    PYTHON = sys.executable

BAD_ENV = {
    "LLM_MODELS": "openrouter/free,openai/gpt-4o",
    "ROUTER_BACKEND": "fake",
    "DB_PATH": "/tmp/does-not-matter.db",
}


def run(module: str, *args: str, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run `python -m <module>` with the given environment."""
    env = {**os.environ, **BAD_ENV, **(env_extra or {})}
    return subprocess.run(
        [PYTHON, "-m", module, *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


class TestStartupRefusal:
    def test_the_api_refuses_to_boot_on_a_paid_model(self) -> None:
        result = run("uvicorn", "app.main:app", "--port", "8099")
        # Either the lifespan or the import fails; both are a refusal to start.
        assert result.returncode != 0, "the server started despite a paid model"
        combined = (result.stdout + result.stderr).lower()
        assert "non-free" in combined or "gpt-4o" in combined

    def test_the_cli_refuses_to_start_on_a_paid_model(self) -> None:
        result = run("app.cli")
        assert result.returncode != 0
        assert "non-free" in (result.stdout + result.stderr).lower()

    def test_the_router_eval_refuses_on_a_paid_model(self) -> None:
        result = run("evals.run_router", "--backend", "fake")
        assert result.returncode != 0
        assert "non-free" in (result.stdout + result.stderr).lower()

    def test_the_message_names_the_offending_model(self) -> None:
        result = run("app.cli")
        assert "gpt-4o" in (result.stdout + result.stderr)

    def test_a_free_configuration_still_works(self) -> None:
        result = run("app.cli", env_extra={"LLM_MODELS": "openrouter/free"})
        # The CLI starts and waits for input, so it ends on EOF rather than erroring.
        combined = result.stdout + result.stderr
        assert "non-free" not in combined.lower()
        assert "Traceback" not in combined
