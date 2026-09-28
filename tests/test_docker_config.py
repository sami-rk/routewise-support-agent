"""The container configuration, checked without Docker.

Docker is not available in every environment this project is worked on, so these
tests verify the things that actually break a `docker compose up`: a COPY
pointing at a path that no longer exists, a service missing a build context, the
console importing the app package it does not ship, or the entrypoint losing its
executable bit. The commands themselves are exercised by
`docker compose up --build`, which is documented in the README.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return (ROOT / "Dockerfile").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ui_dockerfile() -> str:
    return (ROOT / "frontend" / "Dockerfile").read_text(encoding="utf-8")


class TestCompose:
    def test_the_two_services_are_there(self, compose) -> None:
        assert set(compose["services"]) == {"api", "ui"}

    def test_both_services_have_a_build_context_that_exists(self, compose) -> None:
        for name, service in compose["services"].items():
            build = service.get("build")
            assert build, f"{name} has no build section"
            context = ROOT / (build if isinstance(build, str) else build["context"])
            assert context.exists(), f"{name} build context {context} does not exist"
            dockerfile = (ROOT / build["dockerfile"]) if isinstance(build, dict) else ROOT / "Dockerfile"
            assert dockerfile.exists(), f"{name} dockerfile {dockerfile} does not exist"

    def test_the_console_waits_for_the_api_to_be_healthy(self, compose) -> None:
        assert compose["services"]["ui"]["depends_on"]["api"]["condition"] == "service_healthy"

    def test_the_console_reaches_the_api_over_http(self, compose) -> None:
        assert compose["services"]["ui"]["environment"]["API_BASE_URL"].startswith("http://api")

    def test_the_default_router_needs_no_model_download(self, compose) -> None:
        # `laya` pulls 1.7 GB of weights, which is not what a one-command first
        # run should do.
        assert compose["services"]["api"]["environment"]["ROUTER_BACKEND"] == "${ROUTER_BACKEND:-fake}"

    def test_the_default_model_is_free(self, compose) -> None:
        assert compose["services"]["api"]["environment"]["LLM_MODELS"] == "${LLM_MODELS:-openrouter/free}"

    def test_paths_are_consistent_across_the_services(self, compose) -> None:
        environment = compose["services"]["api"]["environment"]
        data_dir = environment["DATA_DIR"]
        assert data_dir == "/data"
        assert compose["services"]["api"]["volumes"] == ["support-data:/data"]
        for variable in ("DB_PATH", "CHECKPOINT_DB_PATH", "KB_DIR"):
            assert environment[variable].startswith(data_dir), (
                f"{variable} is outside {data_dir}, so it would not survive a restart"
            )

    def test_every_environment_value_is_a_string(self, compose) -> None:
        for name, service in compose["services"].items():
            for key, value in (service.get("environment") or {}).items():
                assert isinstance(value, str), f"{name}.{key} is {type(value).__name__}"

    def test_ports_do_not_collide(self, compose) -> None:
        published = [port for service in compose["services"].values() for port in service["ports"]]
        assert len(published) == len(set(published))


class TestDockerfile:
    def test_every_copy_source_exists(self, dockerfile) -> None:
        for source in re.findall(r"^COPY (?!--from)(\S+)\s", dockerfile, re.M):
            if source in (".", "./"):
                continue
            assert (ROOT / source.rstrip("/.")).exists(), f"COPY {source} has no source"

    def test_nothing_copied_is_ignored_by_dockerignore(self, dockerfile) -> None:
        ignore = [
            line.strip()
            for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        for source in re.findall(r"^COPY (?!--from)(\S+)\s", dockerfile, re.M):
            path = source.rstrip("/.")
            if path in (".", "./"):
                continue
            assert not any((ROOT / path).match(rule) for rule in ignore), (
                f"COPY {source} is excluded by .dockerignore, so the image would be broken"
            )

    def test_the_entrypoint_is_copied_in_and_executable(self, dockerfile) -> None:
        assert "docker/entrypoint.sh" in dockerfile
        assert "chmod +x" in dockerfile
        assert (ROOT / "docker" / "entrypoint.sh").stat().st_mode & 0o111

    def test_the_entrypoint_is_the_entrypoint(self, dockerfile) -> None:
        assert 'ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]' in dockerfile

    def test_torch_is_installed_cpu_only(self, dockerfile) -> None:
        assert "download.pytorch.org/whl/cpu" in dockerfile, (
            "a CUDA torch makes the image several GB larger for no benefit"
        )

    def test_the_healthcheck_is_defined_once_in_the_image(self, dockerfile, compose) -> None:
        # Defining it in the image means plain `docker run` is checked too, and
        # the compose `service_healthy` wait reuses it.
        assert dockerfile.count("HEALTHCHECK") == 1
        assert "healthcheck" not in compose["services"]["api"]


class TestFrontendImage:
    def test_the_console_image_ships_only_the_console(self, ui_dockerfile) -> None:
        copies = re.findall(r"^COPY (?!--from)(\S+)\s", ui_dockerfile, re.M)
        assert copies == ["frontend/"], "the console needs no agent code and no torch"

    def test_the_console_never_imports_the_app_package(self) -> None:
        # The console image has no `app/`, and spec section 12 says the frontend
        # talks to the backend over HTTP and nothing else.
        for path in (ROOT / "frontend").rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            assert "from app" not in source and "import app" not in source, (
                f"{path.name} imports the app package, which the console image does not ship"
            )


@pytest.fixture(scope="module")
def entrypoint_script() -> str:
    return (ROOT / "docker" / "entrypoint.sh").read_text(encoding="utf-8")


class TestEntrypointScript:
    def test_it_is_strict(self, entrypoint_script: str) -> None:
        assert "set -euo pipefail" in entrypoint_script

    def test_it_only_initialises_once(self, entrypoint_script: str) -> None:
        assert "MARKER" in entrypoint_script
        assert "already initialised" in entrypoint_script

    def test_it_seeds_and_builds_before_serving(self, entrypoint_script: str) -> None:
        assert "scripts.seed_db" in entrypoint_script
        assert "scripts.build_kb" in entrypoint_script
        assert entrypoint_script.index("initialise") < entrypoint_script.index("uvicorn app.main:app")

    def test_the_app_directory_is_overridable(self, entrypoint_script: str) -> None:
        # /app is the in-image default, but a hardcoded path could not be
        # exercised outside a container.
        assert 'APP_DIR="${APP_DIR:-/app}"' in entrypoint_script
        assert '"/app/data' not in entrypoint_script

    def test_the_reset_script_uses_the_same_project(self) -> None:
        reset = ROOT / "scripts" / "dev_reset.sh"
        assert reset.stat().st_mode & 0o111, "dev_reset.sh must be executable"
        source = reset.read_text(encoding="utf-8")
        assert "docker compose down -v" in source
        assert "docker compose up" in source
