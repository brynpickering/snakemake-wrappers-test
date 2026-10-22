"""Tests of the `download` wrapper, independent of any module.

Each test runs the test `Snakefile` in a copy of the test directory, as in snakemake-wrappers.
The example `config.yaml` is replaced by a pre-resolved configuration of the hosts under test.
Hosts are local files, a local HTTP server and the `download_fake.py` handler.
"""

import hashlib
import http.server
import os
import shutil
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

TEST_DIR = Path(__file__).parent
VERSION = "v1"
FILE = "data_north_v1.txt"
TOKEN_ENV = "DOWNLOAD_WRAPPER_TEST_TOKEN"
EXAMPLE = yaml.safe_load((TEST_DIR / "config.yaml").read_text())
SPEC = EXAMPLE["spec"]
CONTENTS = {
    "file_ok": "file",
    "api_ok": "api",
    "api_secret": "s3cret",
    "http_ok": "http",
    "http_flaky": "http",
}


class FakeServer(http.server.BaseHTTPRequestHandler):
    """Serve a file under `/ok` and `/flaky` (after one 503), HTML under `/html`, 404 elsewhere."""

    def do_GET(self) -> None:
        """Respond to a file request."""
        routes = {
            "/ok/": ("application/octet-stream", b"http"),
            "/flaky/": ("application/octet-stream", b"http"),
            "/html/": ("text/html", b"<html/>"),
        }
        prefix = "/" + self.path.split("/")[1] + "/"
        if prefix == "/flaky/" and not self.server.flaky_failed:
            self.server.flaky_failed = True
            self.send_response(503)
            self.send_header("Retry-After", "0")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if prefix not in routes:
            self.send_error(404)
            return
        content_type, body = routes[prefix]
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        """Silence request logs."""


@pytest.fixture
def http_server() -> Iterator[str]:
    """Root URL of a local HTTP server."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeServer)
    server.flaky_failed = False
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    """Copy of the test directory, next to a `master` link to the wrappers for `--wrapper-prefix`."""
    (tmp_path / "master").symlink_to(TEST_DIR.parents[1])
    return Path(
        shutil.copytree(
            TEST_DIR, tmp_path / "test", ignore=shutil.ignore_patterns("__pycache__")
        )
    )


def host(name: str, url: str, **extras: str) -> dict:
    """A resolved `kind: file` host, or a `kind: api` host of the fake handler if it has extras."""
    if not extras:
        return {"name": name, "kind": "file", "url": url, "priority": 0, "extras": {}}
    return {
        "name": name,
        "kind": "api",
        "url": url,
        "priority": 0,
        "extras": extras,
        "handler": "download_fake.py",
    }


@pytest.fixture
def hosts(http_server: str) -> dict[str, dict]:
    """Working and broken hosts of every kind, by name."""
    all_hosts = [
        host("file_ok", "file:source/{version}/{region}.txt"),
        host("file_broken", "file:///nonexistent/{file}"),
        host("api_ok", "fake://{region}", content=CONTENTS["api_ok"]),
        host("api_broken", "fake://{region}", fail="API is down"),
        host("api_secret", "fake://{region}", token_env=TOKEN_ENV),
        host("http_ok", f"{http_server}/ok/{{file}}"),
        host("http_flaky", f"{http_server}/flaky/{{file}}"),
        host("http_html", f"{http_server}/html/{{file}}"),
        host("http_404", f"{http_server}/missing/{{file}}"),
    ]
    return {item["name"]: item for item in all_hosts}


def snakemake(
    workdir: Path, file: str = FILE, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    """Download a file of the test dataset with the configuration in the work directory."""
    command = [
        "snakemake",
        "--cores",
        "1",
        "--wrapper-prefix",
        f"file://{workdir.parent}/",
        f"downloads/{VERSION}/{file}",
    ]
    return subprocess.run(
        command,
        cwd=workdir,
        env=os.environ | (env or {}),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def run_download(
    workdir: Path,
    hosts: list[dict],
    file: str = FILE,
    sha256: dict[str, str] | None = None,
    env: dict[str, str] | None = None,
    max_retries: int = 0,
) -> subprocess.CompletedProcess:
    """Download a file of the test dataset from the given hosts, in order."""
    handlers = sorted({item["handler"] for item in hosts if item["kind"] == "api"})
    settings = EXAMPLE["_downloads"]["settings"]
    timeouts = settings["timeouts"] | {"max_retries": max_retries}
    downloads = {
        "hosts": {"fake": {VERSION: hosts}},
        "settings": settings | {"timeouts": timeouts},
    }
    config = EXAMPLE | {
        "spec": SPEC | {"sha256": sha256 or {}},
        "handlers": handlers,
        "_downloads": downloads,
    }
    (workdir / "config.yaml").write_text(yaml.safe_dump(config))
    return snakemake(workdir, file, env)


def read_provenance(workdir: Path) -> dict:
    """Read the provenance file of the test download."""
    return yaml.safe_load(
        (workdir / f"downloads/{VERSION}/{FILE}.provenance.yaml").read_text()
    )


def read_log(workdir: Path, file: str = FILE) -> str:
    """Read the log of a test download."""
    return (workdir / f"logs/{VERSION}/{file}.log").read_text()


def test_example(workdir: Path):
    """The example falls back from a broken API host to a file host."""
    result = snakemake(workdir)
    assert result.returncode == 0, result.stdout
    assert read_provenance(workdir)["host"] == "file_ok"
    assert "API is down" in read_log(workdir)


@pytest.mark.parametrize(
    ("order", "expected"),
    [
        (["file_broken", "file_ok"], "file_ok"),
        (["api_broken", "file_ok"], "file_ok"),
        (["api_secret", "file_ok"], "file_ok"),
        (["file_broken", "api_ok"], "api_ok"),
        (["api_ok", "file_ok"], "api_ok"),
        (["file_ok", "api_ok"], "file_ok"),
        (["http_404", "http_ok"], "http_ok"),
        (["http_html", "api_ok"], "api_ok"),
    ],
)
def test_host_fallback(
    workdir: Path, hosts: dict[str, dict], order: list[str], expected: str
):
    """Hosts of any kind are tried in list order until one succeeds."""
    result = run_download(workdir, [hosts[name] for name in order])
    assert result.returncode == 0, result.stdout
    provenance = read_provenance(workdir)
    url = hosts[expected]["url"].format(version=VERSION, file=FILE, region="north")
    assert (provenance["host"], provenance["url"]) == (expected, url)
    assert (workdir / f"downloads/{VERSION}/{FILE}").read_text() == CONTENTS[expected]


@pytest.mark.parametrize(
    ("name", "file", "message"),
    [
        ("file_broken", FILE, "File not found"),
        ("api_broken", FILE, "API is down"),
        (
            "api_secret",
            FILE,
            f"Environment variable '{TOKEN_ENV}' (host extra 'token_env') is not set",
        ),
        ("http_404", FILE, "HTTP 404"),
        ("http_html", FILE, "Received an HTML page"),
        ("file_ok", "other.txt", "does not match any registered file"),
    ],
)
def test_download_fails(
    workdir: Path, hosts: dict[str, dict], name: str, file: str, message: str
):
    """Downloads fail with the reason of each host failure, or if the file is not registered."""
    result = run_download(workdir, [hosts[name]], file=file)
    assert result.returncode != 0
    assert message in read_log(workdir, file)


@pytest.mark.parametrize(("max_retries", "returncode"), [(1, 0), (0, 1)])
def test_retry(
    workdir: Path, hosts: dict[str, dict], max_retries: int, returncode: int
):
    """Transient HTTP errors are retried up to `max_retries` times."""
    result = run_download(workdir, [hosts["http_flaky"]], max_retries=max_retries)
    assert result.returncode == returncode, result.stdout
    if returncode:
        assert "HTTP 503" in read_log(workdir)
    else:
        assert (workdir / f"downloads/{VERSION}/{FILE}").read_text() == "http"


def test_secret(workdir: Path, hosts: dict[str, dict]):
    """Handlers read secrets from the environment variable named by a host extra."""
    result = run_download(
        workdir, [hosts["api_secret"], hosts["file_ok"]], env={TOKEN_ENV: "s3cret"}
    )
    assert result.returncode == 0, result.stdout
    assert read_provenance(workdir)["host"] == "api_secret"
    assert (workdir / f"downloads/{VERSION}/{FILE}").read_text() == "s3cret"


@pytest.mark.parametrize(
    ("sha256", "returncode"), [(hashlib.sha256(b"file").hexdigest(), 0), ("0" * 64, 1)]
)
def test_checksum(workdir: Path, hosts: dict[str, dict], sha256: str, returncode: int):
    """Files are only accepted if they match their checksum."""
    result = run_download(workdir, [hosts["file_ok"]], sha256={FILE: sha256})
    assert result.returncode == returncode, result.stdout
    if returncode:
        assert "Checksum mismatch" in read_log(workdir)
    else:
        assert read_provenance(workdir)["sha256"] == sha256


def test_provenance(workdir: Path, hosts: dict[str, dict]):
    """Provenance records the download, the spec and the handler metadata."""
    assert run_download(workdir, [hosts["api_ok"]]).returncode == 0
    provenance = read_provenance(workdir)
    assert provenance.pop("retrieved_at")
    assert provenance == {
        "dataset": "fake",
        "version": VERSION,
        "file": FILE,
        "host": "api_ok",
        "url": "fake://north",
        "sha256": hashlib.sha256(b"api").hexdigest(),
        "size_bytes": 3,
        "license": SPEC["license"],
        "tags": SPEC["tags"],
        "note": SPEC["note"],
        "handler": "fake",
        "placeholders": {"region": "north"},
    }
