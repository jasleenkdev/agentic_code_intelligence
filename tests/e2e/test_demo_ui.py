"""Browser smoke test through the real UI, real indexes, real server.

* APPS snippet search (home page): example query → ranked snippets, query category, retrieval
  passes, latency; all-versions mode with lineages and a version hint; library switch to Gladys.
* Gladys agent (`/agent`): the three demo questions; the optional LLM explanation (off by
  default) and the LLM-down path.
* Metrics: APPS section.

Run with `./run.sh e2e` (needs the Gladys index, the `apps` and `gladys` snippet libraries
(`./run.sh index-apps`, `./run.sh index-version gladys --ref …`), a built frontend and the
Playwright Chromium in `.cache/ms-playwright`). Two servers are started on free ports: one with
the configured local LLM and one with the LLM pointed at a closed port.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".cache" / "ms-playwright"))

DEMO = [
    # (question, expected route badge, a path that must appear among the results)
    ("Where is the Bluetooth-settings deeplink used?", "usage", "front/src/"),
    ("Which files call tool setValue before tool emit?", "call order", "server/lib/device/light/"),
    ("How does Gladys purge old device states?", "plain English", "server/lib/device/"),
]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_server(extra_env: dict[str, str]):
    port = _free_port()
    env = {**os.environ, **extra_env}
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.app:app", "--port", str(port)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            urllib.request.urlopen(f"{url}/api/health", timeout=1)
            return proc, url
        except OSError:
            time.sleep(0.5)
    proc.kill()
    pytest.fail("API server did not start")


@pytest.fixture(scope="module")
def browser():
    if not (ROOT / "index" / "code.db").exists():
        pytest.skip("no index: run ./run.sh index")
    if not (ROOT / "frontend" / "dist" / "index.html").exists():
        pytest.skip("no frontend build: run ./run.sh setup")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


@pytest.fixture(scope="module")
def server():
    proc, url = _start_server({})
    yield url
    proc.terminate()
    proc.wait(10)


@pytest.fixture(scope="module")
def server_llm_down():
    proc, url = _start_server({"LLM_BASE_URL": f"http://127.0.0.1:{_free_port()}/v1"})
    yield url
    proc.terminate()
    proc.wait(10)


def _ask(page, question: str) -> None:
    page.fill("[data-testid=query-input]", question)
    page.click("[data-testid=ask-button]")
    page.wait_for_selector("[data-testid=result-card]", timeout=60_000)


@pytest.mark.parametrize("question,route,path", DEMO)
def test_demo_question(browser, server, question, route, path):
    page = browser.new_page(viewport={"width": 1366, "height": 850})
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"{server}/agent")
    page.wait_for_selector("[data-testid=llm-status]")
    _ask(page, question)

    assert page.inner_text("[data-testid=route-badge]") == route
    paths = page.locator("[data-testid=result-path]").all_inner_texts()
    assert any(p.startswith(path) for p in paths), paths
    assert page.locator("[data-testid=timeline] li").count() >= 4
    assert page.locator("text=verified").count() >= 1
    if route == "call order":
        page.wait_for_selector("[data-testid=call-graph] .react-flow__node")

    # The LLM explanation is optional and off by default (generation is out of scope).
    page.wait_for_selector("[data-testid=explanation-off]", state="attached", timeout=180_000)
    assert page.locator("[data-testid=explanation]").count() == 0

    # Click-through to the full file around the cited lines.
    page.locator("[data-testid=result-path]").first.click()
    page.wait_for_selector("[role=dialog] .line-hl")
    page.keyboard.press("Escape")
    assert not errors, errors
    page.close()


def test_answers_without_llm(browser, server_llm_down):
    page = browser.new_page()
    page.goto(f"{server_llm_down}/agent")
    page.wait_for_selector("[data-testid=llm-status]:has-text('offline')")
    page.check("[data-testid=explain-toggle]")  # opt in → offline note instead of explanation
    _ask(page, DEMO[1][0])
    page.wait_for_selector("[data-testid=llm-offline]", timeout=30_000)
    assert page.locator("[data-testid=result-card]").count() >= 3
    page.close()


def test_explore_and_metrics_pages(browser, server):
    page = browser.new_page()
    page.goto(f"{server}/explore")
    page.fill("[data-testid=catalog-search]", "bluetooth")
    page.wait_for_selector("[data-testid=catalog-detail] h3")
    assert page.locator("[data-testid=catalog-list] li").count() >= 1
    page.goto(f"{server}/metrics")
    page.wait_for_selector("[data-testid=apps-metrics] table")
    assert "ablations" in page.inner_text("[data-testid=apps-metrics]").lower()
    page.close()


# ------------------------------------------------------------------ APPS snippet search


def _libraries(url: str) -> set[str]:
    import json

    with urllib.request.urlopen(f"{url}/api/cs/libraries", timeout=30) as r:
        return {lib["name"] for lib in json.load(r)}


def _search_example(page, label: str) -> None:
    page.locator("[data-testid=example-chip]", has_text=label).click()
    page.wait_for_selector("[data-testid=hit]", timeout=180_000)


def test_apps_search_flow(browser, server):
    if "apps" not in _libraries(server):
        pytest.skip("APPS library not indexed: run ./run.sh index-apps")
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(server)
    page.wait_for_selector("[data-testid=library-switch] [aria-selected=true]:has-text('APPS')")
    _search_example(page, "Max subarray sum")
    assert page.locator("[data-testid=hit]").count() == 10
    assert "stdin program" in page.inner_text("[data-testid=category]")
    passes = page.locator("[data-testid=passes] li").all_inner_texts()
    assert any("dense recall" in p for p in passes) and any("rerank" in p for p in passes)
    assert "snippets" in page.inner_text("[data-testid=latency]")
    code = page.locator("[data-testid=results] pre").first.inner_text()
    assert "max" in code.lower()
    # A pasted problem, submitted with Ctrl+Enter.
    page.fill("[data-testid=query]", "Given a string s, print YES if it is a palindrome, else NO.")
    with page.expect_response("**/api/cs/search", timeout=120_000) as resp:
        page.keyboard.press("Control+Enter")
    assert resp.value.ok
    page.wait_for_function(
        "document.querySelector('[data-testid=hit]') && "
        "document.querySelector('[data-testid=results]').innerText.includes('YES')",
        timeout=30_000,
    )
    page.wait_for_selector("[data-testid=hit]")
    assert not errors, errors
    page.close()


def test_apps_all_versions_lineages(browser, server):
    if "apps" not in _libraries(server):
        pytest.skip("APPS library not indexed: run ./run.sh index-apps")
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto(server)
    page.wait_for_selector("[data-testid=example-chip]")
    _search_example(page, "Version hint")  # "(original version)" → all-versions mode
    page.wait_for_selector("[data-testid=lineage]")
    assert page.locator("[data-testid=version-select]").input_value() == "all"
    assert "original version" in page.inner_text("aside")
    first = page.locator("[data-testid=hit]").first.inner_text()
    assert "@v1" in first
    page.close()


def test_gladys_library_search(browser, server):
    if "gladys" not in _libraries(server):
        pytest.skip("Gladys library not indexed: run ./run.sh index-version gladys --ref v5.1.1")
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto(server)
    page.locator("[data-testid=library-switch] button", has_text="Gladys").click()
    _search_example(page, "Purge old states")
    ids = page.locator("[data-testid=hit-id]").all_inner_texts()
    assert any(i.startswith("server/") for i in ids), ids
    page.close()
