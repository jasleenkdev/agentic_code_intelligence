from src.agent.verifier import verify
from tests.conftest import FIXTURE_REPO


def test_verify_accepts_real_range_and_rejects_bad_ones():
    ok = verify(
        "server/lib/radio/radio.connect.js", 1, 4, expect=["prepare"], repo_root=FIXTURE_REPO
    )
    assert ok and ok.text.startswith("async function connect(id)")
    assert (
        verify(
            "server/lib/radio/radio.connect.js",
            1,
            4,
            expect=["nonexistentToken"],
            repo_root=FIXTURE_REPO,
        )
        is None
    )
    assert verify("server/lib/radio/radio.connect.js", 500, 510, repo_root=FIXTURE_REPO) is None
    assert verify("../../etc/passwd", 1, 2, repo_root=FIXTURE_REPO) is None
    assert verify("server/nope.js", 1, 2, repo_root=FIXTURE_REPO) is None


def test_verify_clamps_end_and_keeps_highlights():
    sn = verify("front/src/utils.js", 1, 999, highlight=[2], repo_root=FIXTURE_REPO)
    assert sn.end_line == 4 and sn.highlight_lines == [2]
