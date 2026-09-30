from src.codesearch.lexical import BM25Matrix, LiteralMatcher, code_literals


def test_code_literals():
    code = 'MOD = 998244353\nprint("IMPOSSIBLE")\nprint("%d" % x)\nM = 10**9+7'
    lits = code_literals(code)
    assert {"998244353", "IMPOSSIBLE", "1000000007"} <= lits
    assert "%d" not in lits


def test_literal_matcher_scores_shared_rare_literals():
    snippets = ['print("IMPOSSIBLE")', 'print("Yes")', "x = 1"]
    lm = LiteralMatcher(snippets)
    scores = lm.score(["If there is no answer print IMPOSSIBLE. Answer modulo 998244353."])
    assert scores[0, 0] > 0 and scores[0, 1] == 0 and scores[0, 2] == 0
    assert lm.matched("print IMPOSSIBLE", 0) == ["IMPOSSIBLE"]


def test_bm25_prefers_overlapping_doc():
    bm = BM25Matrix(["def palindrome_check(s): pass", "def graph_bfs(g): pass"])
    s = bm.score(["check whether a palindrome"])
    assert s[0, 0] > s[0, 1]


def test_huge_exponent_literals_do_not_crash():
    # A real APPS test snippet contains a 10**k literal too large to print as an integer.
    lits = code_literals("INF = 10**5000\nBIG = 1e400\nM = 10 ** 9 + 7")
    assert "10**5000" in lits and "1e400" in lits and "1000000007" in lits


def test_huge_exponent_in_query_does_not_crash():
    lm = LiteralMatcher(["print(1000000000)"])
    s = lm.score(["n up to 10^{5000} and answers up to $10^9$"])
    assert s[0, 0] > 0
