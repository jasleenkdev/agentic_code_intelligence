from src.codesearch.preprocess import (
    categorise_query,
    categorise_snippet,
    clean_text,
    normalise_code,
    query_view,
    split_sections,
)

CF = """You are given $n$ integers $a_1, a_2, \\ldots, a_n$ ($1 \\le n \\le 10^5$).
Find the maximum sum.

-----Input-----

The first line contains $n$. The second line contains the array.

-----Output-----

Print one integer.

-----Examples-----
Input
3
1 2 3

Output
6
"""

CW = """Write a function `solve` that returns the reversed string.

Examples:
solve("abc") == "cba"
"""


def test_sections_and_views():
    s = split_sections(CF)
    assert set(s) >= {"statement", "input", "output", "examples"}
    info = categorise_query(CF)
    assert info.style == "stdin"
    no_ex = query_view(info, "no_examples")
    assert "Print one integer" in no_ex and "1 2 3" not in no_ex
    assert "Input:" in no_ex
    assert query_view(info, "statement").startswith("You are given n integers")


def test_clean_text_removes_latex():
    t = clean_text("$1 \\le n \\le 10^5$ and $a \\cdot b$, \\ldots")
    assert "$" not in t and "<=" in t and "*" in t and "..." in t


def test_function_style_query():
    info = categorise_query(CW)
    assert info.style == "function"
    assert "solve" in info.function_names


def test_snippet_categories():
    assert categorise_snippet("n = int(input())\nprint(n)").style == "stdin"
    assert categorise_snippet("def solve(s):\n    return s[::-1]").style == "function"
    assert categorise_snippet("class Solution:\n    def f(self):\n        pass").style == "class"


def test_normalise_code_strips_comments_and_docstrings():
    code = 'def f(x):\n    """doc"""\n    # comment\n\n    return x  # tail\n'
    out = normalise_code(code)
    assert "doc" not in out and "comment" not in out and "tail" not in out
    assert "return x" in out
