from eval.structural_checker import agreement, blank_comments_and_strings, call_sites, scan_file

SRC = """const re = /["{]/g; // a { in a comment
async function forward({ message, image }) {
  const a = strip(message);   // strip( in comment
  if (a) { check(a); }
  return "check(";
}
const load = (file) =>
  new Promise((resolve) => resolve(file));
"""


def test_lexer_blanks_comments_strings_and_regex():
    text = blank_comments_and_strings(SRC)
    assert text.count("{") == text.count("}")
    assert "strip(" in text and text.count("check(") == 1
    assert len(text) == len(SRC) and text.count("\n") == SRC.count("\n")


def test_destructured_params_and_expression_arrows():
    fs = scan_file("x.js", SRC)
    names = {(f.name, f.start_line, f.end_line) for f in fs.functions}
    assert ("forward", 2, 6) in names
    assert any(n == "load" and s == 7 for n, s, _ in names)
    offs = call_sites(fs, "check")
    assert len(offs) == 1 and fs.innermost(offs[0]).name == "forward"


def test_agreement_overlap():
    idx = [{"file_path": "a.js", "start_line": 1, "end_line": 5}]
    chk = [
        {"file_path": "a.js", "start_line": 3, "end_line": 3},
        {"file_path": "b.js", "start_line": 1, "end_line": 1},
    ]
    a = agreement(idx, chk)
    assert a["index_confirmed"] == 1 and a["checker_confirmed"] == 1 and a["jaccard"] == 0.5
