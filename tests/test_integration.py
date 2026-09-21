"""End-to-end tests: the CLI, in sequence, as an office would run it.

The unit tests exercise functions. These exercise the *application* -- real
argument parsing, real exit codes, real files on disk, each phase consuming
the previous phase's output. Every bug this file guards against was found by
running the pipeline from a clean directory rather than by reading the code.

Exit-code convention, asserted throughout:
    0  worked
    1  worked, but the result needs a human (quarantine, needs_review)
    2  the invocation was wrong (bad path, missing corpus)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from latters.cli import main
from make_fixture import SAMPLE_LETTER, build

LEGACY_BODY = ("mijksDr fo\"k; ds lanHkZ esa lwfpr fd;k tkrk gS fd vko';d "
               "dk;Zokgh lqfuf'pr djrs gq, izfrosnu bl dk;kZy; dks miyC/k djk;saA")


def _archive(root: Path, n: int = 14) -> Path:
    """A synthetic legacy archive: two offices, several letters per file."""
    root.mkdir(parents=True, exist_ok=True)
    for office, branch, folder in (("ftyk jktLo 'kk[kk", "jk0", "revenue"),
                                   ("ftyk LFkkiuk 'kk[kk", "LFkk0", "estab")):
        letters = []
        for i in range(n):
            letters += [
                [(f"dk;kZy; {office}", "Kruti Dev 010")],
                [("i=kad- ------------/" + branch + ",", "Kruti Dev 010")],
                [("Nijk] fnukad------------", "Kruti Dev 010")],
                [("lsok esa]", "Kruti Dev 010")],
                [("vapy vf/kdkjh]", "Kruti Dev 010")],
                [(f"fo\"k;%& tk¡p izdj.k {i} ds laca/k esaA", "Kruti Dev 010")],
                [("egk'k;]", "Kruti Dev 010")],
                [(f"tk¡p izdj.k {i} - {LEGACY_BODY}", "Kruti Dev 010")],
                [("fo'oklHkktu", "Kruti Dev 010")],
            ]
        build(root / f"{folder}.docx", letters)
    return root


@pytest.fixture(scope="module")
def archive(tmp_path_factory) -> Path:
    return _archive(tmp_path_factory.mktemp("archive"))


@pytest.fixture(scope="module")
def workdir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("work")


# --- phases in order, each consuming the last -----------------------------
def test_phase1_inventory(archive, capsys):
    assert main(["inventory", str(archive)]) == 0
    out = capsys.readouterr().out
    assert "Kruti Dev 010" in out and "krutidev010" in out
    assert "legacy" in out


def test_phase1_ingest_produces_readable_hindi(archive, workdir, capsys):
    out_dir = workdir / "txt"
    assert main(["ingest", str(archive), "-o", str(out_dir), "--repair"]) == 0
    texts = [p.read_text(encoding="utf-8") for p in out_dir.glob("*.txt")]
    assert texts
    joined = "\n".join(texts)
    for expected in ("कार्यालय", "पत्रांक", "सेवा में", "विषय", "महाशय",
                     "विश्वासभाजन", "जाँच"):
        assert expected in joined, expected


def test_phase1_gold_regression_passes(capsys):
    assert main(["fonts", "gold"]) == 0
    assert "100.0000%" in capsys.readouterr().out


def test_phase1_gold_extract_builds_a_review_sheet(archive, workdir):
    out = workdir / "review"
    assert main(["gold", "extract", str(archive), "-o", str(out), "-n", "8"]) == 0
    assert (out / "review.docx").exists() and (out / "review.tsv").exists()


def test_phase2_audit_reports_anchor_health(archive, capsys):
    assert main(["audit", str(archive)]) == 0
    out = capsys.readouterr().out
    assert "anchor firing" in out and "boundary cause" in out


def test_phase2_segment_splits_and_stores(archive, workdir, capsys):
    db = workdir / "corpus.db"
    assert main(["segment", str(archive), "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert db.exists()
    # Two files of 14 letters each; segmentation must find most of them.
    n = int(out.split("letters from")[0].strip().split()[-1])
    assert n >= 20, out


def test_phase2_search_finds_stored_letters(workdir, capsys):
    assert main(["search", "जाँच", "--db", str(workdir / "corpus.db")]) == 0
    assert "trust" in capsys.readouterr().out


def test_phase3_fields_reports_coverage(workdir, capsys):
    assert main(["fields", "--db", str(workdir / "corpus.db")]) == 0
    out = capsys.readouterr().out
    assert "letter_number" in out and "blank" in out


def test_phase3_classify_writes_labels(workdir, capsys):
    assert main(["classify", "--db", str(workdir / "corpus.db"), "--write"]) == 0
    out = capsys.readouterr().out
    assert "majority baseline" in out, "the baseline must always be reported"


def test_phase3_templates_mine_skeletons(workdir, capsys):
    sk = workdir / "skeletons"
    assert main(["templates", "--db", str(workdir / "corpus.db"),
                 "-o", str(sk)]) == 0
    assert list(sk.glob("*.md")), capsys.readouterr().out


def test_phase4_retrieve_returns_grounded_hits(workdir, capsys):
    assert main(["retrieve", "जाँच प्रकरण के संबंध में",
                 "--db", str(workdir / "corpus.db")]) == 0
    out = capsys.readouterr().out
    assert "trust" in out and "letter" in out


def test_phase4_eval_reports_the_noise_floor(workdir, capsys):
    rc = main(["eval", "--db", str(workdir / "corpus.db"), "--min-cell", "3"])
    out, err = capsys.readouterr()
    assert rc == 0, err
    assert "random" in out and "standard error" in out


def test_segment_stores_the_subject(workdir):
    """Regression: it did not, so the FTS subject column and the retrieval
    evaluation's query set were both silently inert in a fresh pipeline."""
    from latters.store import Store
    with Store(str(workdir / "corpus.db")) as s:
        n = s.db.execute(
            "SELECT COUNT(*) FROM letters WHERE subject IS NOT NULL"
        ).fetchone()[0]
    assert n > 0


def test_eval_with_nothing_to_evaluate_says_so(tmp_path, capsys):
    """Regression: it printed empty tables and returned success, which reads
    as an evaluation that found nothing rather than one that never ran."""
    from make_fixture import SAMPLE_LETTER
    root = tmp_path / "one"
    build(root / "o.docx", SAMPLE_LETTER)
    db = tmp_path / "one.db"
    main(["segment", str(root), "--db", str(db)])
    capsys.readouterr()
    assert main(["eval", "--db", str(db)]) == 2
    err = capsys.readouterr().err
    assert "nothing to evaluate" in err and "with a subject line" in err


def test_phase5_draft_end_to_end(workdir, capsys):
    rc = main(["draft", "जाँच प्रकरण का प्रतिवेदन मांगना है",
               "--db", str(workdir / "corpus.db"), "--stub",
               "--skeletons", str(workdir / "skeletons")])
    out = capsys.readouterr().out
    assert rc in (0, 1)
    assert "विश्वासभाजन" in out
    assert "drafted from" in out
    assert "must be read and corrected before dispatch" in out


def test_phase5_draft_uses_the_mined_letterhead(workdir, capsys):
    main(["draft", "जाँच प्रकरण का प्रतिवेदन मांगना है",
          "--db", str(workdir / "corpus.db"), "--stub",
          "--skeletons", str(workdir / "skeletons"),
          "--subject", "जाँच के संबंध में"])
    out = capsys.readouterr().out
    letter = out.split("=" * 72)[1]
    assert "पत्रांक" in letter and "विषय:-" in letter
    # Order matters: the number must precede the salutation.
    assert letter.index("पत्रांक") < letter.index("विश्वासभाजन")


# --- the invocation mistakes an office will actually make -----------------
@pytest.mark.parametrize("argv", [
    ["inventory", "no-such-dir"],
    ["ingest", "no-such-dir"],
    ["audit", "no-such-dir"],
    ["segment", "no-such-dir", "--db", "x.db"],
    ["gold", "extract", "no-such-dir", "-o", "r"],
])
def test_a_mistyped_archive_path_is_an_error(argv, capsys):
    """Reporting '0 files' for a typo is the same output as a genuinely
    empty folder, so the mistake looks like a finding."""
    assert main(argv) == 2
    assert "no such path" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [
    ["stats"], ["search", "x"], ["retrieve", "x"], ["eval"],
    ["classify"], ["templates"], ["fields"], ["draft", "x", "--stub"],
])
def test_a_mistyped_db_path_is_an_error_and_creates_nothing(argv, tmp_path, capsys):
    """sqlite3 creates a database on connect, so a mistyped --db silently
    produced an EMPTY corpus that every later command worked on happily."""
    missing = tmp_path / "nested" / "nope.db"
    assert main([*argv, "--db", str(missing)]) == 2
    assert "no corpus database" in capsys.readouterr().err
    assert not missing.exists()
    assert not missing.parent.exists()


def test_a_mistyped_skeleton_directory_is_an_error(workdir, capsys):
    """It used to be ignored, so the clerk's corrections appeared to have had
    no effect on the output."""
    assert main(["draft", "कुछ", "--db", str(workdir / "corpus.db"), "--stub",
                 "--skeletons", "/no/such/dir"]) == 2
    assert "no skeleton directory" in capsys.readouterr().err


def test_corrupt_documents_are_skipped_with_a_reason(tmp_path, capsys):
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "corrupt.docx").write_bytes(b"PK\x03\x04 not a docx")
    (bad / "empty.docx").write_bytes(b"")
    assert main(["ingest", str(bad)]) == 0
    out = capsys.readouterr().out
    assert "skipped" in out
    assert "corrupt.docx:" in out, "the reason must be shown, not just a count"


def test_a_unicode_only_archive_needs_no_conversion(tmp_path, capsys):
    root = tmp_path / "uni"
    build(root / "u.docx", [[("कार्यालय आदेश संख्या 44/2022", "Mangal")]])
    assert main(["inventory", str(root)]) == 0
    assert "unicode-or-latin" in capsys.readouterr().out


def test_gold_extract_says_so_when_there_is_nothing_legacy(tmp_path, capsys):
    root = tmp_path / "uni2"
    build(root / "u.docx", [[("कार्यालय आदेश संख्या 44/2022", "Mangal")]])
    assert main(["gold", "extract", str(root), "-o", str(tmp_path / "r")]) == 1
    assert "no legacy-font lines" in capsys.readouterr().err


def test_draft_from_a_thin_corpus_demands_review(tmp_path, capsys):
    """A draft from three letters looks exactly as confident as a good one."""
    root = tmp_path / "tiny"
    build(root / "t.docx", SAMPLE_LETTER)
    db = tmp_path / "tiny.db"
    main(["segment", str(root), "--db", str(db)])
    capsys.readouterr()
    assert main(["draft", "कुछ चाहिए", "--db", str(db), "--stub"]) == 1
    out = capsys.readouterr().out
    assert "NEEDS REVIEW" in out


# --- platform reality, found by running rather than reading ---------------
def test_devanagari_survives_a_windows_console_encoding(capsysbinary, monkeypatch):
    """The target machine is a Windows laptop. Python encodes stdout with the
    console code page there -- usually cp1252 -- so the first Devanagari
    character of a draft raised UnicodeEncodeError while everything worked
    perfectly in development."""
    import io
    from latters.cli import _prepare_streams

    raw = io.BytesIO()
    narrow = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", narrow)
    _prepare_streams()
    sys.stdout.write("पत्रांक- 887/रा०")      # must not raise
    sys.stdout.flush()
    assert raw.getvalue()


def test_broken_pipe_does_not_traceback(monkeypatch, capsys):
    """`latters templates | head` raised BrokenPipeError with a traceback
    when head exited. Office staff pipe to head and more constantly."""
    import latters.cli as cli

    def _boom(_args):
        raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr(cli, "build_parser", lambda: _Parser(_boom))
    assert cli.main([]) == 0


class _Parser:
    def __init__(self, fn):
        self.fn = fn

    def parse_args(self, argv):
        import argparse
        ns = argparse.Namespace()
        ns.func = self.fn
        return ns


def test_keyboard_interrupt_is_not_a_traceback(monkeypatch, capsys):
    import latters.cli as cli

    def _boom(_args):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "build_parser", lambda: _Parser(_boom))
    assert cli.main([]) == 130
    assert "interrupted" in capsys.readouterr().err


def test_every_third_party_import_is_declared():
    """Regression: numpy was imported by retrieve.py and declared nowhere.

    The "no corpus database" guard short-circuits before that import, so
    every quick check passed while `latters retrieve` against a real
    database failed with ModuleNotFoundError on a fresh machine.
    """
    import ast
    import sys as _sys

    root = Path(__file__).resolve().parent.parent
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    declared = set()
    for block in ("dependencies", "dense", "dev", "web"):
        m = __import__("re").search(rf"^{block} = \[(.*?)\]", pyproject,
                                    __import__("re").M | __import__("re").S)
        if m:
            for part in m.group(1).split(","):
                name = part.strip().strip('"\'').split(">")[0].split("=")[0]
                if name:
                    declared.add(name.replace("-", "_").lower())

    stdlib = set(getattr(_sys, "stdlib_module_names", ()))
    local = {p.stem for p in (root / "src" / "latters").rglob("*.py")} | {"latters"}
    undeclared = set()
    for path in (root / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module.split(".")[0]] if node.module and node.level == 0 else []
            else:
                continue
            for mod in mods:
                key = mod.lower()
                if key in stdlib or key in local or key in declared:
                    continue
                undeclared.add(mod)
    assert not undeclared, (
        f"imported but not declared in pyproject.toml: {sorted(undeclared)}")
