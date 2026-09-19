import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from latters.extract import (classify_font, convert_document, detect_misfonted_latin,
                             read_document, UnsupportedFormat)
from latters.fonts.convert import Converter, normalize_devanagari
from latters.fonts.tables import TableError, available_tables, load_table
from latters.gold import discover, levenshtein, run as run_gold
from latters.repair import repair, visarga_to_colon
from latters.validate import assess, build_vocabulary
from make_fixture import SAMPLE_LETTER, build


# --- tables ---------------------------------------------------------------
def test_tables_load():
    assert {"krutidev010", "devlys010"} <= set(available_tables())
    t = load_table("krutidev010")
    assert t.mapping["d"] == "क"
    # `#` is a legacy slot, not a comment, when followed by a tab.
    assert t.mapping["#"] == "रु"


def test_devlys_inherits_and_overrides():
    kd, dl = load_table("krutidev010"), load_table("devlys010")
    assert dl.mapping["d"] == kd.mapping["d"]          # inherited
    assert dl.mapping["Ï"] != kd.mapping["Ï"]          # overridden


def test_missing_table_raises():
    with pytest.raises(TableError):
        load_table("no_such_font")


# --- the three conversion passes -----------------------------------------
@pytest.mark.parametrize("legacy,expected", [
    ("fgUnh", "हिन्दी"),          # chhoti-i before a simple consonant
    ("izfr", "प्रति"),            # chhoti-i after a subjoined-ra cluster
    ("deZ", "कर्म"),              # reph on the final consonant
    ("dk;kZy;", "कार्यालय"),      # reph with a following aa-matra
    ("dhfrZ", "कीर्ति"),          # both reorderings in one syllable
    ("Kkiu", "ज्ञापन"),           # maximal munch must not swallow the aa-matra
    ("i=", "पत्र"),               # `=` is the tra slot
    ("¼d½", "(क)"),               # paren slots
])
def test_conversion_pairs(legacy, expected):
    assert Converter().convert(legacy).text == expected


def test_latin_digits_flag():
    assert Converter().convert("2024").text == "२०२४"
    assert Converter(latin_digits=True).convert("2024").text == "2024"


def test_unmapped_is_reported():
    conv = Converter().convert("d★")
    assert "★" in conv.unmapped


def test_no_private_use_leaks():
    """A reph with no cluster in front of it must not emit a PUA sentinel."""
    out = Converter().convert("Z").text
    assert not re.search(r"[-]", out)
    assert out == "र्"


def test_normalize_is_idempotent():
    s = normalize_devanagari("कार्यालय‌  ज्ञापन\r\n")
    assert normalize_devanagari(s) == s


def test_nukta_forms_normalise_together():
    """Composed U+095C and decomposed ड + nukta must compare equal after
    normalisation, or retrieval silently misses half the matches."""
    assert normalize_devanagari("ड़") == normalize_devanagari("ड़")


# --- gold set -------------------------------------------------------------
def test_levenshtein():
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("abc", "abd") == 1
    assert levenshtein("", "abc") == 3


def test_gold_set_passes():
    report = run_gold(discover())
    assert report.n >= 40, "seed gold set shrank unexpectedly"
    assert report.char_accuracy >= 0.98, report.render()


# --- validator ------------------------------------------------------------
def test_clean_text_scores_high():
    q = assess("कार्यालय ज्ञापन संख्या बारह दिनांक पंद्रह मार्च विषय समीक्षा बैठक की सूचना")
    assert q.verdict == "clean" and q.score > 0.9


def test_unconverted_latin_is_quarantined():
    """Regression: with no Devanagari at all, sequence legality is vacuously
    perfect, and an earlier version scored untouched gibberish as 'review'."""
    q = assess("dk;kZy; Kkiu la[;k")
    assert q.verdict == "quarantine" and q.score == 0.0


def test_pdf_visual_order_is_detected():
    clean = "कार्यालय ज्ञापन विषय समीक्षा बैठक की सूचना प्रतिलिपि सूचनार्थ प्रेषित"
    scrambled = re.sub(r"(.)(ि)", r"\2\1", clean)
    assert assess(scrambled).score < assess(clean).score
    assert "matra_word_initial" in assess(scrambled).violations


def test_pua_leak_is_a_violation():
    assert "pua_leak" in assess("कार्यालय").violations


def test_small_vocabulary_is_ignored():
    """A vocabulary from a handful of documents made clean letters look dirty."""
    text = "कार्यालय ज्ञापन संख्या दिनांक विषय समीक्षा बैठक सूचना प्रेषित महोदय"
    tiny = build_vocabulary([text] * 3)
    assert len(tiny) < 500
    assert assess(text, vocabulary=tiny).score == assess(text).score


def test_build_vocabulary_needs_repetition():
    vocab = build_vocabulary(["कार्यालय ज्ञापन", "कार्यालय आदेश", "कार्यालय पत्र"], min_count=3)
    assert "कार्यालय" in vocab and "ज्ञापन" not in vocab


# --- repairs --------------------------------------------------------------
def test_visarga_becomes_colon_after_a_label():
    assert visarga_to_colon("विषयः बैठक")[0] == "विषय: बैठक"


@pytest.mark.parametrize("word", ["अतः", "प्रायः", "स्वतः"])
def test_genuine_visarga_is_preserved(word):
    assert visarga_to_colon(f"{word} यह सूचित है")[0].startswith(word)


def test_repair_reports_counts():
    _, counts = repair("विषयः बैठक")
    assert counts["visarga_to_colon"] == 1


# --- extraction -----------------------------------------------------------
@pytest.mark.parametrize("font,table", [
    ("Kruti Dev 010", "krutidev010"),
    ("KrutiDev 011", "krutidev010"),
    ("DevLys 010", "devlys010"),
    ("Mangal", None),
    ("Times New Roman", None),
    (None, None),
])
def test_font_classification(font, table):
    assert classify_font(font)[0] == table


def test_docx_extraction_keeps_run_fonts(tmp_path):
    doc = read_document(build(tmp_path / "a.docx", SAMPLE_LETTER))
    assert doc.font_histogram["Kruti Dev 010"] > 0
    assert doc.font_histogram["Times New Roman"] > 0


def test_paragraph_style_font_is_resolved(tmp_path):
    """A run with no rFonts must inherit the paragraph style's font, or whole
    paragraphs silently pass through unconverted."""
    path = build(tmp_path / "b.docx", [], styled_paragraphs=["dk;kZy;"])
    text, used = convert_document(read_document(path))
    assert text == "कार्यालय" and used["krutidev010"] > 0


def test_latin_runs_are_not_mangled(tmp_path):
    """The whole point of run-level conversion: a Latin run inside a legacy
    document must survive untouched."""
    path = build(tmp_path / "c.docx", [[("dk;kZy; ", "Kruti Dev 010"),
                                        ("DEO/RPR/2024/1187", "Times New Roman")]])
    text, used = convert_document(read_document(path))
    assert "DEO/RPR/2024/1187" in text
    assert text.startswith("कार्यालय")
    assert used["(passthrough)"] == len("DEO/RPR/2024/1187")


def test_misfonted_latin_detection():
    assert detect_misfonted_latin("dk;kZy; DEO/RPR/2024")
    assert not detect_misfonted_latin("dk;kZy; Kkiu")     # ordinary Hindi
    assert not detect_misfonted_latin("12&2024")          # digits, no capitals


def test_rescue_latin_round_trips(tmp_path):
    path = build(tmp_path / "d.docx", [[("dk;kZy; DEO/RPR/2024", "Kruti Dev 010")]])
    plain, _ = convert_document(read_document(path))
    rescued, used = convert_document(read_document(path), rescue_latin=True)
    assert "DEO/RPR/2024" not in plain
    assert "DEO/RPR/2024" in rescued
    assert used["(rescued-latin)"] == len("DEO/RPR/2024")


def test_pdf_is_refused_with_the_reason(tmp_path):
    p = tmp_path / "x.pdf"
    p.write_bytes(b"%PDF-1.4")
    with pytest.raises(UnsupportedFormat, match="logical-stream"):
        read_document(p)


def test_binary_doc_is_refused_with_the_remedy(tmp_path):
    p = tmp_path / "x.doc"
    p.write_bytes(b"\xd0\xcf\x11\xe0" + b"\x00" * 16)
    with pytest.raises(UnsupportedFormat, match="soffice"):
        read_document(p)


def test_scanned_docx_is_flagged(tmp_path):
    doc = read_document(build(tmp_path / "e.docx", []))
    assert any("scanned" in w for w in doc.warnings)


# --- end to end -----------------------------------------------------------
def test_full_pipeline(tmp_path):
    path = build(tmp_path / "letter.docx", SAMPLE_LETTER)
    text, _ = convert_document(read_document(path), rescue_latin=True)
    text, _ = repair(text)
    assert "कार्यालय" in text and "विषय:" in text and "भवदीय" in text
    assert "(R. K. Sharma)" in text
    assert assess(text).verdict in ("clean", "review")


# --- gold-set builder -----------------------------------------------------
import csv as _csv

from latters import docx_writer as _D
from latters.goldbuild import (Candidate, collect_review, coverage, mark_blind,
                               mine_candidates, select_by_coverage, write_gold,
                               write_review_docx, write_review_tsv)


def _archive(tmp_path):
    root = tmp_path / "archive"
    build(root / "a.docx", SAMPLE_LETTER)
    build(root / "b.docx", [
        [("dk;kZy; vk;qDr] uxj fuxe", "DevLys 010")],
        [("fo\"k;% vkosnu vxzsf\"kr djus ckcr A", "DevLys 010")],
    ])
    build(root / "c.docx", [[("कार्यालय आदेश संख्या 44/2022", "Mangal")]])
    return root


def test_mine_skips_unicode_and_short_runs(tmp_path):
    cands, problems = mine_candidates(_archive(tmp_path))
    assert cands and not problems
    assert all(c.table in ("krutidev010", "devlys010") for c in cands)
    assert all(12 <= len(c.legacy) <= 140 for c in cands)
    # The Mangal paragraph is already Unicode and must not be offered.
    assert not any("कार्यालय आदेश" in c.legacy for c in cands)


def test_mine_deduplicates(tmp_path):
    root = tmp_path / "dup"
    line = [("dk;kZy; ftyk f'k{kk vf/kdkjh] jk;iqj", "Kruti Dev 010")]
    build(root / "x.docx", [line])
    build(root / "y.docx", [line])
    cands, _ = mine_candidates(root)
    assert len(cands) == 1


def test_selection_is_coverage_greedy(tmp_path):
    cands, _ = mine_candidates(_archive(tmp_path))
    chosen = select_by_coverage(cands, 3)
    assert len(chosen) <= 3
    # Greedy means the first pick contributes the most novel slots.
    assert len(chosen[0].novel) == max(len(c.novel) for c in chosen)
    # Selection must beat taking the first N in arbitrary order.
    greedy_keys = set().union(*(c.keys for c in chosen))
    naive_keys = set().union(*(c.keys for c in cands[:3]))
    assert len(greedy_keys) >= len(naive_keys)


def test_selection_is_deterministic(tmp_path):
    cands, _ = mine_candidates(_archive(tmp_path))
    a = [c.legacy for c in select_by_coverage(cands, 5, seed=7)]
    cands2, _ = mine_candidates(_archive(tmp_path))
    b = [c.legacy for c in select_by_coverage(cands2, 5, seed=7)]
    assert a == b


def test_mark_blind_fraction():
    cs = [Candidate(legacy=f"line{i}", table="krutidev010", font="f", source="s")
          for i in range(20)]
    assert mark_blind(cs, 0.25) == 5
    assert sum(c.blind for c in cs) == 5


def test_review_docx_keeps_the_legacy_font(tmp_path):
    """The left column is the ground truth, so it must carry the legacy font
    or the reviewer has nothing to read."""
    cands, _ = mine_candidates(_archive(tmp_path))
    chosen = select_by_coverage(cands, 4)
    path = write_review_docx(chosen, tmp_path / "review.docx")
    doc = read_document(path)
    assert doc.font_histogram["Kruti Dev 010"] > 0
    assert doc.font_histogram["Mangal"] > 0


def test_review_tsv_hides_suggestion_on_blind_rows(tmp_path):
    cands, _ = mine_candidates(_archive(tmp_path))
    chosen = select_by_coverage(cands, 6)
    for c in chosen[:2]:
        c.blind = True
    path = write_review_tsv(chosen, tmp_path / "review.tsv")
    rows = list(_csv.DictReader(path.read_text(encoding="utf-8-sig").splitlines(),
                                delimiter="\t"))
    blind = [r for r in rows if r["blind"]]
    assert len(blind) == 2
    assert all(r["suggested"] == "" for r in blind)
    assert all(r["suggested"] for r in rows if not r["blind"])


def _write_filled(tmp_path, rows, name="filled.tsv", encoding="utf-8-sig"):
    path = tmp_path / name
    with path.open("w", encoding=encoding, newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    return path


def _sheet(tmp_path, n=8):
    cands, _ = mine_candidates(_archive(tmp_path))
    chosen = select_by_coverage(cands, n)
    for c in chosen[:3]:
        c.blind = True
    p = write_review_tsv(chosen, tmp_path / "review.tsv")
    return list(_csv.DictReader(p.read_text(encoding="utf-8-sig").splitlines(),
                                delimiter="\t"))


def test_collect_counts_disagreement_not_typing(tmp_path):
    """Regression: a blind reviewer must type on every row, so counting typed
    rows made the blind disagreement rate 100% by construction and the
    rubber-stamping control useless."""
    rows = _sheet(tmp_path)
    conv = Converter()
    for r in rows:
        r["verdict_or_correction"] = conv.convert(r["legacy"]).text
    col = collect_review(_write_filled(tmp_path, rows))
    assert col.n_typed == len(rows)
    assert col.blind_disagreement == 0.0     # typed, but agreed
    assert col.sighted_disagreement == 0.0


def test_collect_counts_add_up(tmp_path):
    """Regression: blind rows were incremented twice, so sighted+blind
    exceeded the number of pairs."""
    rows = _sheet(tmp_path)
    for r in rows:
        r["verdict_or_correction"] = "ok" if not r["blind"] else "कुछ"
    col = collect_review(_write_filled(tmp_path, rows))
    assert col.n_sighted + col.n_blind == len(col.pairs)


def test_collect_detects_rubber_stamping(tmp_path):
    rows = _sheet(tmp_path)
    conv = Converter()
    for r in rows:
        if r["blind"]:
            r["verdict_or_correction"] = conv.convert(r["legacy"]).text + "क"
        else:
            r["verdict_or_correction"] = "ok"
    col = collect_review(_write_filled(tmp_path, rows))
    assert col.sighted_disagreement == 0.0
    assert col.blind_disagreement == 1.0


def test_collect_rejects_ok_on_a_blind_row(tmp_path):
    rows = _sheet(tmp_path)
    for r in rows:
        r["verdict_or_correction"] = "ok"
    col = collect_review(_write_filled(tmp_path, rows))
    assert any("blind row" in p for p in col.problems)
    assert col.n_blind == 0


def test_collect_excludes_unreviewed_rows(tmp_path):
    rows = _sheet(tmp_path)
    sighted = next(r for r in rows if not r["blind"])
    sighted["verdict_or_correction"] = "ok"
    col = collect_review(_write_filled(tmp_path, rows))
    assert col.n_blank == len(rows) - 1
    assert len(col.pairs) == 1


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16", "utf-8"])
def test_collect_survives_excel_encodings(tmp_path, encoding):
    """Excel writes TSV as UTF-16 or ANSI depending on how it was saved."""
    rows = _sheet(tmp_path, 4)
    for r in rows:
        r["verdict_or_correction"] = "ok" if not r["blind"] else "कार्यालय"
    col = collect_review(_write_filled(tmp_path, rows, f"f_{encoding}.tsv", encoding))
    assert col.pairs
    assert all("�" not in legacy for legacy, *_ in col.pairs)


def test_collect_reports_missing_columns(tmp_path):
    p = tmp_path / "bad.tsv"
    p.write_text("a\tb\n1\t2\n", encoding="utf-8")
    assert collect_review(p).problems


def test_collected_gold_file_is_readable_by_the_harness(tmp_path):
    rows = _sheet(tmp_path)
    conv = Converter()
    for r in rows:
        r["verdict_or_correction"] = conv.convert(r["legacy"]).text
    col = collect_review(_write_filled(tmp_path, rows))
    out = write_gold(col, tmp_path / "office_x.tsv", title="x")
    report = run_gold([out])
    assert report.n == len(col.pairs)
    assert report.char_accuracy == 1.0


def test_coverage_names_untested_slots():
    pairs = [("d", "क", "krutidev010", "")]
    cov = coverage(pairs, "krutidev010")
    assert "d" in cov.covered
    assert "Z" in cov.uncovered          # reph untested by this one pair
    assert 0 < cov.fraction < 1


def test_seed_gold_set_coverage_is_honestly_partial():
    """The seed set passes at 100% accuracy while exercising about half the
    table. Keeping this visible is the point of the coverage report."""
    from latters.gold import discover, load_pairs
    pairs = [p for path in discover() for p in load_pairs(path)]
    cov = coverage(pairs, "krutidev010")
    assert cov.fraction < 0.75, "if this rises, update the README claim"
    assert cov.uncovered


# --- docx writer ----------------------------------------------------------
def test_docx_writer_roundtrips_fonts(tmp_path):
    path = _D.write(tmp_path / "t.docx", [
        _D.heading("H"),
        _D.table([[[_D.para(_D.run("dk;", "Kruti Dev 010"))],
                   [_D.para(_D.run("कार्य", "Mangal"))]]], [4000, 4000]),
    ])
    doc = read_document(path)
    assert doc.font_histogram["Kruti Dev 010"] == 3
    assert doc.font_histogram["Mangal"] == 5


def test_docx_writer_escapes_xml(tmp_path):
    path = _D.write(tmp_path / "e.docx", [_D.para(_D.run('a<b>&"c"', "Calibri"))])
    assert 'a<b>&"c"' in read_document(path).blocks[0].raw_text
