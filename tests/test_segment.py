import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from latters.anchors import ANCHORS, BY_NAME, COMPLETENESS_WEIGHTS, Role, load_anchors
from latters.extract import detect_misfonted_latin, read_document, convert_document
from latters.fonts.convert import Converter
from latters.segment import (BODY_WEIGHT, MIN_SEGMENT_CHARS, segment, tag_lines,
                             trust, verdict)
from latters.store import LetterRow, Store
from latters.validate import assess
from make_fixture import build

HDR = "कार्यालय जिला शिक्षा अधिकारी, रायपुर"
BODY = ("उपरोक्त विषय के संबंध में सूचित किया जाता है कि आवश्यक कार्यवाही "
        "सुनिश्चित करें। समस्त संबंधित अधिकारियों को निर्देशित किया जाता है "
        "कि नियत तिथि तक प्रतिवेदन प्रस्तुत करें।")


def letter(n, *, closing=True, header=True, subject=True):
    p = ([HDR] if header else []) + [
        f"पत्र संख्या शिक्षा/2024/{n}", f"दिनांक {n}.03.2024", "सेवा में,",
        "समस्त प्राचार्य, शासकीय उच्चतर माध्यमिक विद्यालय"]
    if subject:
        p.append(f"विषय: विषय क्रमांक {n} की सूचना।")
    p += ["महोदय,", BODY]
    if closing:
        p += ["भवदीय", "जिला शिक्षा अधिकारी"]
    return "\n".join(p)


# --- anchors --------------------------------------------------------------
def test_completeness_weights_total_one():
    assert abs(sum(COMPLETENESS_WEIGHTS.values()) + BODY_WEIGHT - 1.0) < 1e-9


@pytest.mark.parametrize("line,expected", [
    ("कार्यालय जिला शिक्षा अधिकारी, रायपुर", "header"),
    ("पत्र संख्या शिक्षा/2024/118", "letter_number"),
    ("ज्ञापांक-203", "letter_number"),
    ("फा.सं. 12/2019-स्था", "letter_number"),
    ("दिनांक 15.03.2024", "date"),
    ("सेवा में,", "addressee"),
    ("विषय: मासिक समीक्षा बैठक।", "subject"),
    ("प्रसंग:- इस कार्यालय के पत्रांक-७५", "reference"),
    ("महाशय,", "salutation"),
    ("महोदय,", "salutation"),
    ("विश्वासभाजन", "closing"),
    ("भवदीय", "closing"),
    ("हस्ताक्षर", "closing"),
    ("अनु०यथोक्त।", "closing"),
    ("प्रतिलिपि: सूचनार्थ प्रेषित।", "distribution"),
    ("- 2 -", "noise"),
    ("क्रमशः", "noise"),
])
def test_anchor_fires(line, expected):
    """Every form here was observed in a real district archive."""
    assert expected in tag_lines(line)[0].anchors


def test_reference_suppresses_letter_number():
    """A reference to another letter's number must not open a new segment."""
    tags = tag_lines("प्रसंग: आपके पत्र क्रमांक 44 दिनांक 01.03.2024")[0]
    assert "reference" in tags.anchors
    assert "letter_number" not in tags.anchors


def test_noise_lines_carry_no_other_anchor():
    assert tag_lines("- 2 -")[0].anchors == frozenset({"noise"})


def test_anchors_are_extensible_without_editing_code():
    custom = load_anchors({"closing": [r"इति\s+शुभम्"]})
    by = {a.name: a for a in custom}
    assert by["closing"].pattern.search("इति शुभम्")
    assert by["closing"].pattern.search("भवदीय")      # base forms survive


def test_unknown_anchor_name_is_rejected():
    with pytest.raises(ValueError, match="unknown anchor"):
        load_anchors({"not_an_anchor": ["x"]})


def test_every_anchor_has_a_role():
    assert all(isinstance(a.role, Role) for a in ANCHORS)
    assert {a.name for a in ANCHORS if a.role is Role.END} >= {"closing", "distribution"}


# --- segmentation ---------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_clean_letters_split_exactly(n):
    segs = segment("\n".join(letter(i) for i in range(1, n + 1)))
    assert len(segs) == n
    assert all(s.completeness() == 1.0 for s in segs)


def test_primary_rule_not_the_fallback_does_the_work():
    """Regression: when the closing anchors did not match the office's
    sign-off, 45% of boundaries came from the repeated-subject fallback."""
    segs = segment("\n".join(letter(i) for i in range(1, 6)))
    causes = [s.opened_by for s in segs]
    assert causes.count("repeated-subject") == 0
    assert causes.count("header-after-close") == 4


def test_truncated_letter_is_recovered_with_its_own_letterhead():
    segs = segment(letter(1) + "\n" + letter(2, closing=False) + "\n" + letter(3))
    assert len(segs) == 3
    assert segs[2].opened_by == "repeated-subject"
    # Regression: the back-up loop used to stop at the addressee, leaving the
    # recovered letter without its own header and number.
    assert segs[2].completeness() == 1.0
    assert segs[2].text.startswith(HDR)
    # The genuinely damaged one is the one flagged.
    assert "closing" in segs[1].missing()


def test_page_noise_does_not_split_a_letter():
    assert len(segment(letter(1).replace("महोदय,", "- 2 -\nमहोदय,"))) == 1


def test_letterhead_fragment_merges_forward():
    segs = segment("कार्यालय\n\n" + letter(1))
    assert len(segs) == 1 and segs[0].completeness() == 1.0


@pytest.mark.parametrize("text", ["", "- 2 -\n\nक्रमशः\n..."])
def test_empty_and_noise_only_yield_nothing(text):
    assert segment(text) == []


def test_runt_below_minimum_is_not_emitted_alone():
    segs = segment("कार्यालय एक\n" + letter(1))
    assert all(s.body_chars >= MIN_SEGMENT_CHARS for s in segs)


def test_completeness_reports_what_is_missing():
    seg = segment(letter(1, closing=False, subject=False))[0]
    assert set(seg.missing()) == {"closing", "subject"}
    assert seg.completeness() == pytest.approx(1.0 - 0.15 - 0.20)


# --- trust ----------------------------------------------------------------
def test_trust_orders_by_source_tier():
    scores = [trust(0.9, 0.9, t) for t in ("unicode", "docx", "pdf", "ocr")]
    assert scores == sorted(scores, reverse=True)


def test_trust_gates_on_each_signal_independently():
    """A letter can be clean text but incomplete, or complete but garbled."""
    assert verdict(trust(0.99, 1.0, "unicode")) == "index"
    assert verdict(trust(0.99, 0.2, "docx")) != "index"      # complete text, half a letter
    assert verdict(trust(0.30, 1.0, "docx")) != "index"      # whole letter, garbled
    assert verdict(trust(0.30, 0.2, "ocr")) == "quarantine"


def test_unknown_tier_is_treated_as_worst_case():
    assert trust(1.0, 1.0, "something-else") == trust(1.0, 1.0, "ocr")


# --- store ----------------------------------------------------------------
def test_store_tokenizer_keeps_matras():
    """Regression: FTS5's default unicode61 drops every combining mark, so
    समीक्षा indexes as ["सम","ष"] and a search for की matches कार्यवाही."""
    with Store() as s:
        s.add([LetterRow("a", 1, "समीक्षा बैठक की सूचना", trust=1.0),
               LetterRow("b", 1, "अनुशासनिक कार्यवाही हेतु", trust=1.0)])
        assert len(s.search("की")) == 1
        assert len(s.search("समीक्षा")) == 1
        assert len(s.search("कार्यवाही")) == 1
        # The distinct matras must survive as distinct tokens.
        s.add([LetterRow("c", 1, "कि की कु कू के कै को कौ " + "x" * 5, trust=1.0)])
        assert len(s.search("कु")) == 1


def test_store_dedupes_identical_letters():
    row = LetterRow("a.docx", 1, "वही पाठ बार बार", trust=1.0)
    with Store() as s:
        assert s.add([row]) == (1, 0)
        assert s.add([row]) == (0, 1)
        assert s.count() == 1


def test_store_search_filters_by_trust():
    with Store() as s:
        s.add([LetterRow("a", 1, "समीक्षा बैठक आयोजित", trust=0.9, verdict="index"),
               LetterRow("b", 1, "समीक्षा बैठक स्थगित", trust=0.3, verdict="quarantine")])
        assert len(s.search("समीक्षा", min_trust=0.6)) == 1
        assert len(s.search("समीक्षा", min_trust=0.0)) == 2


def test_store_delete_updates_the_index():
    with Store() as s:
        s.add([LetterRow("a", 1, "समीक्षा बैठक आयोजित", trust=1.0)])
        rid = s.search("समीक्षा")[0]["id"]
        assert s.delete(rid)
        assert s.search("समीक्षा") == []


def test_store_update_updates_the_index():
    with Store() as s:
        s.add([LetterRow("a", 1, "समीक्षा बैठक", trust=1.0)])
        s.db.execute("UPDATE letters SET text='स्थानांतरण आदेश' WHERE id=1")
        s.db.commit()
        assert s.search("समीक्षा") == [] and len(s.search("स्थानांतरण")) == 1


def test_store_quarantined_and_stats():
    with Store() as s:
        s.add([LetterRow("a", 1, "ठीक पाठ है", trust=0.95, verdict="index"),
               LetterRow("b", 1, "खराब पाठ है", trust=0.30, verdict="quarantine")])
        assert [r["verdict"] for r in s.quarantined()] == ["quarantine"]
        assert s.stats()["by_verdict"] == {"index": 1, "quarantine": 1}


def test_store_persists_to_disk(tmp_path):
    db = tmp_path / "c.db"
    with Store(db) as s:
        s.add([LetterRow("a", 1, "समीक्षा बैठक", trust=1.0)])
    with Store(db) as s:
        assert s.count() == 1 and len(s.search("समीक्षा")) == 1


# --- regressions from the real archive ------------------------------------
def test_runs_split_mid_word_are_coalesced(tmp_path):
    """Regression: Word splits paragraphs at revision boundaries with no
    regard for words. A real file had the pre-base i-matra alone in its own
    run ('f' then 'tyk'), so per-run conversion produced िजला, not जिला."""
    path = build(tmp_path / "split.docx", [[("f", "Kruti Dev 010"),
                                            ("tyk", "Kruti Dev 010")]])
    text, _ = convert_document(read_document(path))
    assert text == "जिला"


def test_reph_survives_a_run_boundary(tmp_path):
    path = build(tmp_path / "reph.docx", [[("dk;kZ", "Kruti Dev 010"),
                                           ("y;", "Kruti Dev 010")]])
    text, _ = convert_document(read_document(path))
    assert text == "कार्यालय"


def test_rescue_latin_ignores_legacy_capital_runs():
    """Regression: `mi;qZDRk` (उपर्युक्त) contains the all-caps run ZDR, which
    the heuristic rescued into the output as Latin."""
    assert detect_misfonted_latin("mi;qZDRk") == []
    assert detect_misfonted_latin("ZDR") == []
    assert detect_misfonted_latin("IAS") == []
    assert [s for _, _, s in detect_misfonted_latin("DEO/RPR/2024")] == ["DEO/RPR/2024"]


@pytest.mark.parametrize("legacy,expected", [
    ("Ik=kad", "पत्रांक"),        # half-form + k is the FULL consonant
    ("Lkkj.k", "सारण"),
    ("Jh", "श्री"),
    ("egk’k;", "महाशय"),          # Word smart-quoted the sha slot
    ("Hkw”k.k", "भूषण"),
    ("vkWuykbZu", "ऑनलाईन"),      # kW is one unit; vkW is a precomposed vowel
    ("MkW0", "डॉ०"),
    ("fo:)", "विरूद्ध"),
    ("o`f)", "वृद्धि"),
    ("foÙk", "वित्त"),
    ("lekgŸkkZ", "समाहर्ता"),
    ("jíhdj.k", "रद्दीकरण"),
    ("i=kad&75@fnukad", "पत्रांक-७५/दिनांक"),
])
def test_slots_recovered_from_the_real_archive(legacy, expected):
    assert Converter().convert(legacy).text == expected


def test_genuine_virama_final_words_are_not_flagged():
    """एतद्, तहत्, विधिवत् are correct Hindi, not corruption."""
    assert assess("एतद् अभिलेख के तहत् विधिवत् सुनवाई की गई").violations == {}
    assert "virama_word_final" in assess("कार्य् के भवन् में").violations


def test_word_initial_matra_is_detected_on_every_line():
    """Regression: the rule compiled without re.MULTILINE, so `^` anchored to
    the whole document and the check effectively never fired."""
    q = assess("ठीक पंक्ति है\nिजला बैंकिंग कोषांग\nदूसरी ठीक पंक्ति")
    assert q.violations.get("matra_word_initial") == 1


# --- document form, found on a real archive -------------------------------
from latters.segment import (FORM_WEIGHTS, Form, detect_form,  # noqa: E402
                             is_endorsement)

ORDER = """कार्यालय, प्रखण्ड विकास पदाधिकारी, नगरा, सारण।
आदेश
ज्ञापांक-1254/स्था०, दिनांक 25.10.2022
श्री हृदया प्रसाद, तत्कालीन जनसेवक, प्रखंड एकमा के विरुद्ध संचालित अनुशासनिक
कार्यवाही के संचालन हेतु संचालन पदाधिकारी नियुक्त किया जाता है। आरोपित कर्मी
के विरुद्ध आरोप पत्र निर्गत किया जा चुका है और साक्ष्य संकलित किए जा रहे हैं।
प्रतिलिपि:- उप विकास आयुक्त, सारण को सूचनार्थ प्रेषित।"""


def test_an_order_is_not_a_broken_letter():
    """Measured on a real archive: 211 of 589 segments have no addressee and
    169 of those scored below 0.7 -- 29% of the corpus marked defective for
    being the wrong genre. An आदेश is addressed to nobody, has no विषय line
    and ends with a distribution list rather than भवदीय."""
    seg = segment(ORDER)[0]
    assert seg.form is Form.ORDER
    assert seg.completeness() >= 0.8
    assert "addressee" not in seg.missing()
    assert "closing" not in seg.missing()


def test_a_letter_is_still_scored_as_a_letter():
    seg = segment(letter(1))[0]
    assert seg.form is Form.LETTER
    assert seg.completeness() == 1.0


def test_an_addressee_always_means_letter():
    """Even with an आदेश heading: if it says सेवा में, it was sent to someone."""
    assert detect_form("आदेश\nसेवा में,\nअंचल अधिकारी,\nकुछ पाठ") is Form.LETTER


@pytest.mark.parametrize("marker", [
    "आदेश", "कार्यालय आदेश", "ज्ञापन", "कार्यालय ज्ञापन",
    "अधिसूचना", "परिपत्र", "संकल्प",
])
def test_order_markers(marker):
    assert detect_form(f"कार्यालय जिला राजस्व शाखा\n{marker}\n"
                       "ज्ञापांक-1/2024, दिनांक 01.01.2024\n" + "क" * 400) is Form.ORDER


def test_a_marker_deep_in_the_body_is_not_a_heading():
    body = "\n".join(["कुछ पाठ"] * 30) + "\nआदेश\n"
    assert detect_form(body) is not Form.ORDER


def test_a_fragment_is_not_laundered_into_an_order():
    """The risk of form-awareness: calling every truncated segment an order
    to make its score go up. A document with no addressee is only an order
    if it says so, or behaves like one -- number, date and a real body."""
    seg = segment("कुछ अधूरा पाठ जो न पत्र है न आदेश और जिसमें कोई संख्या नहीं है "
                  "और यह काफी लंबा है ताकि खंड बने।")[0]
    assert seg.form is Form.FRAGMENT
    assert seg.completeness() < 0.7


def test_a_numbered_dated_document_with_a_body_counts_as_an_order():
    text = ("कार्यालय जिला राजस्व शाखा\nज्ञापांक-116, दिनांक 18.03.2024\n"
            + "उक्त के आलोक में आवश्यक कार्यवाही सुनिश्चित की जाय। " * 12)
    assert detect_form(text, {"letter_number": 1, "date": 1}) is Form.ORDER


def test_every_form_weight_set_totals_one():
    from latters.segment import BODY_WEIGHT
    for form, weights in FORM_WEIGHTS.items():
        assert abs(sum(weights.values()) + BODY_WEIGHT - 1.0) < 1e-9, form


def test_store_round_trips_the_form():
    with Store() as s:
        s.add([LetterRow("a.docx", 1, ORDER, trust=0.9, form="order")])
        assert s.get(1)["form"] == "order"
        assert s.stats()["by_form"] == {"order": 1}


# --- endorsements, found on the operator's archive ------------------------
ENDORSEMENT = """ज्ञापांक---------------/रा०, दिनांक----------------
प्रतिलिपि:- समाहर्त्ता, सारण छपरा को सादर सूचनार्थ समर्पित।
प्रतिलिपि:- उप विकास आयुक्त, सारण को सूचनार्थ प्रेषित।"""


def test_an_endorsement_merges_into_the_letter_above_it():
    """A पृष्ठांकन carries its OWN ज्ञापांक and दिनांक, which is exactly what
    the letter-number-after-closing rule fires on, so 98 of 112 "fragments"
    in the sample archive were copy-forwarding tails split off as letters of
    their own. They are part of the same dispatch."""
    segs = segment(letter(1) + "\n" + ENDORSEMENT)
    assert len(segs) == 1
    assert "प्रतिलिपि" in segs[0].text
    assert segs[0].form is Form.LETTER


def test_two_letters_each_with_an_endorsement_stay_two():
    segs = segment("\n".join([letter(1), ENDORSEMENT, letter(2), ENDORSEMENT]))
    assert len(segs) == 2
    assert all("प्रतिलिपि" in s.text for s in segs)


def test_an_endorsement_needs_a_distribution_line():
    """Number plus date alone is an order's signature, not an endorsement's."""
    assert not is_endorsement("ज्ञापांक-12, दिनांक 01.01.2024\nकुछ पाठ",
                              {"letter_number": 1, "date": 1})


def test_a_real_letter_is_never_an_endorsement():
    """It has a subject and a salutation of its own."""
    seg = segment(letter(1))[0]
    assert not is_endorsement(seg.text, seg.anchors)


def test_a_long_block_with_prat_ilipi_is_not_an_endorsement():
    """An order ends with a distribution list too. Length separates them."""
    long_order = ORDER + "\n" + ("उक्त के आलोक में आवश्यक कार्यवाही की जाय। " * 30)
    assert not is_endorsement(long_order, {"distribution": 1})


def test_a_leading_endorsement_is_kept_not_dropped():
    """A file that opens mid-dispatch has an endorsement with no parent.
    Keeping an odd segment beats losing text."""
    segs = segment(ENDORSEMENT + "\n" + letter(1))
    assert segs
    assert any("प्रतिलिपि" in s.text for s in segs)


def test_an_explicit_order_heading_beats_the_endorsement_heuristic():
    """A short आदेश ending in a प्रतिलिपि line looks exactly like an
    endorsement to the length-and-distribution test. An endorsement never
    carries an आदेश heading of its own, so the heading wins."""
    assert detect_form(ORDER, {"distribution": 1, "letter_number": 1,
                               "date": 1}) is Form.ORDER
