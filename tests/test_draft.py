import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from latters.classify import TrainedClassifier
from latters.draft import (BLANK_SLOT, Budget, DraftService, assemble,
                           build_prompt, clean_body, estimate_tokens,
                           unsupported_facts, _exemplar_body)
from latters.llm import DEFAULT_OPTIONS, Ollama, StubLLM
from latters.retrieve import Letter, Retriever, TfidfIndex
from latters.store import LetterRow, Store
from latters.template import mine, parse_skeleton

BODY = ("उपर्युक्त विषय के प्रसंग में कहना है कि आवश्यक कार्यवाही सुनिश्चित "
        "करते हुए प्रतिवेदन इस कार्यालय को उपलब्ध कराएँ।")


_TOPICS = ["भूमि विवाद", "अतिक्रमण", "जमाबंदी सुधार", "लगान वसूली", "सीमांकन",
           "दाखिल खारिज", "नक्शा", "खाता विभाजन", "बंदोबस्ती", "अभिलेख सुधार",
           "मापी", "कब्जा", "वाद", "अपील", "निरीक्षण", "सत्यापन", "प्रगति",
           "अनुपालन", "शिकायत", "स्थल जाँच"]


def _letter(i, office="जिला राजस्व शाखा", branch="रा०", topic="जाँच प्रकरण"):
    # Bodies must differ, as they do in the real archive: 545 of 547 letters
    # had a unique body.
    return (f"कार्यालय {office}\nपत्रांक- ------------/{branch},\n"
            f"छपरा, दिनांक------------\nसेवा में,\nअंचल अधिकारी,\n"
            f"विषय:- {topic} {_TOPICS[i % len(_TOPICS)]} के संबंध में।\n"
            f"महाशय,\n"
            f"{_TOPICS[i % len(_TOPICS)]} के प्रसंग में {BODY}\nविश्वासभाजन")


# --- token budgeting ------------------------------------------------------
def test_token_estimate_scales_with_fertility():
    """Gemma measured 1.97 tokens/word and Qwen3 6.03; at 4096 context that
    is 2,080 Hindi words of room against 680."""
    t = "कार्यालय ज्ञापन समीक्षा बैठक सूचना प्रेषित"
    assert estimate_tokens(t, fertility=6.03) > estimate_tokens(t, fertility=1.97)


def test_budget_reserves_room_for_output():
    b = Budget(context=4096, reserve_for_output=900)
    assert b.for_prompt == 3196


def test_prompt_drops_exemplars_until_it_fits():
    """Five Hindi letter bodies overflow a 4096 context and force a prefill
    this machine cannot afford."""
    huge = ["क " * 3000] * 4
    parts = build_prompt("कुछ चाहिए", huge, budget=Budget(context=1200), max_exemplars=4)
    assert parts.dropped_exemplars >= 1
    assert parts.n_exemplars < 4


def test_prompt_keeps_at_least_one_exemplar():
    parts = build_prompt("कुछ", ["क " * 9000], budget=Budget(context=300))
    assert parts.n_exemplars == 1


def test_system_prompt_is_english():
    """It tokenises about three times cheaper than the same instruction in
    Hindi, and small models follow it at least as well."""
    parts = build_prompt("कुछ", [])
    deva = sum(1 for c in parts.system if "ऀ" <= c <= "ॿ")
    # A few Devanagari terms (पत्रांक, दिनांक) NAME the things the
    # model must not invent; naming is clearer than describing.
    assert deva / len(parts.system) < 0.05
    assert "Hindi" in parts.system


def test_exemplar_body_strips_the_furniture():
    body = _exemplar_body(_letter(1))
    lines = {l.strip() for l in body.splitlines()}
    assert any(BODY in l for l in lines)
    assert "कार्यालय जिला राजस्व शाखा" not in lines
    assert "विश्वासभाजन" not in lines and "सेवा में," not in lines


# --- deterministic repair -------------------------------------------------
@pytest.mark.parametrize("junk,label", [
    ("पत्रांक- 12345/रा०", "invented letter-number line"),
    ("दिनांक: 01.01.2026", "invented date line"),
    ("विषय:- कुछ और।", "duplicate subject line"),
    ("विश्वासभाजन", "closing block the skeleton supplies"),
    ("```", "code fence"),
])
def test_clean_body_strips_what_the_model_was_told_not_to_write(junk, label):
    """Instruction-following in a 1B model is a suggestion, not a guarantee."""
    body, removed = clean_body(f"{junk}\nयह असली मुख्य भाग है।")
    assert "यह असली मुख्य भाग है।" in body
    assert any(label in r for r in removed)


def test_clean_body_reports_rather_than_silently_editing():
    _, removed = clean_body("पत्रांक- 99\nठीक है।")
    assert removed and all(isinstance(r, str) for r in removed)


def test_clean_body_leaves_a_good_body_alone():
    body, removed = clean_body(BODY)
    assert body == BODY and removed == []


def test_inline_letter_number_is_stripped():
    body, removed = clean_body(f"इस पत्रांक 887/रा० के आलोक में {BODY}")
    assert "887" not in body and removed


# --- hallucinated facts ---------------------------------------------------
def test_invented_numbers_are_flagged():
    """The dangerous error is not bad Hindi -- a clerk sees that. It is a
    plausible, wrong file number."""
    got = unsupported_facts("राशि 45000 रुपये, पत्र 887 के आलोक में",
                            "पत्र 887 के बारे में", [])
    assert got == ["45000"]


def test_numbers_from_the_exemplars_are_not_flagged():
    assert unsupported_facts("धारा 144 के तहत", "कुछ", ["धारा 144 लागू है"]) == []


def test_devanagari_and_latin_digits_are_the_same_number():
    assert unsupported_facts("पत्रांक ८८७", "letter 887 refers", []) == []


def test_single_digits_are_not_flagged_as_facts():
    assert unsupported_facts("3 दिन में", "कुछ", []) == []


# --- assembly -------------------------------------------------------------
def _skeleton():
    return mine([_letter(i) for i in range(12)], "राजस्व", "जाँच")


def test_assembly_fills_the_date_with_a_separator():
    """Regression: the separator character class swallowed the space, so the
    line read 'दिनांक19.09.2026'."""
    out = assemble(BODY, skeleton=_skeleton(), subject="जाँच",
                   on=date(2026, 3, 15))
    assert "दिनांक 15.03.2026" in out


def test_assembly_leaves_the_letter_number_blank_by_default():
    """95% of the real archive leaves it blank; a wrong number looks finished
    and a blank one obviously is not."""
    out = assemble(BODY, skeleton=_skeleton(), subject="जाँच")
    assert BLANK_SLOT in out


def test_assembly_fills_a_supplied_letter_number():
    out = assemble(BODY, skeleton=_skeleton(), subject="जाँच", letter_number="887")
    assert "887" in out


def test_assembly_has_no_duplicate_lines():
    """Regression: two canonical forms differing only in whitespace around
    the blank marker both survived, so the date appeared twice."""
    out = assemble(BODY, skeleton=_skeleton(), subject="जाँच")
    lines = [l for l in out.splitlines() if l.strip()]
    assert len(lines) == len(set(lines))


def test_assembly_follows_the_archive_order():
    """The order in the real archive is
    letterhead → number → date → सेवा में → addressee → विषय → महाशय → body
    → closing → signatory.

    Two regressions live here. Boilerplate first came out in frequency
    order, putting the salutation above the letter number; then the
    salutation was emitted above the subject, which reads as wrong to
    anyone who writes these letters."""
    out = assemble(BODY, skeleton=_skeleton(), subject="जाँच")
    lines = [l.strip() for l in out.splitlines()]
    pos = lambda pred: next(i for i, l in enumerate(lines) if pred(l))
    num = pos(lambda l: l.startswith("पत्रांक"))
    to = lines.index("सेवा में,")
    subject = pos(lambda l: l.startswith("विषय"))
    salutation = lines.index("महाशय,")
    body = pos(lambda l: BODY in l)
    closing = lines.index("विश्वासभाजन")
    assert num < to < subject < salutation < body < closing


def test_assembly_works_without_a_skeleton():
    out = assemble(BODY, skeleton=None, subject="कुछ")
    assert BODY in out and "विश्वासभाजन" in out


# --- human-corrected skeletons -------------------------------------------
def test_edited_skeleton_round_trips():
    """Mining cannot recover line order from a heterogeneous cell. A person
    fixes it once; their edit must survive the next ingest."""
    sk = _skeleton()
    edited = parse_skeleton(sk.render(), "राजस्व", "जाँच")
    assert [l for _, _, l in edited.boilerplate] == \
           [l for _, _, l in sk.boilerplate]


def test_parse_skeleton_tolerates_a_clerk_editing_it():
    text = """# राजस्व / जाँच  (53 letters)

## fixed lines
  कार्यालय जिला राजस्व शाखा
  सेवा में,
  महाशय,

## body: write roughly 700 characters
"""
    sk = parse_skeleton(text, "राजस्व", "जाँच")
    assert [l for _, _, l in sk.boilerplate][0] == "कार्यालय जिला राजस्व शाखा"
    assert sk.median_body_chars == 700


# --- LLM client -----------------------------------------------------------
def test_qwen3_thinking_is_suppressed():
    """Qwen3 base tags emit <think> blocks that at ~9 Hindi words/second cost
    a minute the user never sees."""
    assert Ollama("qwen3:1.7b")._payload("x", None, None, False)["think"] is False
    assert "think" not in Ollama("gemma3:1b")._payload("x", None, None, False)


def test_defaults_come_from_the_phase_0_measurements():
    assert DEFAULT_OPTIONS["num_thread"] == 4      # physical cores
    assert DEFAULT_OPTIONS["num_ctx"] == 4096
    assert Ollama("gemma3:1b").keep_alive == -1    # spinning disk


def test_truncation_is_detected():
    from latters.llm import Completion
    assert Completion("x", eval_seconds=1, output_tokens=900, truncated=True).truncated


# --- end to end -----------------------------------------------------------
def _service(llm, **kw):
    # Two departments, or the classifier has nothing to learn and silently
    # declines to fit -- which is what made the request lose its department.
    # Above MIN_USEFUL_CORPUS, or every draft is (correctly) flagged as
    # coming from a corpus too thin to retrieve anything representative.
    letters = [Letter(id=i + 1, text=_letter(i), subject=f"जाँच प्रकरण {i}",
                      department="राजस्व", letter_type="जाँच", trust=0.9)
               for i in range(20)]
    letters += [Letter(id=100 + i,
                       text=_letter(i, office="जिला स्थापना शाखा",
                                    branch="स्था०", topic="वेतन भुगतान"),
                       subject=f"वेतन भुगतान {i}", department="स्थापना",
                       letter_type="भुगतान", trust=0.9)
                for i in range(20)]
    store = Store()
    store.add([LetterRow("f.docx", l.id, l.text, trust=l.trust, subject=l.subject)
               for l in letters])
    rows = store.db.execute("SELECT id FROM letters ORDER BY id").fetchall()
    for l, r in zip(letters, rows):
        l.id = r["id"]
    retriever = Retriever(store.db, letters, tfidf=TfidfIndex().fit(letters))
    return store, DraftService(
        retriever=retriever, llm=llm, skeletons={("राजस्व", "जाँच"): _skeleton()},
        classifier=TrainedClassifier.fit(
            [(l.text, l.department, l.letter_type) for l in letters], min_df=1),
        **kw)


def test_full_draft_pipeline():
    store, svc = _service(StubLLM(reply=BODY))
    with store:
        d = svc.draft("जाँच प्रतिवेदन मांगना है", department="राजस्व",
                      letter_type="जाँच", subject="जाँच के संबंध में")
    assert BODY in d.text and "विश्वासभाजन" in d.text
    assert d.sources and d.quality_verdict == "clean"
    assert not d.needs_review


def test_draft_flags_a_model_that_invents_a_number():
    store, svc = _service(StubLLM(reply="राशि 987654 रुपये स्वीकृत की जाती है।"))
    with store:
        d = svc.draft("कुछ चाहिए", department="राजस्व", letter_type="जाँच")
    assert "987654" in d.unsupported_numbers
    assert d.needs_review


def test_draft_records_its_sources():
    """A draft is only as good as the letters it came from, and the clerk has
    to be able to see which those were."""
    store, svc = _service(StubLLM(reply=BODY))
    with store:
        d = svc.draft("जाँच प्रतिवेदन", department="राजस्व", letter_type="जाँच")
    assert len(d.sources) >= 1
    assert all(isinstance(i, int) and "trust" in why for i, why in d.sources)


def test_draft_warns_when_it_is_ungrounded():
    store, svc = _service(StubLLM(reply=BODY), min_trust=0.99)
    with store:
        d = svc.draft("कुछ बिलकुल अलग", department="कोई-नहीं")
    assert any("no exemplars" in w or "nothing found" in w for w in d.warnings)


def test_classifier_recovers_the_department_from_a_free_text_request():
    """Regression: bootstrap reads branch codes and letterheads, which a
    clerk's request has neither of, so every request lost its department
    filter and its skeleton."""
    store, svc = _service(StubLLM(reply=BODY))
    with store:
        dept, ltype, how = svc.classify("दाखिल-खारिज की जाँच हेतु प्रतिवेदन चाहिए")
    assert dept == "राजस्व"
    assert how["department_source"].startswith("model")


def test_letter_type_from_the_model_is_flagged_as_unreliable():
    """Phase 3 measured 0.64 macro-F1 for letter type, below the 0.85 bar."""
    store, svc = _service(StubLLM(reply=BODY))
    with store:
        d = svc.draft("दाखिल-खारिज की जाँच हेतु प्रतिवेदन चाहिए")
    if d.letter_type:
        assert any("0.64" in w or "Confirm" in w for w in d.warnings)


def test_trained_classifier_never_claims_letter_type_is_confident():
    assert TrainedClassifier().letter_type_confident is False


# --- guards found by end-to-end testing -----------------------------------
def test_thin_corpus_is_flagged():
    """A draft from three letters looks exactly as confident as a good one."""
    store, svc = _service(StubLLM(reply=BODY))
    with store:
        svc.retriever.letters = svc.retriever.letters[:5]
        d = svc.draft("जाँच प्रतिवेदन", department="राजस्व", letter_type="जाँच")
    assert d.thin_corpus and d.needs_review
    assert any("only 5 letter" in w for w in d.warnings)


def test_draft_with_no_sources_needs_review():
    store, svc = _service(StubLLM(reply=BODY), min_trust=0.999)
    with store:
        d = svc.draft("कुछ", department="कोई-नहीं")
    assert not d.sources and d.needs_review


def test_missing_skeleton_directory_is_an_error_not_a_silent_no_op():
    """A mistyped --skeletons path used to be ignored, so the clerk's
    corrections appeared to have had no effect."""
    from latters.draft import load_skeletons
    store, _ = _service(StubLLM())
    with store:
        with pytest.raises(FileNotFoundError, match="no skeleton directory"):
            load_skeletons(store.db, overrides="/no/such/dir")


def test_a_directory_of_non_skeletons_is_not_an_error(tmp_path):
    """CONTRACT CHANGED. This used to raise, and that was wrong: an empty
    or skeleton-free directory is the normal state of a small office --
    `templates` writes nothing until a cell has 8+ letters -- and raising
    meant it could not draft at all. Files that ARE skeletons and cannot
    be parsed still raise; see test_unreadable_skeleton_files_...
    """
    import warnings
    from latters.draft import load_skeletons
    (tmp_path / "notes.txt").write_text("not a skeleton", encoding="utf-8")
    store, _ = _service(StubLLM())
    with store:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sk = load_skeletons(store.db, overrides=str(tmp_path))
        assert any("is empty" in str(w.message) for w in caught)
        assert isinstance(sk, dict)


def test_packaged_gold_set_is_found_without_a_source_checkout():
    """A wheel does not ship tests/, so an installed copy previously found no
    gold files and the deployed office had no regression harness."""
    from latters.gold import PACKAGED_GOLD, discover
    assert PACKAGED_GOLD.is_dir()
    # The packaged copy must be usable on its own...
    packaged = discover(PACKAGED_GOLD)
    assert packaged and all(p.parent == PACKAGED_GOLD for p in packaged)
    # ...and a user's own file of the same name must override it, so an
    # office can correct the seed set without editing its installation.
    assert discover()


# --- what the first real Gemma draft carried -------------------------------
#: Verbatim from the first letter this system ever generated on the office
#: machine, trimmed. Every defect below was in it.
_REAL = """यह पत्र तराना कुमार की स्थानांतरण के संबंध में है।

*   **कार्यक्षेत्र:** वर्तमान कार्यक्षेत्र अधिक उपयुक्त होंगे।
*   **परिवार संबंधी कारण:** स्थानांतरण आवश्यक है।

धन्यवाद,
भवदीय,
[आपका नाम]
[आपका पद]
[संपर्क नंबर]"""


def test_markdown_never_reaches_a_government_letter():
    """A 1B model reaches for bullets and bold because that is what its
    training data looks like. The office's letters contain neither."""
    from latters.draft import clean_body
    out, removed = clean_body(_REAL)
    assert "*" not in out and "#" not in out
    assert "कार्यक्षेत्र:" in out          # the heading survives
    assert any("markdown" in r for r in removed)


def test_invented_placeholders_are_stripped():
    """The model wrote its own signature block below the skeleton's:
    [आपका नाम] [आपका पद] [संपर्क नंबर]."""
    from latters.draft import clean_body
    out, removed = clean_body(_REAL)
    assert "[" not in out and "]" not in out
    assert any("placeholder" in r for r in removed)


def test_a_trailing_comma_no_longer_defeats_the_closing_strip():
    """THE bug: `_CLOSING_LINE` anchored to end-of-line right after the
    word, so `भवदीय,` escaped and the letter carried two sign-offs."""
    from latters.draft import clean_body
    out, _ = clean_body(_REAL)
    assert "भवदीय" not in out and "धन्यवाद" not in out


def test_the_models_salutation_is_replaced_by_the_offices():
    """Measured house style: महाशय 356 times in the archive, महोदय zero.
    The model opened महोदय -- correct Hindi, wrong register, and a second
    salutation on top of the skeleton's."""
    from latters.draft import clean_body
    out, removed = clean_body("महोदय,\n\nयह पत्र भेजा जा रहा है।")
    assert "महोदय" not in out
    assert any("salutation" in r for r in removed)


def test_the_addressee_block_is_NOT_stripped():
    """`सेवा में, जिलाधिकारी, वैशाली` is content the model derived from the
    request. assemble() already skips a skeleton line the body repeats, so
    removing it here would lose the addressee entirely."""
    from latters.draft import clean_body
    out, _ = clean_body("सेवा में,\nजिलाधिकारी,\nवैशाली।\n\nपत्र का मुख्य भाग।")
    assert "जिलाधिकारी" in out and "सेवा में" in out


# --- an empty skeleton directory is not a broken one ----------------------
def test_an_empty_skeleton_directory_warns_and_carries_on(tmp_path):
    """`templates` writes nothing until a cell has 8+ letters, so a small
    office's skeletons/ is legitimately empty -- and refusing there meant
    it could not draft at all, with the skeletons mined from the database
    (including the office-wide fallback) sitting unused."""
    import warnings
    from latters.draft import FALLBACK_CELL, load_skeletons
    from latters.store import LetterRow, Store

    empty = tmp_path / "skeletons"
    empty.mkdir()
    with Store(tmp_path / "c.db") as store:
        store.add([LetterRow(source_file="a.docx", seq=i,
                             text="कार्यालय जिला पदाधिकारी, सारण।\n"
                                  f"विषय: जाँच {i}।\nमहाशय,\n"
                                  f"मुख्य भाग {i}।\nविश्वासभाजन")
                   for i in range(12)])
        store.db.execute("UPDATE letters SET department='राजस्व', "
                         "letter_type='जाँच'")
        store.db.commit()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sk = load_skeletons(store.db, overrides=str(empty))
        assert any("is empty" in str(w.message) for w in caught)
    assert FALLBACK_CELL in sk, "the office-wide letterhead must survive"


def test_unreadable_skeleton_files_are_still_an_error(tmp_path):
    """The opposite case must stay hard: the clerk edited these, and
    silently ignoring their work is the failure the guard exists for."""
    import pytest as _pytest
    from latters.draft import load_skeletons
    from latters.store import LetterRow, Store

    bad = tmp_path / "skeletons"
    bad.mkdir()
    (bad / "broken.md").write_text("no heading, no sections\n", encoding="utf-8")
    with Store(tmp_path / "c.db") as store:
        store.add([LetterRow(source_file="a.docx", seq=1, text="पत्र")])
        with _pytest.raises(ValueError, match="none could be read"):
            load_skeletons(store.db, overrides=str(bad))


def test_the_office_wide_fallback_supplies_a_letterhead(tmp_path):
    """The headers bug: with no skeleton for the cell the draft used to get
    a subject, a body and a closing -- no पत्रांक, no दिनांक, no addressee
    block -- while the warning claimed the letterhead was 'generic'."""
    from latters.draft import FALLBACK_CELL, load_skeletons
    from latters.store import LetterRow, Store

    with Store(tmp_path / "c.db") as store:
        store.add([LetterRow(source_file="a.docx", seq=i,
                             text="कार्यालय जिला पदाधिकारी, सारण।\n"
                                  "पत्रांक- ----------/रा०\n"
                                  f"विषय: जाँच {i}।\nमहाशय,\n"
                                  f"मुख्य भाग {i}।\nविश्वासभाजन")
                   for i in range(12)])
        store.db.execute("UPDATE letters SET department='राजस्व', "
                         "letter_type='जाँच'")
        store.db.commit()
        sk = load_skeletons(store.db)

    fallback = sk[FALLBACK_CELL]
    head = "\n".join(fallback.before_subject())
    assert "कार्यालय" in head, "no letterhead in the office-wide skeleton"
