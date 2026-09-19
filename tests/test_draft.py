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


def _letter(i, office="जिला राजस्व शाखा", branch="रा०", topic="जाँच प्रकरण"):
    # Bodies must differ, as they do in the real archive: 545 of 547 letters
    # had a unique body.
    return (f"कार्यालय {office}\nपत्रांक- ------------/{branch},\n"
            f"छपरा, दिनांक------------\nसेवा में,\nअंचल अधिकारी,\n"
            f"विषय:- {topic} {i} के संबंध में।\nमहाशय,\n"
            f"{topic} {i} के प्रसंग में {BODY}\nविश्वासभाजन")


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


def test_assembly_orders_letterhead_before_salutation():
    """Regression: boilerplate came out in frequency order, which put the
    salutation above the letter number."""
    out = assemble(BODY, skeleton=_skeleton(), subject="जाँच")
    lines = [l.strip() for l in out.splitlines()]
    num = next(i for i, l in enumerate(lines) if l.startswith("पत्रांक"))
    assert num < lines.index("सेवा में,")
    assert lines.index("सेवा में,") < lines.index("महाशय,")
    assert lines.index("महाशय,") < next(i for i, l in enumerate(lines)
                                        if l.startswith("विषय"))


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
    letters = [Letter(id=i + 1, text=_letter(i), subject=f"जाँच प्रकरण {i}",
                      department="राजस्व", letter_type="जाँच", trust=0.9)
               for i in range(12)]
    letters += [Letter(id=100 + i,
                       text=_letter(i, office="जिला स्थापना शाखा",
                                    branch="स्था०", topic="वेतन भुगतान"),
                       subject=f"वेतन भुगतान {i}", department="स्थापना",
                       letter_type="भुगतान", trust=0.9)
                for i in range(12)]
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
