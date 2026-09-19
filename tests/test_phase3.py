import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from latters.classify import (DEFAULT_LETTER_TYPE, NaiveBayes, UNLABELLED,
                              bootstrap, cross_validate, department_features,
                              fold_rare, letter_type_features, ngrams)
from latters.fields import State, extract, normalise_date
from latters.template import canonical, mine, mine_all


# --- fields: the blank / absent distinction -------------------------------
def test_blank_is_not_absent():
    """95% of a real archive has the number left to be filled at dispatch.
    A present-but-empty field is a usable template; an absent one is a defect."""
    blank = extract("पत्रांक- -------------------/रा०,").letter_number
    absent = extract("कोई शीर्षक नहीं").letter_number
    assert blank.state is State.BLANK
    assert absent.state is State.ABSENT
    assert not blank and not absent          # neither is a usable value
    assert blank.state != absent.state       # but they mean different things


@pytest.mark.parametrize("text,expected", [
    ("पत्रांकः39-01/2023------------/स्था०, छपरा/दिनांक--", "39-01/2023"),
    ("पत्रांक-८८७, दिनांक १५.०३.२०२४", "८८७"),
    ("ज्ञापांक-203", "203"),
])
def test_real_letter_numbers_are_extracted(text, expected):
    assert extract(text).letter_number.value == expected


def test_letter_number_does_not_bleed_into_the_next_label():
    """Regression: stripping the dashes left the NEXT field's label as the
    value, so every template reported a letter number of '/दिनांक'."""
    f = extract("पत्रांक--------------------/दिनांक------------------")
    assert f.letter_number.state is State.BLANK
    assert f.letter_number.value is None


def test_letter_number_requires_a_digit():
    """Regression: the branch code '/रा०' survived the blank strip and was
    returned as the number. A real number always contains a digit."""
    assert extract("पत्रांक- --------/रा०,").letter_number.state is State.BLANK


def test_branch_code_survives_a_blank_number():
    """The branch is the department signal, and it is present even when the
    number it follows is not."""
    f = extract("पत्रांक- -------------------/रा०,")
    assert f.letter_number.state is State.BLANK
    assert f.branch.value == "रा०"


def test_branch_is_not_confused_with_the_date_label():
    assert extract("पत्रांक-----/दिनांक-----").branch.state is State.ABSENT


@pytest.mark.parametrize("raw,iso", [
    ("15.03.2024", "2024-03-15"),
    ("१५-०३-२०२४", "2024-03-15"),
    ("2/4/24", "2024-04-02"),
    ("15 मार्च 2024", "2024-03-15"),
    ("१५ सितंबर २०२३", "2023-09-15"),
])
def test_dates_normalise_day_first(raw, iso):
    """Indian official dates are day-first, always."""
    assert normalise_date(raw) == iso


@pytest.mark.parametrize("bad", ["", "कोई तारीख नहीं", "45.13.2024", "99"])
def test_bad_dates_are_rejected(bad):
    assert normalise_date(bad) is None


def test_subject_stops_at_the_sentence_end():
    f = extract("विषय:- समीक्षा बैठक की सूचना।\nमहाशय,\nयह एक लंबा शरीर है जो विषय नहीं है।")
    assert f.subject.value == "समीक्षा बैठक की सूचना"


def test_addressee_block_runs_to_the_subject():
    f = extract("सेवा में,\nअग्रणी जिला प्रबंधक,\nसारण, छपरा।\nविषय:- कुछ।")
    assert "अग्रणी जिला प्रबंधक" in f.addressee.value
    assert "विषय" not in f.addressee.value


def test_office_falls_back_to_the_signature_block():
    """A segment that starts below the letterhead still names its office in
    the signature block."""
    f = extract("कुछ पाठ यहाँ है\nऔर कुछ\nविश्वासभाजन\nप्रभारी पदाधिकारी\nजिला राजस्व शाखा, छपरा")
    assert f.office.state is State.FOUND and "राजस्व" in f.office.value


# --- classification -------------------------------------------------------
def test_bootstrap_reads_the_department_from_the_branch_code():
    b = bootstrap(extract("पत्रांक- -------/रा०, दिनांक-----"))
    assert b.department == "राजस्व" and b.department_source == "branch"


def test_bootstrap_falls_back_to_the_office_line():
    b = bootstrap(extract("कार्यालय, प्रखण्ड विकास पदाधिकारी, नगरा, सारण।\nविषय:- कुछ।"))
    assert b.department == "विकास" and b.department_source == "office"


@pytest.mark.parametrize("subject,expected", [
    ("कारण बताओ सूचना पत्र", "स्पष्टीकरण"),
    ("विभागीय कार्यवाही के संबंध में", "अनुशासनिक"),
    ("मासिक समीक्षा बैठक की सूचना", "बैठक सूचना"),
    ("दाखिल-खारिज वाद के संबंध में", "भूमि"),
    ("Cr. WJC No-1107/2023 बनाम बिहार सरकार", "न्यायालय वाद"),
    ("लोक शिकायत परिवाद संख्या 44", "परिवाद"),
])
def test_letter_type_rules(subject, expected):
    assert bootstrap(extract(f"विषय:- {subject}।")).letter_type == expected


def test_generic_correspondence_gets_a_real_label_not_unlabelled():
    """167 of 251 unmatched subjects in the real archive were the generic
    '... के संबंध में'. Calling that UNLABELLED discards 46% of the corpus."""
    b = bootstrap(extract("विषय:- कुछ असामान्य बात के संबंध में।"))
    assert b.letter_type == DEFAULT_LETTER_TYPE
    assert b.letter_type_source == "default"


def test_no_subject_means_no_type_guess():
    # (no विषय label anywhere, so there is no subject to classify from)
    assert bootstrap(extract("यह पत्र बिना किसी शीर्षक के है")).letter_type == UNLABELLED


# --- the classifier -------------------------------------------------------
def test_ngrams_are_bounded():
    g = ngrams("क" * 5000, 2, 4)
    assert g and max(len(k) for k in g) == 4 and min(len(k) for k in g) == 2


def test_naive_bayes_separates_two_obvious_classes():
    X = ["राजस्व शाखा भूमि दाखिल खारिज"] * 6 + ["स्थापना शाखा वेतन भुगतान देयक"] * 6
    y = ["राजस्व"] * 6 + ["स्थापना"] * 6
    m = NaiveBayes(min_df=1).fit(X, y)
    assert m.predict("भूमि दाखिल खारिज वाद")[0] == "राजस्व"
    assert m.predict("वेतन देयक भुगतान")[0] == "स्थापना"


def test_prediction_returns_a_confidence():
    X = ["क ख ग"] * 4 + ["य र ल"] * 4
    m = NaiveBayes(min_df=1).fit(X, ["a"] * 4 + ["b"] * 4)
    label, conf = m.predict("क ख ग")
    assert label == "a" and 0.0 < conf <= 1.0


def test_unseen_features_do_not_crash():
    m = NaiveBayes(min_df=1).fit(["क ख"] * 3 + ["ग घ"] * 3, ["a"] * 3 + ["b"] * 3)
    assert m.predict("completely unseen text")[0] in ("a", "b")


# --- honest evaluation ----------------------------------------------------
def test_cv_reports_the_majority_baseline():
    """A skewed label set makes a useless classifier look accurate. The
    baseline is reported so that cannot pass unnoticed."""
    y = ["a"] * 90 + ["b"] * 10
    X = [f"कुछ पाठ {i}" for i in range(100)]
    rep = cross_validate(X, y, k=5, min_df=1)
    assert rep.majority_baseline == pytest.approx(0.9)
    assert "lift over baseline" in rep.render()


def test_cv_warns_when_the_classifier_adds_nothing():
    y = ["a"] * 90 + ["b"] * 10
    X = ["identical text"] * 100        # no signal at all
    rep = cross_validate(X, y, k=5, min_df=1)
    assert rep.beats_baseline_by < 0.05
    assert "barely beats" in rep.render()


def test_cv_warns_about_classes_too_small_to_score():
    y = ["a"] * 50 + ["b"] * 50 + ["c"] * 3
    X = ["क" * 20] * 50 + ["ख" * 20] * 50 + ["ग" * 20] * 3
    assert "under 10 examples" in cross_validate(X, y, k=5, min_df=1).render()


def test_cv_is_stratified():
    """Every fold must see every class, or small classes score 0 by accident."""
    y = ["a"] * 40 + ["b"] * 12
    X = ["क" * 20] * 40 + ["ख" * 20] * 12
    rep = cross_validate(X, y, k=5, min_df=1)
    assert rep.per_class["b"]["recall"] > 0.5


def test_fold_rare_names_what_it_folded():
    labels = ["a"] * 20 + ["b"] * 20 + ["c"] * 3 + ["d"] * 2
    folded, dropped = fold_rare(labels, min_support=10)
    assert dropped == {"c", "d"}
    assert folded.count("अन्य") == 5
    assert set(folded) == {"a", "b", "अन्य"}


def test_featurisers_narrow_the_input():
    """Measured: department 0.773 -> 0.812 macro-F1 on the head alone;
    letter type 0.570 -> 0.630 with the subject weighted."""
    text = "कार्यालय राजस्व शाखा\n" + "क" * 2000
    assert len(department_features(text)) <= 400
    f = extract("विषय:- भूमि विवाद।\n" + "ख" * 500)
    assert letter_type_features("विषय:- भूमि विवाद।\n" + "ख" * 500, f).count("भूमि") >= 3


# --- template mining ------------------------------------------------------
def test_canonical_normalises_what_varies_between_copies():
    a = canonical("पत्रांक- -----------/रा०, दिनांक 15.03.2024")
    b = canonical("पत्रांक- ----------------/रा०, दिनांक 22.07.2025")
    assert a == b


def test_mine_finds_shared_boilerplate():
    # Bodies must differ in words, not only in numbers: canonical() folds
    # numbers away on purpose, so numerically-varying lines are the SAME line.
    bodies = ["भूमि विवाद की जाँच", "अतिक्रमण हटाने का आदेश", "जमाबंदी सुधार",
              "लगान वसूली का प्रतिवेदन", "सीमांकन हेतु दल गठन",
              "दाखिल खारिज लंबित", "नक्शा उपलब्ध कराने", "खाता विभाजन",
              "बंदोबस्ती रद्द करने", "अभिलेख सुधार हेतु", "मापी कराने का",
              "कब्जा दिलाने के संबंध"]
    letters = [f"कार्यालय जिला राजस्व शाखा\nसेवा में,\nविषय:- {b}।\n"
               f"{b} का अद्वितीय मुख्य भाग यहाँ लिखा गया है।\nविश्वासभाजन"
               for b in bodies]
    sk = mine(letters, "राजस्व", "भूमि")
    fixed = {c for c, _, _ in sk.boilerplate}
    assert "कार्यालय जिला राजस्व शाखा" in fixed
    assert "विश्वासभाजन" in fixed
    assert not any("अद्वितीय" in c for c in fixed)   # the varying body is not boilerplate
    assert 0 < sk.coverage < 1


def test_mine_reports_the_slots():
    letters = ["पत्रांक-----------\nदिनांक-----------\nकुछ पाठ यहाँ"] * 10
    sk = mine(letters, "d", "t")
    assert {label for label, _ in sk.blanks} >= {"दिनांक", "पत्रांक"}


def test_mine_all_skips_cells_too_small_to_generalise():
    """A skeleton mined from three letters is three letters' quirks."""
    rows = ([("कार्यालय एक\nमुख्य भाग", "A", "x")] * 12
            + [("कार्यालय दो\nमुख्य भाग", "B", "y")] * 3)
    out = mine_all(rows, min_cell=8)
    assert [(s.department, s.letter_type) for s in out] == [("A", "x")]


def test_skeleton_renders_something_a_clerk_can_read():
    sk = mine(["कार्यालय राजस्व\nविश्वासभाजन"] * 10, "राजस्व", "भूमि")
    r = sk.render()
    assert "राजस्व" in r and "fixed lines" in r and "body" in r
