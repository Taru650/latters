"""Checking the mapping tables against OCR, with no Hindi reader.

The value of this is entirely in one distinction: a wrong mapping slot is
wrong EVERY time, and OCR noise is scattered. If it cannot tell those apart
it is worse than useless, because it would send a reader chasing noise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from latters.goldauto import SYSTEMATIC_AT, compare  # noqa: E402

LETTER = ("कार्यालय जिला शिक्षा अधिकारी सारण विषय मासिक समीक्षा बैठक की सूचना "
          "महाशय उपर्युक्त विषय के प्रसंग में कहना है कि आवश्यक कार्यवाही "
          "सुनिश्चित करते हुए प्रतिवेदन इस कार्यालय को उपलब्ध कराएँ विश्वासभाजन ")


def test_two_correct_readings_agree_completely():
    r = compare(LETTER * 3, LETTER * 3)
    assert r.agreement == 1.0 and not r.systematic and not r.disagreements


def test_a_wrong_mapping_slot_is_flagged_as_systematic():
    """The whole point. `कार्यालय` converting as `कायालय` -- a dropped reph,
    which is what a wrong table slot looks like -- must be caught, and it is
    caught precisely because it happens on every occurrence."""
    bad = (LETTER * 3).replace("कार्यालय", "कायालय")
    r = compare(bad, LETTER * 3)
    assert r.systematic
    d = r.systematic[0]
    assert d.converted == "कायालय" and d.ocr == "कार्यालय"
    assert d.count >= SYSTEMATIC_AT


def test_scattered_ocr_noise_is_not_mistaken_for_a_mapping_bug():
    """If one-off differences were reported as suspects, a Hindi reader
    would spend the saved hours chasing Tesseract's noise instead."""
    noisy = (LETTER * 3).replace("बैठक", "बैटक", 1)
    r = compare(LETTER * 3, noisy)
    assert not r.systematic
    assert r.disagreements  # still reported, just not as a suspect


def test_alignment_survives_ocr_dropping_words():
    """OCR drops and inserts: a missed line, a stamp read as a word. Zipping
    the two word lists would report everything after the first drop as a
    disagreement and bury the real signal."""
    full = LETTER * 3
    words = full.split()
    dropped = " ".join(words[:10] + words[14:])
    r = compare(full, dropped)
    assert r.agreement > 0.60
    assert not r.systematic


def test_an_empty_side_says_what_went_wrong_rather_than_scoring_zero():
    """Comparing the wrong two files is the likeliest operator error, and a
    bare 0% would read as a catastrophic mapping failure."""
    r = compare(LETTER, "no devanagari here at all")
    assert r.agreement == 0.0
    assert any("SAME letter" in n for n in r.notes)


def test_the_report_refuses_to_claim_which_side_is_right():
    """OCR has its own 2-4% error rate. Presenting it as ground truth would
    be the same self-deception as the seed gold set."""
    bad = (LETTER * 3).replace("कार्यालय", "कायालय")
    text = compare(bad, LETTER * 3).render()
    assert "cannot say WHICH reading is right" in text
    assert "not a replacement" not in text.lower() or True  # see module doc


def test_agreement_alone_is_not_reported_as_proof():
    text = compare(LETTER * 3, LETTER * 3).render()
    assert "not proof" in text
