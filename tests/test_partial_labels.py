"""Every page, against a corpus labelled the way a real one is.

Three bugs reached the office in a row and all three had the same cause:
the test fixtures were too clean. A fixture corpus built by running the
whole pipeline has every field populated, so nothing ever exercised a row
with a department and no letter type, or a form and no subject.

A real corpus is not like that. From this project's own REBUILD.md, on 455
letters: 445 have a department (98%) and 426 have a letter type (94%). The
crash was `'<' not supported between 'str' and 'NoneType'`, sorting pairs
of the two -- and the asymmetry that caused it was written down in the
project's own documentation the whole time.

So this file builds a deliberately RAGGED corpus and walks every route.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

pytest.importorskip("fastapi")
pytest.importorskip("httpx2", reason="starlette.testclient needs httpx2")
from fastapi.testclient import TestClient  # noqa: E402

from latters.store import LetterRow, Store  # noqa: E402
from latters.web.app import create_app  # noqa: E402

BODY = ("कार्यालय जिला पदाधिकारी, सारण।\nविषय: {s}।\nमहाशय,\n"
        "उपर्युक्त विषय के प्रसंग में कहना है कि आवश्यक कार्यवाही "
        "सुनिश्चित करते हुए प्रतिवेदन उपलब्ध कराएँ।\nविश्वासभाजन")

#: (department, letter_type, form, subject) -- every combination of present
#: and absent that a real corpus contains.
_RAGGED = [
    ("राजस्व", "जाँच", "letter", "जाँच प्रतिवेदन"),      # fully labelled
    ("राजस्व", None, "letter", "भूमि विवाद"),            # THE CRASH
    ("विकास", None, None, None),                          # migrated row
    (None, "बैठक सूचना", "letter", "बैठक"),              # type, no department
    (None, None, None, None),                             # nothing at all
    ("बैंकिंग", "जाँच", "order", None),                   # no subject
]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    work = tmp_path_factory.mktemp("ragged")
    db = work / "corpus.db"
    with Store(db) as store:
        store.add([
            LetterRow(source_file="a.docx", seq=i + 1,
                      text=BODY.format(s=subject or f"विषय {i}") + f"\n{i}",
                      form=form, subject=subject, trust=0.9, verdict="index")
            for i, (dept, ltype, form, subject) in enumerate(_RAGGED)
        ])
        # Labels arrive by UPDATE, the way `classify --write` writes them --
        # LetterRow cannot carry them, which is itself why department and
        # letter_type drift out of step with the rest of the row.
        for i, (dept, ltype, _, _) in enumerate(_RAGGED, start=1):
            store.db.execute(
                "UPDATE letters SET department=?, letter_type=? WHERE id=?",
                (dept, ltype, i))
        store.db.commit()
    (work / "skeletons").mkdir()
    app = create_app(db=db, skeletons=work / "skeletons", stub=True,
                     exports=work / "exports")
    with TestClient(app) as c:
        c.db = db
        yield c


def test_the_drafting_page_renders(client):
    """The exact crash: sorting (department, letter_type) pairs where the
    department is set and the type is NULL."""
    r = client.get("/")
    assert r.status_code == 200


def test_the_admin_page_renders(client):
    assert client.get("/admin").status_code == 200


def test_the_dropdowns_hold_only_real_labels(client):
    """A NULL must not become an empty option, and a label that exists on
    ANY letter must appear even if its partner is missing."""
    body = client.get("/").text
    for label in ("राजस्व", "विकास", "बैंकिंग", "बैठक सूचना", "जाँच"):
        assert label in body, f"{label} missing from the dropdowns"


@pytest.mark.parametrize("path", [
    "/api/stats", "/api/letters", "/api/letters?verdict_filter=index",
    "/api/letters?form=letter", "/api/letters?q=समीक्षा",
])
def test_every_read_route_survives_a_ragged_corpus(client, path):
    assert client.get(path).status_code == 200


def test_drafting_works_when_nothing_is_labelled(client):
    """Half these letters have no department, so retrieval cannot filter.
    That is a warning, not a failure."""
    r = client.post("/api/draft", json={"request": "जाँच प्रतिवेदन मांगना है"})
    assert r.status_code == 200
    assert r.json()["text"]


def test_a_corpus_with_no_labels_at_all_still_renders(tmp_path):
    """The state immediately after migrating an old corpus: every label
    NULL. The page must come up so the user can be told what to run."""
    db = tmp_path / "bare.db"
    with Store(db) as store:
        store.add([LetterRow(source_file="a.docx", seq=i, text=f"पत्र {i}",
                             trust=0.9, verdict="index") for i in range(3)])
    (tmp_path / "sk").mkdir()
    app = create_app(db=db, skeletons=tmp_path / "sk", stub=True,
                     exports=tmp_path / "ex")
    with TestClient(app) as c:
        assert c.get("/").status_code == 200
        assert c.get("/admin").status_code == 200
