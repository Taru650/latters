"""The web application, driven as a browser drives it.

Every bug asserted against here was found by running the app, not by reading
it: a Starlette signature change, SQLite's thread affinity, a DOCX package
LibreOffice refuses, and a corpus with no labels because `classify --write`
has no equivalent in the browser.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest.importorskip("fastapi")
pytest.importorskip("httpx2", reason="starlette.testclient needs httpx2")
from fastapi.testclient import TestClient  # noqa: E402

from latters.cli import main  # noqa: E402
from latters.web.app import create_app  # noqa: E402
from make_fixture import build  # noqa: E402

LEGACY_BODY = ("mijksDr fo\"k; ds lanHkZ esa lwfpr fd;k tkrk gS fd vko';d "
               "dk;Zokgh lqfuf'pr djrs gq, izfrosnu bl dk;kZy; dks miyC/k djk;saA")


def _archive(root: Path, n: int = 14) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for office, branch, folder in (("ftyk jktLo 'kk[kk", "jk0", "revenue"),
                                   ("ftyk LFkkiuk 'kk[kk", "LFkk0", "estab")):
        letters = []
        for i in range(n):
            letters += [
                [(f"dk;kZy; {office}", "Kruti Dev 010")],
                [(f"i=kad- ------------/{branch},", "Kruti Dev 010")],
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
def client(tmp_path_factory):
    work = tmp_path_factory.mktemp("web")
    archive = _archive(work / "archive")
    db = work / "corpus.db"
    main(["segment", str(archive), "--db", str(db)])
    main(["templates", "--db", str(db), "-o", str(work / "skeletons")])
    app = create_app(db=db, skeletons=work / "skeletons", stub=True,
                     exports=work / "exports")
    with TestClient(app) as c:
        c.workdir = work
        yield c


# --- pages ----------------------------------------------------------------
def test_pages_render(client):
    """Regression: Starlette flipped TemplateResponse's signature, so the
    context dict arrived as the template name and Jinja failed with
    'unhashable type: dict'."""
    home = client.get("/")
    assert home.status_code == 200
    assert 'id="request"' in home.text
    assert client.get("/admin").status_code == 200


def test_no_external_script_or_style_is_referenced():
    """The office machine has no internet. A CDN <script src> is a page that
    silently does not work there."""
    web = Path(__file__).resolve().parent.parent / "src" / "latters" / "web"
    for tpl in (web / "templates").glob("*.html"):
        text = tpl.read_text(encoding="utf-8")
        assert "http://" not in text and "https://" not in text, tpl.name


def test_static_files_are_served(client):
    assert client.get("/static/app.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200


# --- drafting -------------------------------------------------------------
def test_draft_returns_the_verification_surface(client):
    """Rendering the letter and hiding its provenance would undo the Phase 5
    design, so these are part of the response schema, not an afterthought."""
    r = client.post("/api/draft", json={
        "request": "जाँच प्रकरण का प्रतिवेदन मांगना है",
        "subject": "जाँच के संबंध में"})
    assert r.status_code == 200
    d = r.json()
    for key in ("sources", "removed", "invented_numbers", "warnings",
                "needs_review", "quality", "timing"):
        assert key in d, key
    assert "विश्वासभाजन" in d["text"]
    assert d["sources"], "a draft must say which letters it came from"


def test_draft_finds_the_department_from_free_text(client):
    """Regression: the browser has no `classify --write` step, so a corpus
    built through the upload page had no labels at all -- and without a
    department the drafting page loses its retrieval filter and skeleton."""
    d = client.post("/api/draft", json={
        "request": "जाँच प्रकरण का प्रतिवेदन मांगना है"}).json()
    assert d["department"] is not None


def test_a_one_word_request_is_rejected(client):
    assert client.post("/api/draft", json={"request": "hi"}).status_code == 422


def test_generation_is_serialised():
    """Two concurrent generations on an 8 GB machine swap to a spinning disk
    and hang it."""
    from latters.web.app import _GENERATION
    assert _GENERATION._value == 1


# --- export ---------------------------------------------------------------
@pytest.mark.parametrize("fmt,magic", [("txt", None), ("docx", b"PK\x03\x04")])
def test_export(client, fmt, magic):
    text = "कार्यालय जिला राजस्व शाखा\nविषय:- जाँच।\nविश्वासभाजन"
    r = client.post(f"/api/export/{fmt}", json={"text": text})
    assert r.status_code == 200 and r.content
    if magic:
        assert r.content.startswith(magic)


def test_exported_docx_is_a_readable_package(client):
    """Regression: [Content_Types].xml declared /word/styles.xml and the
    writer never shipped it. Word tolerates the dangling declaration;
    LibreOffice rejects the package outright, which surfaced as a failed
    PDF export that never mentioned styles."""
    import zipfile
    from latters.extract import read_document

    text = "कार्यालय जिला राजस्व शाखा\nविषय:- जाँच।\nविश्वासभाजन"
    out = client.workdir / "exported.docx"
    out.write_bytes(client.post("/api/export/docx", json={"text": text}).content)
    parts = set(zipfile.ZipFile(out).namelist())
    assert {"[Content_Types].xml", "_rels/.rels", "word/document.xml",
            "word/styles.xml", "word/_rels/document.xml.rels"} <= parts
    doc = read_document(out)
    assert "कार्यालय" in doc.blocks[0].raw_text


def test_empty_export_is_rejected(client):
    assert client.post("/api/export/docx", json={"text": "  "}).status_code == 422


def test_unknown_export_format_is_rejected(client):
    assert client.post("/api/export/exe", json={"text": "क"}).status_code == 400


# --- admin ----------------------------------------------------------------
def test_stats(client):
    s = client.get("/api/stats").json()
    assert s["letters"] > 0 and "by_form" in s and "by_verdict" in s


def test_letters_list_is_lowest_trust_first(client):
    rows = client.get("/api/letters?limit=10").json()
    assert rows
    assert [r["trust"] for r in rows] == sorted(r["trust"] for r in rows)


def test_letters_filter_by_verdict(client):
    rows = client.get("/api/letters?verdict_filter=index&limit=5").json()
    assert all(r["verdict"] == "index" for r in rows)


def test_letters_search(client):
    assert client.get("/api/letters?q=जाँच&limit=5").json()


def test_correcting_a_letter_rescores_it(client):
    """The reason to correct a letter is usually that its conversion was
    wrong, which means its trust score was computed from the wrong text."""
    lid = client.get("/api/letters?limit=1").json()[0]["id"]
    before = client.get(f"/api/letter/{lid}").json()
    good = ("कार्यालय जिला राजस्व शाखा\nपत्रांक- ------------/रा०,\n"
            "छपरा, दिनांक------------\nसेवा में,\nअंचल अधिकारी,\n"
            "विषय:- सुधारा हुआ प्रकरण।\nमहाशय,\n"
            "यह सुधारा हुआ पाठ है और इतना लंबा है कि एक पूरा खंड बने। "
            "आवश्यक कार्यवाही सुनिश्चित करें।\nविश्वासभाजन")
    r = client.post(f"/api/letters/{lid}", json={"text": good}).json()
    assert r["trust"] > (before["trust"] or 0)
    assert r["form"] == "letter"
    assert client.get(f"/api/letter/{lid}").json()["text"] == good


def test_correction_cannot_empty_a_letter(client):
    lid = client.get("/api/letters?limit=1").json()[0]["id"]
    assert client.post(f"/api/letters/{lid}", json={"text": " "}).status_code == 422


def test_delete(client):
    lid = client.get("/api/letters?limit=1").json()[0]["id"]
    assert client.delete(f"/api/letters/{lid}").status_code == 200
    assert client.get(f"/api/letter/{lid}").status_code == 404
    assert client.delete(f"/api/letters/{lid}").status_code == 404


def test_upload_ingests_and_labels(client, tmp_path):
    root = _archive(tmp_path / "more", n=3)
    before = client.get("/api/stats").json()["letters"]
    files = [("files", (p.name, p.read_bytes(),
                        "application/vnd.openxmlformats-officedocument."
                        "wordprocessingml.document"))
             for p in root.glob("*.docx")]
    r = client.post("/api/upload", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["files"] and "labelled" in body
    assert client.get("/api/stats").json()["letters"] >= before


def test_upload_rejects_non_docx(client):
    r = client.post("/api/upload",
                    files=[("files", ("old.doc", b"\xd0\xcf\x11\xe0", "application/msword"))])
    assert r.status_code == 422
    assert "LibreOffice" in r.json()["detail"]


# --- skeletons ------------------------------------------------------------
def test_skeleton_round_trip(client):
    name = sorted(p.name for p in (client.workdir / "skeletons").glob("*.md"))[0]
    text = client.get(f"/api/skeleton/{name}").json()["text"]
    assert "## above the subject line" in text
    assert client.post(f"/api/skeleton/{name}",
                       json={"text": text}).status_code == 200


def test_a_skeleton_without_its_headings_is_rejected(client):
    """The headings carry the structure. A signatory line and a body line
    are identical out of context, so a file without them cannot be used."""
    name = sorted(p.name for p in (client.workdir / "skeletons").glob("*.md"))[0]
    r = client.post(f"/api/skeleton/{name}", json={"text": "just some text\n"})
    assert r.status_code == 422
    assert "headings" in r.json()["detail"]


@pytest.mark.parametrize("name", ["../../../etc/passwd", "..%2Fsecret.md",
                                  "notes.txt", "x.md"])
def test_skeleton_path_traversal_is_impossible(client, name):
    """The name arrives from a URL, so escaping the directory has to be
    impossible rather than merely unlikely."""
    assert client.get(f"/api/skeleton/{name}").status_code in (400, 404)
