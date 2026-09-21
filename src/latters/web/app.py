"""FastAPI application.

Three things here are load-bearing rather than incidental:

**One generation at a time.** An ``asyncio.Semaphore(1)`` guards the model.
Two concurrent 2.5 GB generations on an 8 GB machine swap to a spinning disk
and hang it. The second request waits and is told it is waiting.

**The verification surface is not optional.** Every draft is returned with
its sources and their trust scores, what was stripped from the model's
output, and any number the model invented. Phase 5 made those available; a
UI that renders the letter and hides them would undo the whole design. They
are part of the response schema, not an afterthought a later version adds.

**Blocking work never runs on the event loop.** Conversion, segmentation and
generation are all slow and synchronous. They go to a thread, so uploading
an archive does not freeze the page someone else is drafting on.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..classify import TrainedClassifier
from ..draft import Budget, DraftService, load_skeletons
from ..extract import UnsupportedFormat, convert_document, read_document
from ..fields import extract as extract_fields
from ..llm import LLM, Ollama, OllamaError, StubLLM
from ..repair import repair as repair_text
from ..retrieve import Filters, Retriever, TfidfIndex, load_letters
from ..segment import segment as split_letters, trust as trust_score, verdict
from ..store import LetterRow, Store
from ..template import parse_skeleton
from ..validate import assess, build_vocabulary

HERE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(HERE / "templates"))

#: Generations are serialised. See the module docstring.
_GENERATION = asyncio.Semaphore(1)

#: What the admin upload box accepts. .doc and .rtf are excluded on
#: purpose: LibreOffice must convert them first or the per-run font
#: information -- the whole basis of the legacy conversion -- is lost.
UPLOAD_SUFFIXES = (".docx", ".pdf", ".png", ".jpg", ".jpeg",
                   ".tif", ".tiff", ".bmp")

_MEDIA_TYPES = {
    "txt": "text/plain; charset=utf-8",
    "pdf": "application/pdf",
    "docx": ("application/vnd.openxmlformats-officedocument"
             ".wordprocessingml.document"),
}


class Workspace:
    """Everything the app needs, rebuilt when the corpus changes.

    The indexes are held in memory and invalidated on write rather than
    rebuilt per request: TF-IDF over 5,000 letters takes a couple of
    seconds, which is fine once and not fine on every keystroke.
    """

    def __init__(self, db_path: Path, skeleton_dir: Path, llm: LLM,
                 *, template_docx: Path | None = None):
        self.db_path = Path(db_path)
        self.skeleton_dir = Path(skeleton_dir)
        self.llm = llm
        self.template_docx = template_docx
        self._store: Store | None = None
        self._service: DraftService | None = None
        self._dirty = True

    @property
    def store(self) -> Store:
        if self._store is None:
            self._store = Store(self.db_path)
        return self._store

    def invalidate(self) -> None:
        self._dirty = True

    def ensure_labels(self) -> int:
        """Bootstrap department and letter-type labels for unlabelled rows.

        `latters classify --write` is a command-line step with no equivalent
        in the browser, so a corpus built entirely through the upload page
        had no labels at all -- and without a department the drafting page
        silently loses both its retrieval filter and its skeleton. Labelling
        happens on ingest instead, from the branch code in the letter number
        and the office line in the letterhead.
        """
        from ..classify import UNLABELLED, bootstrap

        rows = self.store.db.execute(
            "SELECT id, text FROM letters "
            "WHERE department IS NULL OR letter_type IS NULL").fetchall()
        n = 0
        for row in rows:
            b = bootstrap(extract_fields(row["text"]), row["text"])
            dept = None if b.department == UNLABELLED else b.department
            ltype = None if b.letter_type == UNLABELLED else b.letter_type
            if dept or ltype:
                self.store.write(
                    "UPDATE letters SET department=COALESCE(?, department), "
                    "letter_type=COALESCE(?, letter_type) WHERE id=?",
                    (dept, ltype, row["id"]))
                n += 1
        if n:
            self.invalidate()
        return n

    @property
    def service(self) -> DraftService:
        if self._service is None or self._dirty:
            self.ensure_labels()
            letters = load_letters(self.store.db)
            retriever = Retriever(self.store.db, letters,
                                  tfidf=TfidfIndex().fit(letters))
            rows = [(r["text"], r["department"], r["letter_type"])
                    for r in self.store.db.execute(
                        "SELECT text, department, letter_type FROM letters")]
            overrides = (str(self.skeleton_dir)
                         if self.skeleton_dir.is_dir()
                         and any(self.skeleton_dir.glob("*.md")) else None)
            self._service = DraftService(
                retriever=retriever, llm=self.llm,
                skeletons=load_skeletons(self.store.db, overrides=overrides),
                classifier=TrainedClassifier.fit(rows), budget=Budget())
            self._dirty = False
        return self._service

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None


def create_app(db: str | Path = "corpus.db",
               skeletons: str | Path = "skeletons",
               *, model: str = "gemma3:1b", host: str = "http://127.0.0.1:11434",
               stub: bool = False, template_docx: str | Path | None = None,
               exports: str | Path = "exports") -> FastAPI:
    llm: LLM = StubLLM(
        reply="उपर्युक्त विषय के प्रसंग में कहना है कि आवश्यक कार्यवाही "
              "सुनिश्चित करते हुए प्रतिवेदन इस कार्यालय को उपलब्ध कराएँ।"
    ) if stub else Ollama(model, host=host)

    ws = Workspace(Path(db), Path(skeletons), llm,
                   template_docx=Path(template_docx) if template_docx else None)
    export_dir = Path(exports)
    export_dir.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        ws.close()

    app = FastAPI(title="latters", docs_url=None, redoc_url=None,
                  lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")
    app.state.ws = ws
    app.state.export_dir = export_dir
    app.state.model = "stub" if stub else model

    # ---------------------------------------------------------------- pages
    @app.get("/", response_class=HTMLResponse)
    async def drafting_page(request: Request):
        stats = ws.store.stats()
        cells = sorted({(r["department"], r["letter_type"]) for r in ws.store.db.execute(
            "SELECT DISTINCT department, letter_type FROM letters "
            "WHERE department IS NOT NULL")})
        return templates.TemplateResponse(request, "draft.html", {
            "stats": stats, "model": app.state.model,
            "departments": sorted({d for d, _ in cells if d}),
            "letter_types": sorted({t for _, t in cells if t}),
        })

    @app.get("/admin", response_class=HTMLResponse)
    async def admin_page(request: Request):
        # Say up front which readers are missing. Finding out that scans
        # cannot be read by uploading forty of them and getting forty
        # identical errors is the wrong order.
        from ..ocr import available, ocr_languages
        tools = available()
        return templates.TemplateResponse(request, "admin.html", {
            "stats": ws.store.stats(), "model": app.state.model,
            "skeletons": sorted(p.name for p in ws.skeleton_dir.glob("*.md"))
                         if ws.skeleton_dir.is_dir() else [],
            "ocr": {"missing": [n for n, ok in tools.items() if not ok],
                    "hindi": "hin" in ocr_languages()},
        })

    # --------------------------------------------------------------- drafting
    @app.post("/api/draft")
    async def api_draft(payload: dict):
        request_text = (payload.get("request") or "").strip()
        if len(request_text) < 10:
            raise HTTPException(422, "Describe the letter you need in a sentence.")

        if _GENERATION.locked():
            waiting = True
        else:
            waiting = False
        async with _GENERATION:
            try:
                d = await asyncio.to_thread(
                    ws.service.draft, request_text,
                    department=payload.get("department") or None,
                    letter_type=payload.get("letter_type") or None,
                    subject=(payload.get("subject") or "").strip() or None,
                    letter_number=(payload.get("letter_number") or "").strip() or None,
                )
            except OllamaError as exc:
                raise HTTPException(503, str(exc))

        # Phase 7: log every generation, exported or not. See store.py --
        # a table that only held exported drafts would report the tool as
        # flawless while people quietly stopped using it.
        draft_id = await asyncio.to_thread(
            ws.store.record_draft, request=request_text, draft_text=d.text,
            department=d.department, letter_type=d.letter_type,
            seconds=d.seconds, model=app.state.model,
            needs_review=d.needs_review)

        return JSONResponse({
            "draft_id": draft_id,
            "text": d.text, "body": d.body,
            "department": d.department, "letter_type": d.letter_type,
            "subject": d.subject,
            # The verification surface. See the module docstring: rendering
            # the letter without these would undo the Phase 5 design.
            "sources": [{"id": i, "why": w} for i, w in d.sources],
            "removed": d.removed_from_model_output,
            "invented_numbers": d.unsupported_numbers,
            "warnings": d.warnings,
            "needs_review": d.needs_review,
            "quality": {"score": d.quality_score, "verdict": d.quality_verdict},
            "timing": {"seconds": round(d.seconds, 1),
                       "prompt_tokens": d.prompt_tokens,
                       "output_tokens": d.output_tokens},
            "queued": waiting,
        })

    @app.get("/api/letter/{letter_id}")
    async def api_letter(letter_id: int):
        row = ws.store.get(letter_id)
        if row is None:
            raise HTTPException(404, "no such letter")
        return JSONResponse({k: row[k] for k in row.keys()})

    # ----------------------------------------------------------------- export
    @app.post("/api/export/{fmt}")
    async def api_export(fmt: str, payload: dict):
        if fmt not in ("docx", "pdf", "txt"):
            raise HTTPException(400, "format must be docx, pdf or txt")
        text = (payload.get("text") or "").strip()
        if not text:
            raise HTTPException(422, "nothing to export")
        # The other half of the pair. An unknown or absent draft_id is not
        # an error: the letter is written, and refusing to hand it over to
        # protect a statistic would be the wrong trade.
        draft_id = payload.get("draft_id")
        if isinstance(draft_id, int):
            await asyncio.to_thread(ws.store.record_dispatch, draft_id, text, fmt)

        name = f"letter-{date.today():%Y%m%d}-{uuid.uuid4().hex[:8]}"
        path = await asyncio.to_thread(
            _write_export, text, fmt, export_dir / name, ws.template_docx)
        # Send the real type. octet-stream makes Windows offer "open with"
        # instead of Word, and a mail client attaching the file passes the
        # wrong type on to the recipient -- for a letter that goes out of the
        # office, that is the reader's problem, not ours.
        return FileResponse(path, filename=path.name,
                            media_type=_MEDIA_TYPES[fmt])

    # ------------------------------------------------------------------ admin
    @app.post("/api/upload")
    async def api_upload(files: list[UploadFile]):
        staged = Path(tempfile.mkdtemp(prefix="latters-upload-"))
        try:
            saved = []
            for f in files:
                name = (f.filename or "").lower()
                if not name.endswith(UPLOAD_SUFFIXES):
                    continue
                dest = staged / Path(f.filename).name
                dest.write_bytes(await f.read())
                saved.append(dest)
            if not saved:
                raise HTTPException(
                    422, "Nothing readable here. Accepted: .docx, .pdf, and "
                         "scans or photographs (.png .jpg .tif). Convert .doc "
                         "and .rtf first with LibreOffice -- it preserves the "
                         "per-run font information this pipeline depends on.")
            result = await asyncio.to_thread(_ingest, ws, staged)
        finally:
            shutil.rmtree(staged, ignore_errors=True)
        ws.invalidate()
        result["labelled"] = await asyncio.to_thread(ws.ensure_labels)
        return JSONResponse(result)

    @app.get("/api/letters")
    async def api_letters(q: str = "", verdict_filter: str = "",
                          form: str = "", limit: int = 50):
        if q.strip():
            rows = ws.store.search(q.strip(), limit=limit, min_trust=0.0)
        else:
            sql = "SELECT * FROM letters"
            params: list = []
            where = []
            if verdict_filter:
                where.append("verdict = ?")
                params.append(verdict_filter)
            if form:
                where.append("form = ?")
                params.append(form)
            if where:
                sql += " WHERE " + " AND ".join(where)
            sql += " ORDER BY trust ASC LIMIT ?"
            params.append(limit)
            rows = ws.store.db.execute(sql, params).fetchall()
        return JSONResponse([{
            "id": r["id"], "source_file": r["source_file"], "seq": r["seq"],
            "subject": r["subject"], "department": r["department"],
            "letter_type": r["letter_type"], "form": r["form"],
            "trust": r["trust"], "verdict": r["verdict"],
            "missing": r["missing"], "violations": r["violations"],
            "preview": (r["text"] or "")[:160],
        } for r in rows])

    @app.post("/api/letters/{letter_id}")
    async def api_correct(letter_id: int, payload: dict):
        """Re-score a letter after a human corrects its text.

        The corrected text has to be re-validated, not just stored: the point
        of correcting it is usually that the conversion was wrong, and its
        trust score was computed from the wrong text.
        """
        row = ws.store.get(letter_id)
        if row is None:
            raise HTTPException(404, "no such letter")
        text = (payload.get("text") or "").strip()
        if not text:
            raise HTTPException(422, "letter text cannot be empty")

        q = assess(text)
        segs = split_letters(text)
        completeness = segs[0].completeness() if segs else 0.0
        form = segs[0].form.value if segs else None
        score = trust_score(q.score, completeness, row["source_tier"])
        subject = extract_fields(text).subject
        ws.store.write(
            """UPDATE letters SET text=?, subject=?, conversion_confidence=?,
                   completeness=?, trust=?, verdict=?, form=?,
                   violations=?, missing=?
               WHERE id=?""",
            (text, subject.value if subject else None, q.score, completeness,
             score, verdict(score), form,
             json.dumps(q.violations, ensure_ascii=False),
             json.dumps(segs[0].missing() if segs else [], ensure_ascii=False),
             letter_id))
        ws.invalidate()
        return JSONResponse({"id": letter_id, "trust": score,
                             "verdict": verdict(score), "form": form,
                             "quality": q.score, "violations": q.violations})

    @app.delete("/api/letters/{letter_id}")
    async def api_delete(letter_id: int):
        if not ws.store.delete(letter_id):
            raise HTTPException(404, "no such letter")
        ws.invalidate()
        return JSONResponse({"deleted": letter_id})

    @app.get("/api/skeleton/{name}")
    async def api_skeleton(name: str):
        path = _safe_skeleton_path(ws.skeleton_dir, name)
        if not path.exists():
            raise HTTPException(404, "no such skeleton")
        return JSONResponse({"name": name,
                             "text": path.read_text(encoding="utf-8")})

    @app.post("/api/skeleton/{name}")
    async def api_save_skeleton(name: str, payload: dict):
        path = _safe_skeleton_path(ws.skeleton_dir, name)
        text = payload.get("text") or ""
        try:
            parsed = parse_skeleton(text, "x", "y")
        except Exception as exc:
            raise HTTPException(422, f"could not parse: {exc}")
        if not parsed.boilerplate:
            raise HTTPException(
                422, "No lines found. Keep the '## above the subject line' "
                     "and '## below the body' headings — they carry the "
                     "structure, and a line cannot be placed without them.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        ws.invalidate()
        return JSONResponse({"name": name, "lines": len(parsed.boilerplate)})

    @app.get("/api/stats")
    async def api_stats():
        return JSONResponse(ws.store.stats())

    return app


# --------------------------------------------------------------------------
def _safe_skeleton_path(root: Path, name: str) -> Path:
    """Reject anything that escapes the skeleton directory.

    The name arrives from a URL, so `../../etc/passwd` has to be impossible
    rather than merely unlikely.
    """
    candidate = (root / Path(name).name).resolve()
    if candidate.parent != root.resolve() or not candidate.name.endswith(".md"):
        raise HTTPException(400, "bad skeleton name")
    return candidate


def _ingest(ws: Workspace, folder: Path) -> dict:
    """Convert, segment, score and store an uploaded batch. Runs in a thread."""
    converted, skipped, notes = [], [], []
    for path in sorted(p for p in folder.iterdir()
                       if p.suffix.lower() in UPLOAD_SUFFIXES):
        suffix = path.suffix.lower()
        try:
            if suffix == ".docx":
                doc, report = read_document(path), None
            else:
                from ..ocr import read_image, read_pdf
                doc, report = (read_pdf(path) if suffix == ".pdf"
                               else read_image(path))
        except (UnsupportedFormat, Exception) as exc:
            skipped.append({"file": path.name,
                            "reason": str(exc).splitlines()[0]})
            continue

        # A PDF or a scan is already Unicode, so every run carries font=None
        # and convert_document passes it through. Calling it anyway keeps one
        # code path and costs nothing.
        text, _ = convert_document(doc, rescue_latin=True)
        text, _ = repair_text(text)

        # The tier and the OCR confidence travel with the text. Without the
        # confidence an OCR'd letter scores like a clean DOCX: the validator
        # only catches ILLEGAL Devanagari, and a misrecognised word is
        # normally perfectly legal Devanagari that is simply wrong.
        tier = "docx" if report is None else report.tier
        penalty = 1.0 if report is None else report.confidence
        converted.append((path.name, text, tier, penalty))
        if report is not None:
            notes.extend(report.warnings)
            if report.tier == "ocr":
                notes.append(
                    f"{path.name}: read by OCR over {report.ocr_pages} page(s), "
                    f"mean confidence "
                    f"{report.mean_word_confidence:.0f}/100. OCR text is a "
                    f"transcription, not the original -- read it before you "
                    f"rely on it, and check the letter number by eye.")

    existing = [r["text"] for r in ws.store.db.execute("SELECT text FROM letters")]
    vocab = build_vocabulary(existing + [t for _, t, _, _ in converted])

    rows: list[LetterRow] = []
    per_file = []
    for name, text, tier, penalty in converted:
        segs = split_letters(text)
        per_file.append({"file": name, "letters": len(segs), "source": tier})
        for i, seg in enumerate(segs, 1):
            q = assess(seg.text, vocabulary=vocab or None)
            comp = seg.completeness()
            conv = q.score * penalty
            score = trust_score(conv, comp, tier)
            subject = extract_fields(seg.text).subject
            rows.append(LetterRow(
                source_file=name, seq=i, text=seg.text,
                start_line=seg.start_line, end_line=seg.end_line,
                source_tier=tier, conversion_confidence=conv,
                completeness=comp, trust=score, verdict=verdict(score),
                opened_by=seg.opened_by, form=seg.form.value,
                anchors=seg.anchors, violations=q.violations,
                missing=seg.missing(),
                subject=subject.value if subject else None))
    inserted, duplicates = ws.store.add(rows)
    return {"files": per_file, "skipped": skipped, "inserted": inserted,
            "duplicates": duplicates, "notes": notes,
            "stats": ws.store.stats()}


def _write_export(text: str, fmt: str, stem: Path,
                  template_docx: Path | None) -> Path:
    from .. import docx_writer as D

    if fmt == "txt":
        out = stem.with_suffix(".txt")
        out.write_text(text, encoding="utf-8")
        return out

    docx = stem.with_suffix(".docx")
    # Unicode Devanagari, not the legacy font: the archive is converted, and
    # writing the output back in Kruti Dev would recreate the problem this
    # project exists to solve.
    body = [D.para(D.run(line or " ", "Nirmala UI", size_pt=12))
            for line in text.split("\n")]
    D.write(docx, body)
    if fmt == "docx":
        return docx

    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise HTTPException(
            501, "PDF export needs LibreOffice, which is not installed. "
                 "Export DOCX and print to PDF from Word instead — the "
                 "letter is identical either way.")

    # ReportLab is not an option: it does not shape Devanagari conjuncts, so
    # the PDF would be subtly wrong in a way nobody notices until it is
    # printed. LibreOffice reuses the system's own shaping engine.
    try:
        proc = subprocess.run(
            [soffice, "--headless", "--convert-to", "pdf",
             "--outdir", str(docx.parent), str(docx)],
            capture_output=True, timeout=180)
    except subprocess.TimeoutExpired:
        raise HTTPException(
            504, "LibreOffice did not finish within three minutes. Export "
                 "DOCX instead; the first LibreOffice run on a machine is "
                 "much slower than later ones.")

    pdf = docx.with_suffix(".pdf")
    if not pdf.exists():
        # Say what LibreOffice said. "Produced no PDF" sends the reader
        # looking at their letter, when the cause is nearly always a broken
        # or first-run LibreOffice profile and has nothing to do with the
        # document -- in one environment it could not convert a plain text
        # file either.
        detail = (proc.stderr or b"").decode("utf-8", "replace").strip()
        detail = " / ".join(l for l in detail.splitlines()
                            if l.strip() and "javaldx" not in l)[:300]
        raise HTTPException(
            502, "LibreOffice could not produce a PDF"
                 + (f": {detail}. " if detail else ". ")
                 + "The DOCX export is unaffected — use that and print to "
                   "PDF from Word. To check LibreOffice itself, try "
                   "converting any file from the command line.")
    return pdf
