"""Minimal DOCX writer (stdlib only) for the gold-set review sheet.

Only what the review sheet needs: a heading, paragraphs, and a table whose
cells can each carry their own font. The font-per-cell part is the whole
point — the review sheet has to render the legacy text in the legacy font so
a Hindi reader can see what the page originally said.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'

_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""


def run(text: str, font: str | None = None, *, size_pt: int | None = None,
        bold: bool = False, color: str | None = None) -> str:
    props = []
    if font:
        props.append(f'<w:rFonts w:ascii="{escape(font)}" w:hAnsi="{escape(font)}" '
                     f'w:cs="{escape(font)}"/>')
    if bold:
        props.append("<w:b/>")
    if size_pt:
        props.append(f'<w:sz w:val="{size_pt * 2}"/><w:szCs w:val="{size_pt * 2}"/>')
    if color:
        props.append(f'<w:color w:val="{color}"/>')
    rpr = f"<w:rPr>{''.join(props)}</w:rPr>" if props else ""
    # Word collapses runs of spaces without xml:space="preserve".
    return f'<w:r>{rpr}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def para(runs: str | list[str], *, spacing_after: int = 120) -> str:
    body = runs if isinstance(runs, str) else "".join(runs)
    return (f'<w:p><w:pPr><w:spacing w:after="{spacing_after}"/></w:pPr>'
            f"{body}</w:p>")


def heading(text: str, *, size_pt: int = 16) -> str:
    return para(run(text, "Calibri", size_pt=size_pt, bold=True), spacing_after=200)


def _cell(paragraphs: list[str], width_dxa: int) -> str:
    return (f'<w:tc><w:tcPr><w:tcW w:w="{width_dxa}" w:type="dxa"/>'
            f'<w:tcBorders>'
            f'<w:top w:val="single" w:sz="4" w:color="BFBFBF"/>'
            f'<w:bottom w:val="single" w:sz="4" w:color="BFBFBF"/>'
            f'<w:left w:val="single" w:sz="4" w:color="BFBFBF"/>'
            f'<w:right w:val="single" w:sz="4" w:color="BFBFBF"/>'
            f'</w:tcBorders></w:tcPr>'
            f'{"".join(paragraphs) or "<w:p/>"}</w:tc>')


def table(rows: list[list[list[str]]], widths: list[int]) -> str:
    """`rows[r][c]` is a list of paragraph XML strings for that cell."""
    out = ['<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/>'
           '<w:tblLayout w:type="fixed"/></w:tblPr><w:tblGrid>'
           + "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
           + "</w:tblGrid>"]
    for row in rows:
        out.append("<w:tr>" + "".join(_cell(c, widths[i]) for i, c in enumerate(row)) + "</w:tr>")
    out.append("</w:tbl>")
    return "".join(out)


def write(path: Path, body_parts: list[str]) -> Path:
    document = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                f'<w:document {W}><w:body>{"".join(body_parts)}'
                f'<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
                f'<w:pgMar w:top="720" w:right="720" w:bottom="720" w:left="720"/>'
                f'</w:sectPr></w:body></w:document>')
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _CONTENT_TYPES)
        z.writestr("_rels/.rels", _RELS)
        z.writestr("word/document.xml", document)
    return path
