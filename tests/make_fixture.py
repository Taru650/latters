"""Build a synthetic .docx that reproduces the mixed-font reality of a real
departmental letter: Kruti Dev body, Latin letter number, Unicode signature.

Used by the tests so the suite needs no binary fixtures in git.
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

W_NS = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'

_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

_STYLES = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles {W_NS}>
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/></w:rPr></w:rPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:styleId="HindiBody"><w:rPr><w:rFonts w:ascii="Kruti Dev 010" w:hAnsi="Kruti Dev 010"/></w:rPr></w:style>
</w:styles>"""


def _run(text: str, font: str | None) -> str:
    rpr = f'<w:rPr><w:rFonts w:ascii="{font}" w:hAnsi="{font}"/></w:rPr>' if font else ""
    return f'<w:r>{rpr}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def build(path: Path, paragraphs: list[list[tuple[str, str | None]]],
          styled_paragraphs: list[str] | None = None) -> Path:
    """`paragraphs` is a list of paragraphs, each a list of (text, font) runs.

    `styled_paragraphs` are texts with no run font at all -- they inherit the
    HindiBody paragraph style, which exercises style-level font resolution.
    """
    body = []
    for runs in paragraphs:
        body.append("<w:p>" + "".join(_run(t, f) for t, f in runs) + "</w:p>")
    for text in styled_paragraphs or []:
        body.append(
            '<w:p><w:pPr><w:pStyle w:val="HindiBody"/></w:pPr>'
            f'<w:r><w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>'
        )
    document = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<w:document {W_NS}><w:body>{"".join(body)}</w:body></w:document>'

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _CONTENT_TYPES)
        z.writestr("_rels/.rels", _RELS)
        z.writestr("word/styles.xml", _STYLES)
        z.writestr("word/document.xml", document)
    return path


SAMPLE_LETTER = [
    [("dk;kZy; ftyk f'k{kk vf/kdkjh] jk;iqj", "Kruti Dev 010")],
    [("i= la[;k / F.No. ", "Kruti Dev 010"), ("DEO/RPR/2024/1187", "Times New Roman")],
    [("fnukad ", "Kruti Dev 010"), ("15.03.2024", "Times New Roman")],
    [("lsok esa]", "Kruti Dev 010")],
    [("leLr izkpk;Z] 'kkldh; mPprj ek/;fed fo|ky;", "Kruti Dev 010")],
    [("fo\"k;% ekfld leh{kk cSBd dh lwpuk A", "Kruti Dev 010")],
    [("egksn;]", "Kruti Dev 010")],
    [("mijksDr fo\"k; ds lanHkZ esa lwfpr fd;k tkrk gS fd fnukad ", "Kruti Dev 010"),
     ("25.03.2024", "Times New Roman"),
     (" dks ekfld leh{kk cSBd vk;ksftr dh tk jgh gS A", "Kruti Dev 010")],
    [("Hkonh;", "Kruti Dev 010")],
    [("(R. K. Sharma)", "Times New Roman")],
    [("ftyk f'k{kk vf/kdkjh", "Kruti Dev 010")],
    [("प्रतिलिपि: सूचनार्थ प्रेषित।", "Mangal")],
]

if __name__ == "__main__":
    import sys
    out = build(Path(sys.argv[1] if len(sys.argv) > 1 else "sample.docx"), SAMPLE_LETTER,
                styled_paragraphs=["vuqlj.k gsrq izsf\"kr A"])
    print(out)
