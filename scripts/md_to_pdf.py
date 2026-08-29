"""Render arxiv-preprint.md to a simple multi-page PDF (no TeX required)."""
from pathlib import Path

from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib import colors

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "arxiv-preprint.md"
OUT = ROOT / "latex" / "main.pdf"


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def md_inline(text: str) -> str:
    text = _esc(text)
    # **bold**
    parts = text.split("**")
    out = []
    for i, p in enumerate(parts):
        out.append(f"<b>{p}</b>" if i % 2 else p)
    text = "".join(out)
    # `code`
    parts = text.split("`")
    out = []
    for i, p in enumerate(parts):
        out.append(f'<font face="Courier" size="9">{p}</font>' if i % 2 else p)
    return "".join(out)


def build():
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "T",
        parent=styles["Title"],
        fontSize=13,
        leading=16,
        spaceAfter=8,
        alignment=TA_CENTER,
    )
    meta = ParagraphStyle(
        "M",
        parent=styles["Normal"],
        fontSize=10,
        leading=13,
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    h1 = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontSize=12,
        spaceBefore=12,
        spaceAfter=6,
    )
    h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontSize=11,
        spaceBefore=8,
        spaceAfter=4,
    )
    body = ParagraphStyle(
        "B",
        parent=styles["Normal"],
        fontSize=10,
        leading=13,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    code = ParagraphStyle(
        "C",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=8,
        leading=10,
        leftIndent=12,
        spaceAfter=8,
    )
    story = []
    lines = MD.read_text(encoding="utf-8").splitlines()
    i = 0
    in_code = False
    code_buf = []
    table_buf = []

    def flush_table():
        nonlocal table_buf
        if not table_buf:
            return
        rows = []
        for raw in table_buf:
            cells = [c.strip() for c in raw.strip().strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            rows.append([Paragraph(md_inline(c), body) for c in cells])
        if rows:
            t = Table(rows, hAlign="LEFT")
            t.setStyle(
                TableStyle(
                    [
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.92, 0.92, 0.92)),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 3),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )
            story.append(t)
            story.append(Spacer(1, 8))
        table_buf = []

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            if in_code:
                story.append(Preformatted("\n".join(code_buf), code))
                code_buf = []
                in_code = False
            else:
                flush_table()
                in_code = True
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue
        if line.startswith("|"):
            table_buf.append(line)
            i += 1
            continue
        flush_table()
        if not line.strip() or line.strip() == "---":
            i += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(md_inline(line[2:]), title))
        elif line.startswith("## "):
            story.append(Paragraph(md_inline(line[3:]), h1))
        elif line.startswith("### "):
            story.append(Paragraph(md_inline(line[4:]), h2))
        elif line.startswith("> "):
            story.append(Paragraph("<i>" + md_inline(line[2:]) + "</i>", body))
        elif line.startswith("- ") or line.startswith("1. "):
            items = []
            while i < len(lines) and (
                lines[i].startswith("- ")
                or (len(lines[i]) > 2 and lines[i][0].isdigit() and lines[i][1:3] == ". ")
            ):
                txt = lines[i].lstrip("- ")
                if txt[:3].endswith(". "):
                    txt = txt[3:] if txt[1] == "." else txt
                # numbered
                if lines[i][0].isdigit() and ". " in lines[i][:4]:
                    txt = lines[i].split(". ", 1)[1]
                items.append(ListItem(Paragraph(md_inline(txt), body)))
                i += 1
            story.append(ListFlowable(items, bulletType="bullet", leftIndent=18))
            continue
        elif line.startswith("**Author:**") or line.startswith("**Affiliation:**") or line.startswith("**Correspondence:**"):
            story.append(Paragraph(md_inline(line), meta))
        else:
            story.append(Paragraph(md_inline(line), body))
        i += 1
    flush_table()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=letter,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="How Realistic Are Successful Evasion Attacks?",
        author="Sathyaraj Kolandasamy",
    )
    doc.build(story)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
