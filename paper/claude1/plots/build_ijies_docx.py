#!/usr/bin/env python3
"""
claude1 — Re-render the full comprehensive paper into the IJIES template
(International Journal of Intelligent Engineering and Systems,
 "(Ver. 2025.5.18) IJIES_Format (2).docx").

Strategy: read the finished content (paragraphs / tables / inline figures) from
claude1_paper_full.docx *in document order* and re-emit it with IJIES styling:

  * A4, Times New Roman throughout, 0.75" L/R + 1.0" T/B margins
  * Title block full-width (1 column); body in 2 columns (continuous break)
  * Main title 14 pt bold centred; authors 11 pt; affiliation 10 pt
  * Abstract / Keywords 10 pt; first-order headings ("1. Introduction") 12 pt
    bold; second-order ("1.1 ...") 11 pt bold; body 11 pt justified
  * Figure captions 10 pt centred below; table titles 10 pt above; tables 8-9 pt

Run:  python3 paper/claude1/plots/build_ijies_docx.py
"""
import os
from io import BytesIO
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC  = os.path.join(ROOT, "paper", "claude1", "claude1_paper_full.docx")
TPL  = os.path.join(ROOT, "paper", "claude1", "(Ver. 2025.5.18) IJIES_Format (2).docx")
OUT  = os.path.join(ROOT, "paper", "claude1", "claude1_paper_IJIES.docx")

FONT = "Times New Roman"
COL_W = Inches(3.15)          # in-column image width

# ── helpers ───────────────────────────────────────────────────────────────
def set_cols(section, num, space=432):
    sectPr = section._sectPr
    cols = sectPr.find(qn('w:cols'))
    if cols is None:
        cols = OxmlElement('w:cols'); sectPr.append(cols)
    cols.set(qn('w:num'), str(num))
    cols.set(qn('w:space'), str(space))
    cols.set(qn('w:equalWidth'), "1")

def style_run(r, size, bold=False, italic=False):
    r.font.name = FONT
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    # ensure east-asian/complex also map to Times
    rPr = r._element.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts'); rPr.insert(0, rFonts)
    for a in ('w:ascii', 'w:hAnsi', 'w:cs'):
        rFonts.set(qn(a), FONT)

def para(doc, text="", size=11, bold=False, italic=False,
         align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=0, after=4, hanging=None):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.space_before = Pt(before); pf.space_after = Pt(after)
    pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
    if hanging is not None:
        pf.left_indent = Pt(hanging); pf.first_line_indent = Pt(-hanging)
    if text:
        style_run(p.add_run(text), size, bold, italic)
    return p

def set_cell_borders(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement('w:tcBorders')
    for edge in ('top', 'left', 'bottom', 'right'):
        e = OxmlElement(f'w:{edge}')
        e.set(qn('w:val'), 'single'); e.set(qn('w:sz'), '4')
        e.set(qn('w:space'), '0'); e.set(qn('w:color'), '000000')
        borders.append(e)
    tcPr.append(borders)

def add_table(doc, rows, ncols):
    fsize = 7.5 if ncols >= 7 else (8.5 if ncols >= 5 else 9)
    t = doc.add_table(rows=len(rows), cols=ncols)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    for ri, row in enumerate(rows):
        for ci in range(ncols):
            cell = t.cell(ri, ci)
            set_cell_borders(cell)
            cell.paragraphs[0].text = ""
            r = cell.paragraphs[0].add_run(row[ci] if ci < len(row) else "")
            style_run(r, fsize, bold=(ri == 0))
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cell.paragraphs[0].paragraph_format.space_after = Pt(0)
            cell.paragraphs[0].paragraph_format.space_before = Pt(0)
    return t

# ── read source in document order ─────────────────────────────────────────
from docx.text.paragraph import Paragraph
from docx.table import Table

src = Document(SRC)
src_part = src.part

def para_size(p):
    for r in p.runs:
        if r.font.size:
            return r.font.size.pt
    return None

def para_images(p):
    """Return list of image blobs embedded in this paragraph, in order."""
    blobs = []
    for blip in p._element.findall('.//' + qn('a:blip')):
        rId = blip.get(qn('r:embed'))
        if rId:
            blobs.append(src_part.related_parts[rId].blob)
    return blobs

# ── build IJIES document ──────────────────────────────────────────────────
doc = Document(TPL)
# wipe template body content (keep styles + section geometry)
for el in list(doc.element.body):
    if el.tag in (qn('w:p'), qn('w:tbl')):
        doc.element.body.remove(el)

set_cols(doc.sections[0], 1)          # title block: single column
body_started = False

for child in src.element.body.iterchildren():
    # ---- table ----
    if child.tag == qn('w:tbl'):
        tb = Table(child, src)
        rows = [[c.text.strip() for c in row.cells] for row in tb.rows]
        add_table(doc, rows, len(tb.columns))
        para(doc, "", after=2)
        continue
    if child.tag != qn('w:p'):
        continue

    p = Paragraph(child, src)
    text = p.text.strip()
    sname = p.style.name
    imgs = para_images(p)

    # ---- inline figure(s) ----
    if imgs:
        for blob in imgs:
            ip = doc.add_paragraph()
            ip.alignment = WD_ALIGN_PARAGRAPH.CENTER
            ip.paragraph_format.space_before = Pt(4)
            ip.paragraph_format.space_after = Pt(2)
            ip.add_run().add_picture(BytesIO(blob), width=COL_W)
        continue

    if not text:
        continue

    # ---- title block (single column, before first H1) ----
    if sname == 'Title':
        para(doc, text, size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
             before=6, after=4)
        continue
    if sname == 'Subtitle':
        para(doc, text, size=11, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER,
             after=6)
        continue
    if sname == 'Author':
        para(doc, text, size=11, align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
        continue
    if sname == 'Date':
        para(doc, text, size=10, align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
        # corresponding-author placeholder line (IJIES expects an email)
        para(doc, "* Corresponding author's Email: research@rbru.ac.th",
             size=10, align=WD_ALIGN_PARAGRAPH.CENTER, after=8)
        continue
    if sname == 'Abstract Title':
        continue                       # merged into Abstract body below
    if sname == 'Abstract':
        ap = doc.add_paragraph()
        ap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        ap.paragraph_format.space_after = Pt(6)
        style_run(ap.add_run("Abstract: "), 10, bold=True)
        style_run(ap.add_run(text), 10)
        # Keywords line (IJIES-required), derived from the paper topic
        kp = doc.add_paragraph()
        kp.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        kp.paragraph_format.space_after = Pt(6)
        style_run(kp.add_run("Keywords: "), 10, bold=True)
        style_run(kp.add_run(
            "Small language models, Intent-driven networking, 6G network "
            "slicing, Post-training quantization, Edge inference, "
            "Model benchmarking."), 10)
        continue

    # ---- first H1 → switch to two-column body ----
    is_h1 = (sname == 'Normal' and para_size(p) and para_size(p) >= 15)
    if is_h1 and not body_started:
        sec = doc.add_section(WD_SECTION.CONTINUOUS)
        for m in ('left_margin', 'right_margin', 'top_margin', 'bottom_margin'):
            setattr(sec, m, getattr(doc.sections[0], m))
        set_cols(sec, 2, space=288)
        body_started = True

    # ---- headings ----
    if is_h1:
        para(doc, text, size=12, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT,
             before=8, after=4)
        continue
    if sname == 'Normal' and para_size(p) and 12 <= para_size(p) < 15:
        para(doc, text, size=11, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT,
             before=6, after=3)
        continue

    # ---- figure/table captions ----
    if sname == 'Caption':
        cp = para(doc, text.replace("Figure ", "Figure. "), size=10,
                  align=WD_ALIGN_PARAGRAPH.CENTER, before=2, after=8)
        continue
    if sname == 'Body Text' and (text.startswith('Table ')):
        para(doc, text, size=10, align=WD_ALIGN_PARAGRAPH.CENTER,
             before=6, after=2)
        continue

    # ---- bullets ----
    if text.startswith('•'):
        para(doc, text, size=11, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
             after=2, hanging=12)
        continue

    # ---- normal body ----
    para(doc, text, size=11, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=4)

# ── References section (source paper has no bibliography yet) ──────────────
para(doc, "References", size=12, bold=True, before=10, after=4)
para(doc, "[Bibliography to be added — the source manuscript does not yet "
     "contain numbered references. Add IJIES-style (modified IEEE) entries "
     "in 11-point Times New Roman here.]", size=11, italic=True)

doc.save(OUT)
print("saved:", OUT)
print("sections:", len(doc.sections),
      "cols:", [s._sectPr.find(qn('w:cols')).get(qn('w:num'))
                if s._sectPr.find(qn('w:cols')) is not None else '1'
                for s in doc.sections])
print("paragraphs:", len(doc.paragraphs), "tables:", len(doc.tables))
