# -*- coding: utf-8 -*-
"""把 manuscript_jvcir.md 转成 Elsevier 投稿用 DOCX（基础排版）。"""
import os
import re

from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "manuscript_jvcir.md")
OUT = os.path.join(HERE, "manuscript_jvcir.docx")


def add_runs(p, text):
    for part in re.split(r"(\*\*.+?\*\*)", text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            r = p.add_run(part[2:-2]); r.bold = True
        else:
            p.add_run(part)


def main():
    lines = open(SRC, encoding="utf-8").read().splitlines()
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(11)

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line or line == "---":
            i += 1
            continue
        if line.startswith("# "):
            p = doc.add_heading(line[2:], level=0)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            i += 1
            continue
        if line.startswith("## "):
            doc.add_heading(line[3:], level=1); i += 1; continue
        if line.startswith("### "):
            doc.add_heading(line[4:], level=2); i += 1; continue
        if line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i]); i += 1
            if len(block) >= 2:
                rows = [[c.strip() for c in r.strip("|").split("|")] for r in block]
                rows = [r for r in rows if not all(set(c) <= set("-: ") for c in r)]
                if rows:
                    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
                    t.style = "Table Grid"
                    for r, row in enumerate(rows):
                        for c, val in enumerate(row[:len(rows[0])]):
                            t.cell(r, c).text = val
                            for par in t.cell(r, c).paragraphs:
                                for run in par.runs:
                                    run.font.name = "Times New Roman"
                                    run.font.size = Pt(9)
                                    run.bold = (r == 0)
            continue
        p = doc.add_paragraph()
        add_runs(p, line)
        i += 1

    doc.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
