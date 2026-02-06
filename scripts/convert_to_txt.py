#!/usr/bin/env python3
"""Convert PDFs and PPTX files in a subject folder to .txt files in documents/txt/."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import fitz  # PyMuPDF
from pptx import Presentation


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n\n".join(pages)


def extract_text_from_pptx(pptx_path: Path) -> str:
    """Extract text from a PowerPoint file."""
    prs = Presentation(pptx_path)
    slides_text = []
    for i, slide in enumerate(prs.slides, 1):
        parts = [f"--- Slide {i} ---"]
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    text = paragraph.text.strip()
                    if text:
                        parts.append(text)
        slides_text.append("\n".join(parts))
    return "\n\n".join(slides_text)


def main():
    subject_dir = project_root / "data" / "subjects" / "TIØ4165"
    txt_dir = subject_dir / "documents" / "txt"
    txt_dir.mkdir(parents=True, exist_ok=True)

    # Collect all PDFs and PPTX from the top-level subject directory
    files = sorted(
        f for f in subject_dir.iterdir()
        if f.is_file() and f.suffix.lower() in (".pdf", ".pptx")
    )

    if not files:
        print("No PDF/PPTX files found in", subject_dir)
        return

    for file_path in files:
        stem = file_path.stem
        txt_path = txt_dir / f"{stem}.txt"

        if txt_path.exists():
            print(f"⏭️  Already exists: {txt_path.name}")
            continue

        ext = file_path.suffix.lower()
        print(f"📄 Converting: {file_path.name} ...", end=" ", flush=True)

        try:
            if ext == ".pdf":
                text = extract_text_from_pdf(file_path)
            elif ext == ".pptx":
                text = extract_text_from_pptx(file_path)
            else:
                continue

            if not text.strip():
                print("⚠️  No text extracted!")
                continue

            txt_path.write_text(text, encoding="utf-8")
            word_count = len(text.split())
            print(f"✅ {word_count:,} words → {txt_path.name}")
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\nDone! All txt files in:", txt_dir)


if __name__ == "__main__":
    main()
