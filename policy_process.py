import glob
import os

from docling.document_converter import DocumentConverter

DOCS_DIR = os.path.join("data", "docs")


def convert_all() -> list:
    """Batch-converts every PDF in data/docs/ to a same-named Markdown file.
    This is a one-time, offline preprocessing step -- it is not on the live
    retrieval path. Re-run it whenever a source PDF changes, then re-run
    rag/ingest.py to rebuild the vector index from the new Markdown."""
    pdf_paths = sorted(glob.glob(os.path.join(DOCS_DIR, "*.pdf")))
    if not pdf_paths:
        print(f"No PDFs found in {DOCS_DIR}. Run data_gen.py first.")
        return []

    converter = DocumentConverter()
    converted = []
    for pdf_path in pdf_paths:
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        md_path = os.path.join(DOCS_DIR, f"{base}.md")
        print(f"Processing {pdf_path} -> {md_path}...")
        result = converter.convert(pdf_path)
        markdown_content = result.document.export_to_markdown()
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        converted.append(md_path)

    print(f"Conversion complete. {len(converted)} document(s) processed.")
    return converted


if __name__ == "__main__":
    os.makedirs(DOCS_DIR, exist_ok=True)
    convert_all()
