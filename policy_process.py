import os
from docling.document_converter import DocumentConverter

def convert_policy_to_markdown():
    # Define paths
    pdf_path = os.path.join("data", "docs", "Corporate_Policy_2025.pdf")
    md_path = os.path.join("data", "docs", "policy.md")
    
    print(f"📄 Processing {pdf_path}...")
    
    # 1. Initialize Docling Converter
    converter = DocumentConverter()
    
    # 2. Convert PDF
    result = converter.convert(pdf_path)
    
    # 3. Export to Markdown
    # This turns complex tables and headers into clean Markdown text
    markdown_content = result.document.export_to_markdown()
    
    # 4. Save the Markdown file
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    
    print(f"Conversion Complete! Saved to {md_path}")
    print("   (The Agent will now read this Markdown file instead of the PDF)")

if __name__ == "__main__":
    # Ensure the directory exists
    if not os.path.exists(os.path.join("data", "docs")):
        os.makedirs(os.path.join("data", "docs"))
        print("⚠️ Warning: Created data/docs folder. Please put your 'policy.pdf' there!")
    else:
        convert_policy_to_markdown()