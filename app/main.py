from pypdf import PdfReader

# Path to the PDF
pdf_path = "data/pdfs/Notes.pdf"

# Open the PDF
reader = PdfReader(pdf_path)

# Print number of pages
print(f"Total Pages: {len(reader.pages)}")

# Read each page
for page in reader.pages:
    text = page.extract_text()
    print(text)