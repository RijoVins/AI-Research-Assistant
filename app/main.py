from pypdf import PdfReader

pdf_path = "data/pdfs/Notes.pdf"

reader = PdfReader(pdf_path)

print(f"Total Pages: {len(reader.pages)}")

# Store all text
full_text = ""

for page_number, page in enumerate(reader.pages, start=1):

    print(f"Reading Page {page_number}...")

    text = page.extract_text()

    if text:
        full_text += text + "\n"

print("\nExtraction Completed!")

print(f"Total Characters: {len(full_text)}")