from pypdf import PdfReader

pdf_path = "data/pdfs/Notes.pdf"

reader = PdfReader(pdf_path)

full_text = ""

for page in reader.pages:

    text = page.extract_text()

    if text:
        full_text += text + "\n"


# ------------------------
# Chunking
# ------------------------

chunk_size = 1000

chunks = []

for i in range(0, len(full_text), chunk_size):
    chunk = full_text[i:i + chunk_size]
    chunks.append(chunk)

print(f"Total Chunks: {len(chunks)}")

print("\nFirst Chunk:\n")
print(chunks[0])




print(f"Total Chunks: {len(chunks)}")

for i in range(3):
    print("\n----------------------")
    print(f"Chunk {i+1}")
    print("----------------------")
    print(chunks[i][:200])