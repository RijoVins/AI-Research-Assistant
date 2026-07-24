from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Read PDF
reader = PdfReader("data/pdfs/Notes.pdf")

full_text = ""

for page in reader.pages:
    text = page.extract_text()
    if text:
        full_text += text + "\n"

# Smart Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_text(full_text)

print(f"Total Chunks: {len(chunks)}")

for i in range(3):
    print("\n====================")
    print(f"Chunk {i+1}")
    print("====================")
    print(chunks[i][:300])