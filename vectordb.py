import faiss
from sentence_transformers import SentenceTransformer
import docx2txt
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Read the document
text = docx2txt.process("notes/Socket (1).docx")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

# Create embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

print("Number of chunks:", len(chunks))
print("FAISS vectors:", index.ntotal)