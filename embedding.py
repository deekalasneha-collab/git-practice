from sentence_transformers import SentenceTransformer
import docx2txt
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = docx2txt.process("notes/Socket (1).docx")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)