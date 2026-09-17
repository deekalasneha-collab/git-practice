import faiss
from sentence_transformers import SentenceTransformer
import docx2txt
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load API key from .env
load_dotenv()

# Connect to OpenAI
llm = ChatOpenAI(
    model="gpt-5.6-luna",
    temperature=0
)

# Read document
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

# Ask the user a question
question = input("Ask your question: ")

# Convert question into an embedding
question_embedding = model.encode([question])

# Search for 3 relevant chunks
distances, indices = index.search(question_embedding, 3)

# Combine the relevant chunks
context = ""

for i in indices[0]:
    context += chunks[i] + "\n"

# Create prompt for OpenAI
prompt = f"""
Answer the question using only the following college notes.

Notes:
{context}

Question:
{question}
"""

# Send the question + notes to OpenAI
response = llm.invoke(prompt)

# Display the final answer
print("\nAnswer:")
print(response.content)