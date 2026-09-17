from langchain_text_splitters import RecursiveCharacterTextSplitter
import docx2txt

text = docx2txt.process("notes/Socket (1).docx")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

print("Number of chunks:", len(chunks))
print("\nFirst chunk:\n")
print(chunks[0])