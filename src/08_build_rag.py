import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Ensure API Key is set
if "GROQ_API_KEY" not in os.environ:
    print(" WARNING: GROQ_API_KEY environment variable not found.")
    exit()

print(" Step 1: Loading the Aviation Protocols...")
# Ensure aviation_protocols.txt is in the same directory as this script
loader = TextLoader("rag/aviation_protocols.txt")
documents = loader.load()

print(" Step 2: Slicing the document into searchable chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

print(" Step 3: Generating embeddings (using gemini-embedding-001)...")
# Verified model name from your system diagnostics
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_db = FAISS.from_documents(chunks, embeddings)

save_path = "./faiss_aviation_index"
vector_db.save_local(save_path)

print(f"\n SUCCESS! Vector Database saved to '{save_path}'.")