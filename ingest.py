from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter

def ingest():
    loader = PyPDFLoader("data/sample_docs.pdf")
    documents = loader.load()

    #Chunking 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    # metadata (source tracking)
    for chunk in chunks:
        chunk.metadata["source"] = "sample_docs.pdf"

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("faiss_index")

    print(f"Indexed {len(chunks)} chunks with metadata")

if __name__ == "__main__":
    ingest()
