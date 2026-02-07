from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def ingest():
    loader = PyPDFLoader("data/sample_docs.pdf")
    documents = loader.load()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(documents, embeddings)
    db.save_local("faiss_index")

    print("Documents indexed successfully")

if __name__ == "__main__":
    ingest()
