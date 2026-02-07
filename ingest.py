from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def ingest():
    loader = PyPDFLoader("data/Learning%20SQL%20Generate%2C%20Manipulate%2C%20%26%20Retrieve%20Data%203rd%20Ed.pdf")
    documents = loader.load()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(documents, embeddings)
    db.save_local("faiss_index")

    print("Documents indexed successfully")

if __name__ == "__main__":
    ingest()
