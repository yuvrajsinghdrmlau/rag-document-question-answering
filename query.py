from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

def ask_question(question):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = db.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    llm = pipeline("text-generation", model="google/flan-t5-base")

    prompt = f"""
    Answer the question using ONLY the context below.

    Context:
    {context}

    Question:
    {question}
    """

    response = llm(prompt, max_length=200)
    print(response[0]["generated_text"])

if __name__ == "__main__":
    ask_question("What is this document about?")