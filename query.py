from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

def ask_question(question, k=4):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = db.similarity_search(question, k=k)

    context = "\n\n".join(
        [f"[Source: {doc.metadata.get('source')}]\n{doc.page_content}" for doc in docs]
    )

    llm = pipeline("text-generation", model="google/flan-t5-base")

    prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm(prompt, max_length=250)
    print(response[0]["generated_text"])

if __name__ == "__main__":
    ask_question("What is the main topic of this document?", k=4)
