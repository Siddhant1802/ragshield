from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from vectorstore import load_vectorstore

load_dotenv()

MIN_RELEVANCE = 0.25   # chosen from Step 5: real answers scored 0.44+, off-topic under 0.06
TOP_K = 3              # how many chunks to retrieve
NO_ANSWER = "I don't know based on Acme's documents. Please contact HR or the IT Service Desk."

store = load_vectorstore()
llm = ChatAnthropic(model="claude-haiku-5-5")

# The prompt now has a {context} slot where the retrieved chunks go
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are Acme Corp's internal HR and IT helpdesk assistant. "
     "Answer the employee's question using ONLY the documents below. "
     "If the documents do not contain the answer, say you don't know. "
     "Keep answers short and mention which document you used.\n\n"
     "Documents:\n{context}"),
    ("human", "{question}"),
])
chain = prompt | llm | StrOutputParser()


def format_docs(docs):
    """Turn chunks into one block of text, each labelled with its source."""
    return "\n\n".join(
        f"[Source: {d.metadata['source']} | {d.metadata.get('h2', '-')}]\n{d.page_content}"
        for d in docs
    )


def ask(question: str, vector_store=None, defense=None) -> dict:
    vs = vector_store or store   # use the given store, or the clean Acme index

    # 1. RETRIEVE: find the closest chunks, with scores
    results = vs.similarity_search_with_relevance_scores(question, k=TOP_K)

    # 2. FILTER: keep only chunks that are relevant enough
    docs = [doc for doc, score in results if score >= MIN_RELEVANCE]

    # 3. DEFEND: let the defense remove suspicious chunks BEFORE Claude sees them
    blocked = []
    if defense is not None:
        docs, blocked = defense(docs)

    if not docs:
        return {"answer": NO_ANSWER, "sources": [], "blocked": blocked}

    # 4. GENERATE: Claude answers using only the chunks that survived
    answer = chain.invoke({"context": format_docs(docs), "question": question})

    sources = sorted({f"{d.metadata['source']} > {d.metadata.get('h2', '-')}" for d in docs})
    return {"answer": answer, "sources": sources, "blocked": blocked}

if __name__ == "__main__":
    questions = [
        "I forgot my login, what should I do?",
        "How many vacation days can I carry over?",
        "Can I expense alcohol at a client dinner?",
        "Who won the football World Cup in 1998?",
    ]
    for q in questions:
        result = ask(q)
        print("Q:", q)
        print("A:", result["answer"])
        print("Sources:", result["sources"])
        print("-" * 60)