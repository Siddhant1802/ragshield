from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from defenses import DEFENSES
from rag import ask

app = FastAPI(title="RAGShield API")

# Allow the React app (which runs on port 5173) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    """What the frontend sends: the employee's question and which defense to use."""
    question: str
    defense: str = "full"


def build_trace(result: dict) -> list[dict]:
    """Turn ask()'s 'blocked' list into the 3-stage guardrail trace shown in the Chat screen."""
    blocked = result["blocked"]
    q = [b for b in blocked if b["source"] == "question"]
    out = [b for b in blocked if b["source"] == "answer"]
    docs = [b for b in blocked if b["source"] not in ("question", "answer")]

    if q:  # the question itself was blocked, so nothing else ran
        return [
            {"stage": "input", "status": "blocked", "details": [b["reason"] for b in q]},
            {"stage": "retrieval", "status": "skipped", "details": []},
            {"stage": "output", "status": "skipped", "details": []},
        ]
    return [
        {"stage": "input", "status": "passed", "details": []},
        {"stage": "retrieval", "status": "blocked" if docs else "passed",
         "details": [f"{b['source']} › {b['section']}: {b['reason']}" for b in docs]},
        {"stage": "output", "status": "modified" if out else "passed",
         "details": [b["reason"] for b in out]},
    ]


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/defenses")
def list_defenses():
    return {"defenses": list(DEFENSES), "default": "full"}


@app.post("/api/ask")
def ask_endpoint(req: AskRequest):
    if req.defense not in DEFENSES:
        raise HTTPException(status_code=400, detail=f"Unknown defense '{req.defense}'")
    result = ask(req.question, defense=DEFENSES[req.defense])
    return {
        "question": req.question,
        "defense": req.defense,
        "answer": result["answer"],
        "sources": result["sources"],
        "blocked": result["blocked"],
        "trace": build_trace(result),
    }