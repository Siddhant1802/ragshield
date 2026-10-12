# RAGShield

RAGShield protects a company's internal AI helpdesk from poisoned documents
that try to hijack its answers (indirect prompt injection), and measures which
defenses work best. This is my master's project, due Oct 31. I am learning LangChain.

## Stack
- Backend: Python, FastAPI, LangChain, Chroma
- LLM: Anthropic Claude Haiku via langchain-anthropic
- Embeddings: HuggingFace sentence-transformers (free, local)
- Frontend: React (Vite) + Tailwind
- Docker Compose

## How to work with me
- Build ONE small step at a time. Never build multiple features at once.
- Before writing code, explain the plan and wait for my approval.
- After writing code, explain what each new file and function does in simple terms.
- Tell me exactly how to run and test each step.
- Keep code simple and well commented. I need to understand and defend it.
- Write a test for each feature.
- Keep API costs low: use Claude Haiku and warn me before anything that makes many API calls.