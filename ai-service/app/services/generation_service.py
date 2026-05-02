import requests


class GenerationService:
    def __init__(self, ollama_url: str = "http://localhost:11434/api/generate", model: str = "llama3"):
        self.ollama_url = ollama_url
        self.model = model

    def generate(self, prompt: str) -> str:
        response = requests.post(
            self.ollama_url,
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=30
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    def generate_answer(self, query: str, docs: list[str]) -> str:
        context = "\n".join(docs)

        prompt = f"""
You are a precise AI assistant.

Rules:
- Use ONLY the provided context
- If answer is not clearly present, say "I don't know"
- Do NOT hallucinate

Context:
{context}

Question:
{query}

Answer:
"""
        answer = self.generate(prompt)
        return answer if answer else "I don't know."