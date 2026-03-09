import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

def ask_llm(prompt, model="phi3"):

    response = requests.post(
        OLLAMA_URL,
        json={
    "model": "phi3",
    "prompt": prompt,
    "stream": False,
    "options": {
        "num_predict": 120
    }
}
    )

    return response.json()["response"]