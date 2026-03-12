import os
import requests
import json

# Try different Ollama URLs for Render compatibility
OLLAMA_URLS = [
    "http://localhost:11434/api/generate",
    "http://127.0.0.1:11434/api/generate",
    "http://0.0.0.0:11434/api/generate"
]

def ask_llm(prompt, model="phi3", timeout=30):
    """
    Safe LLM call with error handling and multiple URL fallbacks
    """
    for ollama_url in OLLAMA_URLS:
        try:
            response = requests.post(
                ollama_url,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 120,
                        "temperature": 0.7
                    }
                },
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if "response" in result:
                    return result["response"]
                else:
                    print(f"Unexpected response format: {result}")
                    continue
            else:
                print(f"HTTP {response.status_code} from {ollama_url}")
                continue
                
        except requests.exceptions.Timeout:
            print(f"Timeout from {ollama_url}")
            continue
        except requests.exceptions.ConnectionError:
            print(f"Connection failed to {ollama_url}")
            continue
        except Exception as e:
            print(f"Error calling {ollama_url}: {e}")
            continue
    
    # Fallback response if all URLs fail
    return "AI advice service temporarily unavailable. Please try again later."