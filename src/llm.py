import os
from dotenv import load_dotenv
import requests

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/api/generate"
MODEL = os.getenv("MODEL", "gemma4:31b-cloud")

def generate(prompt: str, model=MODEL):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

from deepeval.models import DeepEvalBaseLLM

class OllamaEvalModel(DeepEvalBaseLLM):
    def __init__(self, model_name=MODEL):
        self.model_name = model_name

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str) -> str:
        return generate(prompt, model=self.model_name)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name
