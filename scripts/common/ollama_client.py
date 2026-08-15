import subprocess
import time

import ollama
from ollama import ChatResponse, chat

from .paths import REPO_ROOT

DEFAULT_MODEL = "muse-glimmer:latest"
DEFAULT_TEMPERATURE = 0.05
DEFAULT_MAX_TOKENS = 8192

# Hybrid-reasoning models read a literal "/no_think" in the prompt to skip
# emitting a hidden <think> block; on other models it is just text, so it is
# only prepended where it does something.
THINKING_MODELS = ("qwen3", "qwen3.5")

OLLAMA_LOG = REPO_ROOT / "ollama.log"


def build_system_prompt(model_id: str, base_prompt: str) -> str:
    if any(name in model_id.lower() for name in THINKING_MODELS):
        return "/no_think " + base_prompt
    return base_prompt


def ensure_server_running(retries: int = 30, delay: float = 1.0) -> None:
    try:
        ollama.list()
        return
    except Exception:
        pass

    print("Ollama server not reachable, starting `ollama serve` locally...")
    with open(OLLAMA_LOG, "a") as log_file:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    for _ in range(retries):
        time.sleep(delay)
        try:
            ollama.list()
            print("Ollama server is up.")
            return
        except Exception:
            continue
    raise RuntimeError(f"Could not reach an Ollama server after starting it. Check {OLLAMA_LOG}.")


def ensure_model_pulled(model_id: str) -> None:
    if model_id in {m.model for m in ollama.list().models}:
        return
    print(f"Pulling {model_id} via Ollama (not found locally, this may take a while)...")
    for progress in ollama.pull(model_id, stream=True):
        print(f"\r{progress.status}", end="", flush=True)
    print("\nModel pulled successfully.")


def call_model(
    model_id: str,
    messages: list[dict],
    temperature: float = DEFAULT_TEMPERATURE,
    num_predict: int = DEFAULT_MAX_TOKENS,
) -> str:
    response: ChatResponse = chat(
        model=model_id,
        messages=messages,
        # Without this, hybrid-reasoning models spend the whole num_predict
        # budget on hidden <think> tokens and return an empty answer.
        think=False,
        options={"temperature": temperature, "num_predict": num_predict},
    )
    return response["message"]["content"].strip()
