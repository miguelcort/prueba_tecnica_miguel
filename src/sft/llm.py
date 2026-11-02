"""Optional LLM generation behind a provider-independent interface.

The default is offline. No request leaves this machine unless the configuration enables it and
LLM_MODE=api, LLM_BASE_URL and LLM_MODEL are set. Output is untrusted: it only becomes a candidate,
and the grounding check decides. The credential is read from the environment, never from a file.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Callable, Dict, List, Protocol

from .grounding import check_grounding

PROMPT = (
    "Using ONLY the documentation fragment below, write one technical question that names the service "
    "{project} and its answer. Do not add information. Reply with JSON only: "
    '{{"question": "...", "answer": "...", "category": "..."}}.\n\nFRAGMENT:\n{text}'
)


class LLMProvider(Protocol):
    def complete(self, prompt: str) -> str:
        ...


class OfflineProvider:
    def complete(self, prompt: str) -> str:
        raise RuntimeError("LLM generation is offline: set LLM_MODE=api, LLM_BASE_URL and LLM_MODEL to enable it")


class OpenAICompatibleProvider:
    """Any chat-completions endpoint, chosen entirely through environment variables."""

    def __init__(self, base_url: str, model: str, api_key: str = ""):
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key

    def complete(self, prompt: str) -> str:
        body = json.dumps({"model": self.model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)["choices"][0]["message"]["content"]


def provider_from_env(env: Dict[str, str] = None) -> LLMProvider:
    env = os.environ if env is None else env
    if env.get("LLM_MODE") == "api" and env.get("LLM_BASE_URL") and env.get("LLM_MODEL"):
        return OpenAICompatibleProvider(env["LLM_BASE_URL"], env["LLM_MODEL"], env.get("LLM_API_KEY", ""))
    return OfflineProvider()


def complete_with_retries(provider: LLMProvider, prompt: str, max_retries: int, sleep: Callable[[float], None] = time.sleep) -> str:
    for attempt in range(max_retries):
        try:
            return provider.complete(prompt)
        except (OSError, ValueError, KeyError):
            if attempt == max_retries - 1:
                raise
            sleep(2**attempt)
    raise RuntimeError("max_retries must be at least 1")


def generate_candidates(chunks: List[dict], provider: LLMProvider, settings: dict, min_overlap: float, sleep: Callable[[float], None] = time.sleep) -> List[dict]:
    """Ask for one pair per chunk. A candidate that fails grounding never reaches training."""
    candidates = []
    for chunk in chunks:
        raw = complete_with_retries(provider, PROMPT.format(project=chunk["project"], text=chunk["text"]), settings["max_retries"], sleep)
        try:
            pair = json.loads(raw)
            question, answer = pair["question"], pair["answer"]
        except (ValueError, KeyError, TypeError):
            candidates.append({"chunk_id": chunk["chunk_id"], "validation_status": "rejected", "reasons": ["malformed_response"]})
            continue
        result = check_grounding(answer, chunk["text"], [], chunk["project"], min_overlap)
        candidates.append({"chunk_id": chunk["chunk_id"], "question": question, "answer": answer, "category": pair.get("category"), "generation_method": "llm", "validation_status": result["status"], "reasons": result["reasons"]})
    return candidates
