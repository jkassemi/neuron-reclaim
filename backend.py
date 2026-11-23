from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class OpenAICompatBackend:
    """
    Zero-dependency OpenAI-compatible HTTP client for vLLM, llama.cpp (llama-server),
    SGLang, Ollama, or LM Studio.

    Powers:
      - Live Chat (`POST /v1/chat/completions`)
      - Next-Token Probability Probes (`POST /v1/completions` with `logprobs=top_k`)
    """

    api_base: str
    model_name: str | None = None
    api_key: str = "EMPTY"
    timeout_s: float = 60.0

    def __post_init__(self) -> None:
        self.api_base = self.api_base.rstrip("/")
        if not self.api_base.endswith("/v1"):
            self.api_base = self.api_base + "/v1"

    def _request(self, path: str, payload: dict | None = None) -> dict:
        url = f"{self.api_base}{path}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"API {exc.code} at {url}: {body[:200]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach {url}: {exc.reason}") from exc

    def resolve_model_name(self) -> str:
        if self.model_name:
            return self.model_name
        resp = self._request("/models")
        models = resp.get("data", [])
        if not models:
            raise RuntimeError(f"No models reported by {self.api_base}/models; pass --api-model explicitly.")
        self.model_name = str(models[0]["id"])
        return self.model_name

    def chat_once(self, user_text: str, max_new_tokens: int = 64) -> str:
        model_id = self.resolve_model_name()
        resp = self._request(
            "/chat/completions",
            {
                "model": model_id,
                "messages": [{"role": "user", "content": user_text}],
                "max_tokens": max_new_tokens,
                "temperature": 0.0,
            },
        )
        choices = resp.get("choices", [])
        if not choices:
            return ""
        return str(choices[0].get("message", {}).get("content", "")).strip()

    def probe_next_tokens(self, prompt: str, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Query top-K next-token probabilities via `/v1/completions` (supported by vLLM,
        SGLang, and llama-server). Falls back gracefully if a server returns llama.cpp
        `probs` format instead of OpenAI `top_logprobs`.
        """
        model_id = self.resolve_model_name()
        resp = self._request(
            "/completions",
            {
                "model": model_id,
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": top_k,
            },
        )
        choices = resp.get("choices", [])
        if not choices:
            return []
        lp_obj = choices[0].get("logprobs") or {}

        # Standard OpenAI / vLLM / SGLang format: top_logprobs = [{tok: logprob, ...}]
        top_lps = lp_obj.get("top_logprobs")
        if isinstance(top_lps, list) and top_lps and isinstance(top_lps[0], dict):
            items = [(tok, round(math.exp(float(lp)) * 100.0, 1)) for tok, lp in top_lps[0].items()]
            items.sort(key=lambda kv: -kv[1])
            return items[:top_k]

        # llama.cpp alternate format: content = [{"top_logprobs": [{"token": ..., "logprob": ...}]}]
        content = lp_obj.get("content")
        if isinstance(content, list) and content and isinstance(content[0], dict):
            tlist = content[0].get("top_logprobs", [])
            items = [
                (str(e["token"]), round(math.exp(float(e["logprob"])) * 100.0, 1))
                for e in tlist
                if "token" in e and "logprob" in e
            ]
            items.sort(key=lambda kv: -kv[1])
            return items[:top_k]

        # Fallback: single chosen token
        tok_txt = str(choices[0].get("text", ""))
        return [(tok_txt, 100.0)] if tok_txt else []
