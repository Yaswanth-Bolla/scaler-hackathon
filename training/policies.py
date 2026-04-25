"""
Policy abstractions — used by the baseline runner, ablation runners, and the
curriculum driver to drive the environment without committing to any single
RL framework.

A `Policy` is a callable: (observation_dict, phase) -> action_dict.

Concrete policies provided here (no GPU/training required):
  - `NullPhase1Policy`           : refuses to do P1 — emits transition immediately
                                   with empty belief.  Used by Pool-B baseline.
  - `RandomPhase2Policy`         : P2 only — lists root, reads top files,
                                   proposes empty patch.  Lower-bound baseline.
  - `OracleP1ThenRandomP2Policy` : Combines synthetic oracle belief with
                                   RandomPhase2Policy — the *upper* baseline
                                   for "how much does perfect P1 help a
                                   weak P2 agent?"
  - `OpenAIChatPolicy`           : Adapter around the OpenAI Chat Completions
                                   API (or any compatible endpoint set via
                                   `base_url`).  This is the policy you
                                   actually want in production runs.

The runners only depend on the abstract `Policy` protocol — you can plug in a
vLLM-served Qwen, an HF-Transformers model, or an Unsloth model behind the
same interface by writing a 50-line adapter.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Protocol

from ..models import ActionType, BeliefState
from ..pools import oracle_belief
from ..tasks import get_scenario


# ──────────────────────────────────────────────────────────────────────
# Protocol
# ──────────────────────────────────────────────────────────────────────


class Policy(Protocol):
    """Anything that maps an observation to an action dict."""

    def __call__(
        self,
        observation: Dict[str, Any],
        phase:       int,
        task_name:   str,
    ) -> Dict[str, Any]:
        ...


# ──────────────────────────────────────────────────────────────────────
# Built-in policies
# ──────────────────────────────────────────────────────────────────────


class NullPhase1Policy:
    """
    Refuses to investigate — emits `transition_to_phase2` immediately
    with an *empty* BeliefState.  This is the canonical "no Phase-1
    context" baseline used to estimate `null_context_p2_score`.
    """

    def __call__(self, observation, phase, task_name):
        if phase == 1:
            empty = BeliefState()
            return {
                "action_type":    ActionType.TRANSITION_TO_PHASE2.value,
                "target_service": None,
                "parameters":     {"belief": asdict(empty)},
            }
        # Should never hit this — Pool B auto-transitions during reset
        return {"action_type": ActionType.DECLARE_NO_CHANGE.value, "parameters": {}}


class RandomPhase2Policy:
    """
    Deterministic 'cheap' P2 explorer:

        1. list_dir(.)              — sees top-level structure
        2. read_file(<first .py>)   — random pick of one file
        3. get_git_log()            — sees last 5 commits
        4. get_file_diff(bad_sha)   — pulls the bad commit's diff
        5. propose_patch(diff)      — re-emits the diff (often correct on
                                      easy bugs, very wrong on hard ones)

    This is intentionally a *weak* policy.  It exercises the full P2 action
    set, gives a non-degenerate baseline score, and is useful as the
    reference policy for ablations (the LLM should beat it by a wide margin).
    """

    def __init__(self, seed: int = 0):
        self._rng    = random.Random(seed)
        self._step   = 0
        self._files: List[str] = []
        self._target: Optional[str] = None

    def reset(self) -> None:
        self._step = 0
        self._files = []
        self._target = None

    def __call__(self, observation, phase, task_name):
        if phase != 2:
            # Should not be called in P1 — Pool B auto-transitions
            return {"action_type": ActionType.DECLARE_NO_CHANGE.value, "parameters": {}}

        self._step += 1
        action_result = observation.get("action_result", {}) or {}

        if self._step == 1:
            return {"action_type": ActionType.LIST_DIR.value,
                    "parameters": {"path": "."}}

        if self._step == 2:
            entries = action_result.get("entries", []) or []
            files = [e["name"] for e in entries if e.get("type") == "file"]
            if files:
                self._target = self._rng.choice(files)
                return {"action_type": ActionType.READ_FILE.value,
                        "parameters": {"path": self._target}}
            # Fallback: try git log
            return {"action_type": ActionType.GET_GIT_LOG.value,
                    "parameters": {"n_commits": 5}}

        if self._step == 3:
            return {"action_type": ActionType.GET_GIT_LOG.value,
                    "parameters": {"n_commits": 5}}

        if self._step == 4:
            sha = observation.get("bad_commit_sha")
            if sha:
                return {"action_type": ActionType.GET_FILE_DIFF.value,
                        "parameters": {"commit_sha": sha}}
            return {"action_type": ActionType.PROPOSE_PATCH.value,
                    "parameters": {"diff": ""}}

        # Step 5+: propose whatever diff we have, otherwise empty.
        diff = action_result.get("diff", "") or ""
        return {"action_type": ActionType.PROPOSE_PATCH.value,
                "parameters": {"diff": diff}}


class OracleP1ThenRandomP2Policy(RandomPhase2Policy):
    """
    Pool-A/C control policy: synthesises the perfect Phase-1 belief from
    the scenario's static config, transitions immediately, then runs the
    cheap P2 explorer.

    Used as the "ceiling" Phase-1 baseline (no P1 mistakes possible) so we
    can isolate Phase-2 quality from Phase-1 quality during diagnostic runs.
    """

    def __call__(self, observation, phase, task_name):
        if phase == 1:
            scenario = get_scenario(task_name)
            return {
                "action_type":    ActionType.TRANSITION_TO_PHASE2.value,
                "target_service": None,
                "parameters":     {"belief": asdict(oracle_belief(scenario))},
            }
        return super().__call__(observation, phase, task_name)


# ──────────────────────────────────────────────────────────────────────
# OpenAI / OpenAI-compatible chat policy
# ──────────────────────────────────────────────────────────────────────


_OPENAI_SYSTEM_PROMPT = (
    "You are an SRE incident responder.  At each step you receive a JSON "
    "observation describing the current incident state and the list of "
    "valid_actions.  Reply with a single JSON object:\n"
    '  {"action_type": "<one of valid_actions>",\n'
    '   "target_service": "<service or null>",\n'
    '   "parameters": {...}}\n'
    "Take the most informative action.  When you have enough evidence to "
    "diagnose the runtime fault, emit `transition_to_phase2` with a "
    "structured `belief` parameter.  In Phase 2 you read code, search, "
    "and finally `propose_patch` (or `declare_no_change` for spurious "
    "issues)."
)


class OpenAIChatPolicy:
    """
    Thin wrapper around an OpenAI-compatible chat-completions endpoint.

    Construct with model name; reads OPENAI_API_KEY (and optionally
    OPENAI_BASE_URL) from env.  Maintains a short rolling chat history.

    NOTE: This is the production-style policy.  The runners can use it as
    a drop-in for any other Policy.  It is *not* required for any of the
    other modules to work — the baseline runner and ablations all default
    to a deterministic policy.
    """

    def __init__(
        self,
        model:           str = "gpt-4o-mini",
        max_history:     int = 8,
        temperature:     float = 0.2,
        base_url:        Optional[str] = None,
    ):
        try:
            from openai import OpenAI       # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "openai package not installed. `pip install openai`") from e
        self._client = OpenAI(
            api_key  = os.environ.get("OPENAI_API_KEY", "_unset_"),
            base_url = base_url or os.environ.get("OPENAI_BASE_URL"),
        )
        self._model       = model
        self._max_history = max_history
        self._temperature = temperature
        self._messages: List[Dict[str, str]] = []
        self._task: Optional[str] = None

    def reset(self, task_name: str) -> None:
        self._task = task_name
        self._messages = [{"role": "system", "content": _OPENAI_SYSTEM_PROMPT}]

    def __call__(self, observation, phase, task_name):
        if self._task != task_name or not self._messages:
            self.reset(task_name)
        user_payload = {
            "phase":         phase,
            "observation":   observation,
            "valid_actions": observation.get("valid_actions", []),
        }
        self._messages.append({"role": "user",
                               "content": json.dumps(user_payload)[:6000]})
        try:
            resp = self._client.chat.completions.create(
                model       = self._model,
                messages    = self._messages,
                temperature = self._temperature,
                max_tokens  = 600,
            )
            text = resp.choices[0].message.content or "{}"
            self._messages.append({"role": "assistant", "content": text})
            self._trim_history()
            return _parse_action_json(text)
        except Exception as e:
            # On any API failure, do something safe rather than crash
            return {"action_type": ActionType.DECLARE_NO_CHANGE.value,
                    "parameters": {"reason": f"policy error: {e}"}}

    def _trim_history(self) -> None:
        # Keep system + last (max_history * 2) turns
        if len(self._messages) > 1 + self._max_history * 2:
            self._messages = (
                self._messages[:1] + self._messages[-(self._max_history * 2):]
            )


# ──────────────────────────────────────────────────────────────────────
# HuggingFace Transformers (local) chat policy — Colab / Runpod
# ──────────────────────────────────────────────────────────────────────


class HFTransformersChatPolicy:
    """
    Local HF Transformers adapter for instruction-tuned chat models (e.g. Gemma).

    Designed for Colab/Runpod: loads a model with Transformers, runs generate(),
    then parses a JSON action via `_parse_action_json`.

    Notes for Gemma on Colab:
      - You likely want 4-bit loading (bitsandbytes) to fit in VRAM.
      - Many Gemma checkpoints are gated; use `huggingface-cli login` first.
    """

    def __init__(
        self,
        model_id: str,
        *,
        device_map: str = "auto",
        load_in_4bit: bool = True,
        torch_dtype: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_history: int = 6,
    ) -> None:
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "transformers/torch not installed. `pip install torch transformers accelerate bitsandbytes`"
            ) from e

        self._torch = torch
        self._model_id = model_id
        self._max_new_tokens = int(max_new_tokens)
        self._temperature = float(temperature)
        self._top_p = float(top_p)
        self._max_history = int(max_history)

        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

        model_kwargs: Dict[str, Any] = {"device_map": device_map}
        if load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        if torch_dtype != "auto":
            # allow passing "float16" / "bfloat16"
            model_kwargs["torch_dtype"] = getattr(torch, torch_dtype)

        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        model.eval()

        self._tokenizer = tok
        self._model = model
        self._messages: List[Dict[str, str]] = []
        self._task: Optional[str] = None

    def reset(self, task_name: str) -> None:
        self._task = task_name
        self._messages = [{"role": "system", "content": _OPENAI_SYSTEM_PROMPT}]

    def __call__(self, observation, phase, task_name):
        if self._task != task_name or not self._messages:
            self.reset(task_name)

        user_payload = {
            "phase":         phase,
            "observation":   observation,
            "valid_actions": observation.get("valid_actions", []),
        }
        self._messages.append(
            {"role": "user", "content": json.dumps(user_payload)[:6000]}
        )

        prompt = self._format_chat(self._messages)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        # Put tensors on the same device as the model's embedding matrix if possible
        try:
            device = self._model.get_input_embeddings().weight.device
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except Exception:
            pass

        gen_kwargs = {
            "max_new_tokens": self._max_new_tokens,
            "do_sample":      self._temperature > 0,
            "temperature":    self._temperature,
            "top_p":          self._top_p,
            "eos_token_id":   self._tokenizer.eos_token_id,
        }

        with self._torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)

        text = self._tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:],
                                      skip_special_tokens=True)
        self._messages.append({"role": "assistant", "content": text})
        self._trim_history()
        return _parse_action_json(text)

    def _format_chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Conservative formatting: plain-text transcript. Works across most
        instruction-tuned LMs even without native chat templates.
        """
        parts: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"[SYSTEM]\n{content}\n")
            elif role == "user":
                parts.append(f"[USER]\n{content}\n")
            else:
                parts.append(f"[ASSISTANT]\n{content}\n")
        parts.append("[ASSISTANT]\n")
        return "\n".join(parts)

    def _trim_history(self) -> None:
        if len(self._messages) > 1 + self._max_history * 2:
            self._messages = self._messages[:1] + self._messages[-(self._max_history * 2):]


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _parse_action_json(text: str) -> Dict[str, Any]:
    """
    Best-effort extract a JSON object from a model's text reply.
    Falls back to declare_no_change on parse failure.
    """
    text = text.strip()
    # Strip ``` fences
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    # Find first { ... }
    a = text.find("{")
    b = text.rfind("}")
    if a == -1 or b == -1 or b <= a:
        return {"action_type": ActionType.DECLARE_NO_CHANGE.value,
                "parameters": {"reason": "could not parse JSON"}}
    try:
        obj = json.loads(text[a:b + 1])
        if not isinstance(obj, dict):
            raise ValueError("not a dict")
        obj.setdefault("parameters", {})
        return obj
    except Exception:
        return {"action_type": ActionType.DECLARE_NO_CHANGE.value,
                "parameters": {"reason": "JSON parse error"}}
