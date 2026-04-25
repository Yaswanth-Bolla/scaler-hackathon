"""
inference.py — SRE Incident Response + Code Attribution Agent

Two execution modes:
  baseline  — flat P1-only loop (comparison baseline, no orchestrator)
  unified   — orchestrator → ops subagent → code subagent (research mode)

Three LLM backends (set BACKEND env var):
  local     — load checkpoint from LOCAL_MODEL_PATH using transformers + Unsloth
  vllm      — serve checkpoint via vLLM (faster, needs vllm installed + server running)
  api       — OpenAI-compatible HTTP API (baseline comparisons only)

stdout contract (OpenEnv evaluator parses this — do not change field names):
  [START] task=<n>
  [STEP]  step=<n> phase=<1|2> action=<json>
  [END]   task=<n> score=<f> reward=<f> steps=<n>

Environment variables:
  BACKEND          local | vllm | api          (default: local)
  LOCAL_MODEL_PATH path to checkpoint dir      (default: ./checkpoint)
  LOAD_IN_4BIT     1 | 0                       (default: 1)
  VLLM_BASE_URL    vLLM server URL             (default: http://localhost:8001)
  API_BASE_URL     OpenAI-compatible API URL   (api backend only)
  API_KEY          API key                     (api backend only)
  MODEL_NAME       model name string           (api / vllm backends)
  ENV_BASE_URL     OpenEnv server URL          (default: http://localhost:8000)
  MODE             baseline | unified          (default: unified)
  COLLECT          1 to write trajectory JSON  (default: 0)
  MAX_NEW_TOKENS   token budget per call       (default: 512)
  TEMPERATURE      sampling temperature        (default: 0.3)
  ORCH_TEMPERATURE orchestrator temperature    (default: 0.1)
"""

from __future__ import annotations

import json
import os
import re
import sys
import traceback
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


# ══════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════

BACKEND          = os.environ.get("BACKEND",          "local")
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "./checkpoint")
VLLM_BASE_URL    = os.environ.get("VLLM_BASE_URL",    "http://localhost:8001")
API_BASE_URL     = os.environ.get("API_BASE_URL",     "")
API_KEY          = os.environ.get("API_KEY",          "no-key")
MODEL_NAME       = os.environ.get("MODEL_NAME",       "checkpoint")
ENV_BASE_URL     = os.environ.get("ENV_BASE_URL",     "http://localhost:8000")
MODE             = os.environ.get("MODE",             "unified")
COLLECT          = os.environ.get("COLLECT",          "0") == "1"

MAX_NEW_TOKENS   = int(os.environ.get("MAX_NEW_TOKENS",    "512"))
TEMPERATURE      = float(os.environ.get("TEMPERATURE",     "0.3"))
ORCH_TEMPERATURE = float(os.environ.get("ORCH_TEMPERATURE","0.1"))

MAX_P1_STEPS = 20
MAX_P2_STEPS = 15


# ══════════════════════════════════════════════════════════════════
# Prompts
# ══════════════════════════════════════════════════════════════════

OPS_SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) responding to a production incident.

Your goal:
  1. DIAGNOSE the root cause from observable symptoms only
  2. REMEDIATE by acting on the correct service
  3. DECLARE when you are confident

## Action schema — respond with ONE JSON object per turn

Diagnostic (no state mutation):
  {"action_type": "view_alerts"}
  {"action_type": "query_logs",           "target_service": "<svc>", "parameters": {"level": "ERROR"}}
  {"action_type": "check_metrics",        "target_service": "<svc>"}
  {"action_type": "check_dependencies",   "target_service": "<svc>"}
  {"action_type": "check_deploy_history", "target_service": "<svc>"}
  {"action_type": "run_health_check",     "target_service": "<svc>"}

Remediation (mutates state):
  {"action_type": "restart_service", "target_service": "<svc>"}
  {"action_type": "rollback_deploy", "target_service": "<svc>"}
  {"action_type": "scale_service",   "target_service": "<svc>", "parameters": {"replicas": 5}}

Terminal:
  {"action_type": "declare_root_cause", "parameters": {"root_cause": "<diagnosis>"}}

Services: api_gateway, auth, orders, payment, cache, database, queue

## Strategy
  - view_alerts first to understand scope
  - check_metrics + query_logs on the highest-severity service
  - check_dependencies to trace upstream root causes
  - check_deploy_history before any rollback
  - remediate the ROOT cause service first
  - declare when confident — do not delay unnecessarily

IMPORTANT: Output ONLY valid JSON. No markdown, no explanation.
"""

ORCHESTRATOR_PROMPT = """You are the orchestrator of a two-phase SRE incident response system.

After each ops agent action you assess the current belief state and decide whether to
continue Phase 1 (gather more evidence) or transition to Phase 2 (codebase attribution).

Rules:
  - transition only when suspected_service is identified with reasonable confidence
  - do NOT transition just because steps are high — bad evidence is worse than no transition
  - evidence_gaps must list specific missing checks (e.g. "deploy_history_unchecked")
  - estimated_p2_cost reflects how broad the codebase search will need to be

Output ONLY this XML block — no other text:

<belief_state>
  <suspected_service>{service name or "unknown"}</suspected_service>
  <suspected_fault_class>{memory_leak|config_change|deadlock|resource_exhaustion|cascading|none}</suspected_fault_class>
  <service_confidence>{0.00 to 1.00}</service_confidence>
  <fault_confidence>{0.00 to 1.00}</fault_confidence>
  <evidence_gaps>{comma-separated list or "none"}</evidence_gaps>
  <estimated_p2_cost>{low|medium|high}</estimated_p2_cost>
  <decision>{continue|transition}</decision>
  <reasoning>{one concise sentence}</reasoning>
</belief_state>
"""

CODE_AGENT_PROMPT = """You are a senior software engineer performing code attribution for a production incident.

Runtime diagnosis handed off from SRE phase:
  Faulty service : {service}
  Fault class    : {fault_class}
  Bad deploy SHA : {commit_sha}
  Confidence     : service={service_confidence}  fault={fault_confidence}

Your job: explore the codebase snapshot, find the exact change that caused the incident,
then either propose a patch or declare that no code change is needed.

## Action schema — respond with ONE JSON object per turn

Exploration:
  {"action_type": "list_dir",      "parameters": {"path": "."}}
  {"action_type": "read_file",     "parameters": {"path": "<rel_path>"}}
  {"action_type": "search_code",   "parameters": {"query": "<text>", "file_pattern": "*.py"}}
  {"action_type": "get_git_log",   "parameters": {"path": "<rel_path>", "n_commits": 5}}
  {"action_type": "get_file_diff", "parameters": {"commit_sha": "<sha>", "path": "<rel_path>"}}

Terminal:
  {"action_type": "propose_patch",     "parameters": {"diff": "<unified diff>", "explanation": "<reason>"}}
  {"action_type": "declare_no_change", "parameters": {"reason": "<why no code change is needed>"}}

## Strategy
  1. list_dir to understand repo structure
  2. get_git_log on the bad commit SHA to see which files changed
  3. read_file on each changed file to understand the bug
  4. propose_patch with a minimal correct unified diff
  5. If symptoms are infra-only (config, scaling) with no bad code: declare_no_change

IMPORTANT: Output ONLY valid JSON. No markdown, no explanation.
"""


# ══════════════════════════════════════════════════════════════════
# LLM backend abstraction
# ══════════════════════════════════════════════════════════════════

Message = Dict[str, str]  # {"role": "system"|"user"|"assistant", "content": str}


class LLMBackend(ABC):
    """
    Uniform interface over local checkpoint, vLLM, and API backends.
    All call sites use backend.generate(messages, temperature, max_new_tokens).
    Swapping backends requires only changing the BACKEND env var.
    """

    @abstractmethod
    def generate(
        self,
        messages:       List[Message],
        temperature:    float = TEMPERATURE,
        max_new_tokens: int   = MAX_NEW_TOKENS,
    ) -> str:
        """Return the assistant response text, stripped."""


# ── Local checkpoint ──────────────────────────────────────────────

class LocalModelBackend(LLMBackend):
    """
    Loads a HuggingFace checkpoint from LOCAL_MODEL_PATH.

    Uses Unsloth when available for 2x faster inference with identical output.
    Falls back to vanilla transformers if Unsloth is not installed.

    The model loads once at construction and is reused across all episodes.
    apply_chat_template handles the system/user/assistant turn format for
    Qwen, Llama, Mistral and other chat models automatically.
    """

    def __init__(self, model_path: str, load_in_4bit: bool = True):
        self.model_path   = model_path
        self.load_in_4bit = load_in_4bit
        self.model        = None
        self.tokenizer    = None
        self._load()

    def _load(self) -> None:
        _log(f"Loading checkpoint: {self.model_path}")

        try:
            from unsloth import FastLanguageModel
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name    = self.model_path,
                max_seq_length= 8192,
                load_in_4bit  = self.load_in_4bit,
                dtype         = None,  # auto — bfloat16 on Ampere+
            )
            FastLanguageModel.for_inference(self.model)
            _log(f"Backend: Unsloth  4bit={self.load_in_4bit}")

        except ImportError:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype      = "auto",
                device_map       = "auto",
                trust_remote_code= True,
            )
            self.model.eval()
            _log("Backend: transformers (Unsloth not found, using vanilla)")

    def generate(
        self,
        messages:       List[Message],
        temperature:    float = TEMPERATURE,
        max_new_tokens: int   = MAX_NEW_TOKENS,
    ) -> str:
        import torch

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize              = False,
            add_generation_prompt = True,
        )
        inputs    = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens = max_new_tokens,
                temperature    = temperature,
                do_sample      = temperature > 0,
                pad_token_id   = self.tokenizer.eos_token_id,
            )

        # Decode only newly generated tokens — strip the echoed prompt
        new_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── vLLM backend ──────────────────────────────────────────────────

class VLLMBackend(LLMBackend):
    """
    Calls a locally running vLLM server via its OpenAI-compatible endpoint.

    Start the server with:
      python -m vllm.entrypoints.openai.api_server \\
        --model ./checkpoint --port 8001

    No model loading here — the server handles it. Use this when running
    rapid eval loops and model load time is a bottleneck.
    """

    def __init__(self, base_url: str, model_name: str):
        try:
            from openai import OpenAI as _OpenAI
        except ImportError:
            raise ImportError("pip install openai  (required for vllm backend)")
        self._client = _OpenAI(api_key="vllm-local", base_url=base_url)
        self._model  = model_name
        _log(f"Backend: vLLM at {base_url}  model={model_name}")

    def generate(
        self,
        messages:       List[Message],
        temperature:    float = TEMPERATURE,
        max_new_tokens: int   = MAX_NEW_TOKENS,
    ) -> str:
        resp = self._client.chat.completions.create(
            model       = self._model,
            messages    = messages,
            temperature = temperature,
            max_tokens  = max_new_tokens,
        )
        return (resp.choices[0].message.content or "").strip()


# ── API backend ───────────────────────────────────────────────────

class APIBackend(LLMBackend):
    """
    Wraps any OpenAI-compatible HTTP API.
    Use only for baseline comparisons — not for checkpoint inference.
    """

    def __init__(self, api_key: str, base_url: Optional[str], model_name: str):
        try:
            from openai import OpenAI as _OpenAI
        except ImportError:
            raise ImportError("pip install openai  (required for api backend)")
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = _OpenAI(**kwargs)
        self._model  = model_name
        _log(f"Backend: API  model={model_name}  base={base_url or 'openai'}")

    def generate(
        self,
        messages:       List[Message],
        temperature:    float = TEMPERATURE,
        max_new_tokens: int   = MAX_NEW_TOKENS,
    ) -> str:
        resp = self._client.chat.completions.create(
            model       = self._model,
            messages    = messages,
            temperature = temperature,
            max_tokens  = max_new_tokens,
        )
        return (resp.choices[0].message.content or "").strip()


# ── Factory ───────────────────────────────────────────────────────

def build_backend() -> LLMBackend:
    if BACKEND == "local":
        return LocalModelBackend(
            model_path   = LOCAL_MODEL_PATH,
            load_in_4bit = os.environ.get("LOAD_IN_4BIT", "1") == "1",
        )
    if BACKEND == "vllm":
        return VLLMBackend(base_url=VLLM_BASE_URL, model_name=MODEL_NAME)
    if BACKEND == "api":
        return APIBackend(
            api_key    = API_KEY,
            base_url   = API_BASE_URL or None,
            model_name = MODEL_NAME,
        )
    raise ValueError(f"Unknown BACKEND={BACKEND!r}. Choose: local | vllm | api")


# ══════════════════════════════════════════════════════════════════
# Data containers
# ══════════════════════════════════════════════════════════════════

@dataclass
class BeliefState:
    suspected_service:    str   = "unknown"
    suspected_fault_class: str  = "none"
    service_confidence:   float = 0.0
    fault_confidence:     float = 0.0
    evidence_gaps:        str   = "none"
    estimated_p2_cost:    str   = "unknown"
    decision:             str   = "continue"
    reasoning:            str   = ""

    def confident_enough(self) -> bool:
        """
        Orchestrator stopping criterion.
        Stage 4 GRPO training trains the model to emit the correct
        <decision> tag — this method is therefore the learned policy
        expressed as a single field check.
        """
        return self.decision == "transition"


@dataclass
class StepRecord:
    step_number:  int
    phase:        int
    action:       Dict[str, Any]
    reward:       float
    obs_summary:  Dict[str, Any]
    belief:       Optional[Dict[str, Any]] = None  # P1 only


@dataclass
class EpisodeRecord:
    task_name:           str
    mode:                str
    seed:                int
    p1_trajectory:       List[StepRecord] = field(default_factory=list)
    p2_trajectory:       List[StepRecord] = field(default_factory=list)
    belief_history:      List[Dict]       = field(default_factory=list)
    declared_patch:      Optional[str]    = None
    declared_no_change:  bool             = False
    phase_transition_at: Optional[int]    = None
    score:               float            = 0.0
    cumulative_reward:   float            = 0.0
    total_steps:         int              = 0


# ══════════════════════════════════════════════════════════════════
# Environment client
# ══════════════════════════════════════════════════════════════════

class EnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session  = requests.Session()

    def reset(self, task_name: str, seed: int = 42) -> Dict[str, Any]:
        r = self.session.post(
            f"{self.base_url}/reset",
            json={"task_name": task_name, "seed": seed},
        )
        r.raise_for_status()
        return r.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        r = self.session.post(f"{self.base_url}/step", json=action)
        r.raise_for_status()
        return r.json()

    def state(self) -> Dict[str, Any]:
        r = self.session.get(f"{self.base_url}/state")
        r.raise_for_status()
        return r.json()

    def unified_score(
        self,
        declared_patch:     Optional[str],
        declared_no_change: bool,
        belief_history:     List[Dict],
    ) -> Dict[str, float]:
        try:
            r = self.session.post(
                f"{self.base_url}/score",
                json={
                    "declared_patch":     declared_patch,
                    "declared_no_change": declared_no_change,
                    "belief_history":     belief_history,
                },
            )
            r.raise_for_status()
            return r.json()
        except Exception:
            return {"final": 0.01}


# ══════════════════════════════════════════════════════════════════
# Parsing helpers
# ══════════════════════════════════════════════════════════════════

def parse_action(text: str) -> Dict[str, Any]:
    """
    Extract JSON action from model output.
    Local models sometimes wrap output in prose or markdown — we
    defensively extract the first complete JSON object.
    """
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(
            l for l in text.split("\n")
            if not l.strip().startswith("```")
        ).strip()
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(text[start:end])
    raise ValueError(f"No JSON in model output: {text[:300]}")


def parse_belief(xml_text: str) -> BeliefState:
    def _x(tag: str) -> str:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", xml_text, re.DOTALL)
        return m.group(1).strip() if m else ""

    return BeliefState(
        suspected_service    = _x("suspected_service")     or "unknown",
        suspected_fault_class= _x("suspected_fault_class") or "none",
        service_confidence   = _safe_float(_x("service_confidence")),
        fault_confidence     = _safe_float(_x("fault_confidence")),
        evidence_gaps        = _x("evidence_gaps")         or "none",
        estimated_p2_cost    = _x("estimated_p2_cost")     or "unknown",
        decision             = _x("decision")              or "continue",
        reasoning            = _x("reasoning")             or "",
    )


def _safe_float(s: str) -> float:
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def summarise_obs(obs: Dict[str, Any]) -> str:
    parts = [
        f"Incident : {obs.get('incident_summary', 'N/A')}",
        f"Severity : {obs.get('severity', 'N/A')}",
        f"Time     : {obs.get('time_elapsed_minutes', 0)}/{obs.get('time_budget_minutes', 30)} min",
        f"Steps    : {obs.get('steps_taken', 0)}/{obs.get('max_steps', 20)}",
        f"Reward   : {obs.get('current_reward', 0):.3f}  (Σ {obs.get('cumulative_reward', 0):.3f})",
    ]
    statuses = obs.get("service_statuses", {})
    if statuses:
        parts.append("Services : " + "  ".join(f"{k}={v}" for k, v in statuses.items()))
    parts.append(f"Alerts   : {obs.get('active_alerts_count', 0)} active")
    parts.append(f"Result   : {obs.get('action_message', '')}")
    data = obs.get("action_result", {})
    if data:
        blob = json.dumps(data, indent=2, default=str)
        if len(blob) > 2000:
            blob = blob[:2000] + "\n… (truncated)"
        parts.append(f"Data:\n{blob}")
    return "\n".join(parts)


def obs_summary_dict(obs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "incident_summary":    obs.get("incident_summary", ""),
        "severity":            obs.get("severity", ""),
        "service_statuses":    obs.get("service_statuses", {}),
        "active_alerts_count": obs.get("active_alerts_count", 0),
        "action_message":      obs.get("action_message", ""),
        "current_phase":       obs.get("current_phase", 1),
    }


# ══════════════════════════════════════════════════════════════════
# Phase 1 — Ops subagent
# ══════════════════════════════════════════════════════════════════

def run_phase1(
    env:      EnvClient,
    backend:  LLMBackend,
    init_obs: Dict[str, Any],
    episode:  EpisodeRecord,
) -> tuple[BeliefState, Dict[str, Any]]:
    """
    Ops diagnostic loop.

    The orchestrator is invoked after each ops action using a separate
    system prompt and lower temperature.  It sees the full ops conversation
    history so it can reason about cumulative evidence, not just the last step.

    The ops and orchestrator calls are kept as separate generate() calls
    rather than a single call with combined prompt — this lets them be
    trained independently in Stage 2 and the orchestrator auxiliary loss
    in Stage 4 without entangling their gradients.
    """
    ops_messages: List[Message] = [
        {"role": "system", "content": OPS_SYSTEM_PROMPT},
        {"role": "user",   "content": f"INCIDENT TRIGGERED:\n{summarise_obs(init_obs)}"},
    ]

    belief   = BeliefState()
    last_obs = init_obs

    for p1_step in range(1, MAX_P1_STEPS + 1):

        # ── Ops agent selects next action ─────────────────────────
        try:
            ops_text = backend.generate(ops_messages, temperature=TEMPERATURE)
            action   = parse_action(ops_text)
        except Exception as e:
            _warn(f"P1 ops error step {p1_step}: {e}")
            action   = {"action_type": "view_alerts"}
            ops_text = json.dumps(action)

        # ── Orchestrator evaluates belief after seeing the action ─
        # Orchestrator gets its own system prompt, then the full ops
        # conversation up to and including the chosen action.
        orch_messages: List[Message] = (
            [{"role": "system", "content": ORCHESTRATOR_PROMPT}]
            + ops_messages[1:]  # history without the ops system prompt
            + [
                {"role": "assistant", "content": ops_text},
                {"role": "user",
                 "content": "Based on all evidence gathered so far, output your belief state now."},
            ]
        )
        try:
            orch_text = backend.generate(
                orch_messages,
                temperature    = ORCH_TEMPERATURE,
                max_new_tokens = 300,
            )
            belief = parse_belief(orch_text)
        except Exception as e:
            _warn(f"Orchestrator error step {p1_step}: {e}")

        episode.belief_history.append(asdict(belief))

        print(
            f"[STEP] step={p1_step} phase=1 "
            f"action={json.dumps(action)} "
            f"svc={belief.suspected_service} "
            f"svc_conf={belief.service_confidence:.2f} "
            f"decision={belief.decision}"
        )

        # ── Execute in environment ────────────────────────────────
        try:
            step_result = env.step(action)
        except Exception as e:
            _warn(f"Env step error P1 step {p1_step}: {e}")
            break

        last_obs   = step_result.get("observation", {})
        reward     = step_result.get("reward",  0.0)
        done       = step_result.get("done",    False)
        episode.cumulative_reward += reward
        episode.total_steps        = p1_step

        episode.p1_trajectory.append(StepRecord(
            step_number = p1_step,
            phase       = 1,
            action      = action,
            reward      = reward,
            obs_summary = obs_summary_dict(last_obs),
            belief      = asdict(belief),
        ))

        ops_messages.append({"role": "assistant", "content": ops_text})
        ops_messages.append({
            "role":    "user",
            "content": f"Step {p1_step} result (reward={reward:.3f}):\n{summarise_obs(last_obs)}",
        })

        if done:
            break

        if belief.confident_enough():
            episode.phase_transition_at = p1_step
            break

    return belief, last_obs


# ══════════════════════════════════════════════════════════════════
# Phase 2 — Code subagent
# ══════════════════════════════════════════════════════════════════

def run_phase2(
    env:     EnvClient,
    backend: LLMBackend,
    belief:  BeliefState,
    episode: EpisodeRecord,
) -> None:
    """
    Triggers environment phase transition then runs code exploration.

    The code agent gets a fresh context window — it does NOT receive
    the P1 ops conversation history.  The handoff is only the structured
    belief state fields (service, fault class, commit SHA).

    This is deliberate: it forces the code agent to form its own
    code-level evidence independently, and means belief state quality
    directly gates Phase 2 search efficiency (the r_cross mechanism).
    """
    try:
        p2_init = env.step({
            "action_type": "transition_to_phase2",
            "parameters":  {"belief": asdict(belief)},
        })
        p2_obs = p2_init.get("observation", {})
    except Exception as e:
        _warn(f"Phase transition failed: {e}")
        return

    commit_sha = p2_obs.get("bad_commit_sha", "unknown")

    code_prompt = CODE_AGENT_PROMPT.format(
        service            = belief.suspected_service,
        fault_class        = belief.suspected_fault_class,
        commit_sha         = commit_sha,
        service_confidence = f"{belief.service_confidence:.2f}",
        fault_confidence   = f"{belief.fault_confidence:.2f}",
    )

    messages: List[Message] = [
        {"role": "system", "content": code_prompt},
        {"role": "user",   "content": f"Codebase context:\n{summarise_obs(p2_obs)}"},
    ]

    for p2_step in range(1, MAX_P2_STEPS + 1):
        global_step = episode.total_steps + p2_step

        try:
            resp_text = backend.generate(
                messages,
                temperature    = TEMPERATURE,
                max_new_tokens = MAX_NEW_TOKENS,
            )
            action = parse_action(resp_text)
        except Exception as e:
            _warn(f"P2 action error step {p2_step}: {e}")
            action    = {"action_type": "list_dir", "parameters": {"path": "."}}
            resp_text = json.dumps(action)

        a_type = action.get("action_type", "")
        print(f"[STEP] step={global_step} phase=2 action={json.dumps(action)}")

        if a_type == "propose_patch":
            episode.declared_patch = action.get("parameters", {}).get("diff", "")
        elif a_type == "declare_no_change":
            episode.declared_no_change = True

        try:
            step_result = env.step(action)
        except Exception as e:
            _warn(f"Env step error P2 step {p2_step}: {e}")
            break

        step_obs   = step_result.get("observation", {})
        reward     = step_result.get("reward",  0.0)
        done       = step_result.get("done",    False)
        episode.cumulative_reward += reward
        episode.total_steps        = global_step

        episode.p2_trajectory.append(StepRecord(
            step_number = global_step,
            phase       = 2,
            action      = action,
            reward      = reward,
            obs_summary = obs_summary_dict(step_obs),
        ))

        messages.append({"role": "assistant", "content": resp_text})
        messages.append({
            "role":    "user",
            "content": f"Step result:\n{summarise_obs(step_obs)}",
        })

        if done or a_type in {"propose_patch", "declare_no_change"}:
            break


# ══════════════════════════════════════════════════════════════════
# Episode runners
# ══════════════════════════════════════════════════════════════════

def run_episode_baseline(
    env:       EnvClient,
    backend:   LLMBackend,
    task_name: str,
    seed:      int = 42,
) -> EpisodeRecord:
    """
    Flat P1-only loop — no orchestrator, no Phase 2.
    Ablation Claim 1: compare against run_episode_unified to prove
    the orchestrator adds value beyond a fixed-strategy baseline.
    """
    print(f"[START] task={task_name}")
    episode = EpisodeRecord(task_name=task_name, mode="baseline", seed=seed)
    result  = env.reset(task_name, seed)
    obs     = result["observation"]

    messages: List[Message] = [
        {"role": "system", "content": OPS_SYSTEM_PROMPT},
        {"role": "user",   "content": f"INCIDENT TRIGGERED:\n{summarise_obs(obs)}"},
    ]

    final_info: Dict[str, Any] = {}

    for step_num in range(1, MAX_P1_STEPS + 1):
        try:
            resp_text = backend.generate(messages, temperature=TEMPERATURE)
            action    = parse_action(resp_text)
        except Exception as e:
            _warn(f"Baseline action error step {step_num}: {e}")
            action    = {"action_type": "view_alerts"}
            resp_text = json.dumps(action)

        print(f"[STEP] step={step_num} phase=1 action={json.dumps(action)}")

        try:
            step_result = env.step(action)
        except Exception as e:
            _warn(f"Baseline env step error step {step_num}: {e}")
            break

        obs     = step_result.get("observation", {})
        reward  = step_result.get("reward", 0.0)
        done    = step_result.get("done",   False)
        info    = step_result.get("info",   {})
        episode.cumulative_reward += reward
        episode.total_steps        = step_num

        episode.p1_trajectory.append(StepRecord(
            step_number = step_num,
            phase       = 1,
            action      = action,
            reward      = reward,
            obs_summary = obs_summary_dict(obs),
        ))

        messages.append({"role": "assistant", "content": resp_text})
        messages.append({
            "role":    "user",
            "content": f"Step {step_num} result (reward={reward:.3f}):\n{summarise_obs(obs)}",
        })

        if done:
            final_info = info
            break

    episode.score = final_info.get("score", 0.01)
    print(
        f"[END] task={task_name} score={episode.score:.3f} "
        f"reward={episode.cumulative_reward:.3f} steps={episode.total_steps}"
    )
    return episode


def run_episode_unified(
    env:       EnvClient,
    backend:   LLMBackend,
    task_name: str,
    seed:      int = 42,
) -> EpisodeRecord:
    """
    Full two-phase episode:
      Phase 1 — ops subagent diagnoses runtime incident
      Orchestrator — belief state + stopping criterion after each P1 step
      Phase 2 — code subagent explores codebase and proposes patch
    """
    print(f"[START] task={task_name}")
    episode = EpisodeRecord(task_name=task_name, mode="unified", seed=seed)
    result  = env.reset(task_name, seed)
    obs     = result["observation"]

    belief, _ = run_phase1(env, backend, obs, episode)

    if episode.phase_transition_at is not None:
        run_phase2(env, backend, belief, episode)

    score_breakdown = env.unified_score(
        declared_patch     = episode.declared_patch,
        declared_no_change = episode.declared_no_change,
        belief_history     = episode.belief_history,
    )
    episode.score = score_breakdown.get("final", 0.01)

    print(
        f"[END] task={task_name} score={episode.score:.3f} "
        f"reward={episode.cumulative_reward:.3f} steps={episode.total_steps} "
        f"transition_at={episode.phase_transition_at}"
    )
    return episode


# ══════════════════════════════════════════════════════════════════
# Trajectory persistence (SFT / GRPO data collection)
# ══════════════════════════════════════════════════════════════════

def save_trajectory(episode: EpisodeRecord) -> None:
    """
    Write episode to trajectories/<task>_<mode>_<n>.json.

    Schema matches training/trajectory_collector.py:
      p1_reward   — grader p1_rca + p1_efficiency  (filled post-hoc)
      p2_reward   — grader patch + no_change scores (filled post-hoc)
      r_cross     — counterfactual cross-phase reward (filled in Stage 4)
      belief_history — per-step orchestrator beliefs, primary signal for r_cross
    """
    out_dir = Path("trajectories")
    out_dir.mkdir(exist_ok=True)

    idx  = len(list(out_dir.glob(f"{episode.task_name}_{episode.mode}_*.json")))
    path = out_dir / f"{episode.task_name}_{episode.mode}_{idx:04d}.json"

    record = {
        "task_name":           episode.task_name,
        "mode":                episode.mode,
        "seed":                episode.seed,
        "backend":             BACKEND,
        "score":               episode.score,
        "cumulative_reward":   episode.cumulative_reward,
        "total_steps":         episode.total_steps,
        "phase_transition_at": episode.phase_transition_at,
        "declared_patch":      episode.declared_patch,
        "declared_no_change":  episode.declared_no_change,
        "belief_history":      episode.belief_history,
        "p1_actions": [
            {"step": r.step_number, "action": r.action,
             "reward": r.reward, "belief": r.belief}
            for r in episode.p1_trajectory
        ],
        "p2_actions": [
            {"step": r.step_number, "action": r.action, "reward": r.reward}
            for r in episode.p2_trajectory
        ],
        # Reward components filled post-hoc by grader / trajectory_collector
        "p1_reward": 0.0,
        "p2_reward": episode.score,
        "r_cross":   0.0,
    }

    path.write_text(json.dumps(record, indent=2, default=str))
    _log(f"Trajectory saved → {path}")


# ══════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════

def _log(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr, flush=True)


def _print_summary(results: List[EpisodeRecord]) -> None:
    print(f"\n{'═' * 64}")
    print(f"  RESULTS SUMMARY   mode={MODE}  backend={BACKEND}")
    print(f"{'═' * 64}")
    for r in results:
        tr = f"→P2@step{r.phase_transition_at}" if r.phase_transition_at else "P1-only"
        print(
            f"  {r.task_name:30s}  score={r.score:.3f}  "
            f"steps={r.total_steps:2d}  {tr}"
        )
    if results:
        avg = sum(r.score for r in results) / len(results)
        print(f"\n  {'AVERAGE':30s}  score={avg:.3f}")
    print(f"{'═' * 64}")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    tasks = ["memory_leak", "cascading_failure", "distributed_deadlock"]

    print("═" * 64)
    print("  SRE Incident Response — OpenEnv Inference")
    print(f"  Backend : {BACKEND}")
    print(f"  Model   : {LOCAL_MODEL_PATH if BACKEND == 'local' else MODEL_NAME}")
    print(f"  Mode    : {MODE}")
    print(f"  Env     : {ENV_BASE_URL}")
    print(f"  Collect : {COLLECT}")
    print("═" * 64)

    env     = EnvClient(ENV_BASE_URL)
    backend = build_backend()   # model loads once here

    run_fn = run_episode_unified if MODE == "unified" else run_episode_baseline
    results: List[EpisodeRecord] = []

    for task in tasks:
        print(f"\n{'─' * 40}")
        print(f"  Task: {task}")
        print(f"{'─' * 40}")
        try:
            episode = run_fn(env, backend, task)
            results.append(episode)
            if COLLECT:
                save_trajectory(episode)
        except Exception as e:
            _warn(f"Task {task} failed: {e}")
            traceback.print_exc()
            print(f"[END] task={task} score=0.010 reward=0.000 steps=0")
            results.append(
                EpisodeRecord(task_name=task, mode=MODE, seed=42, score=0.01)
            )

    _print_summary(results)


if __name__ == "__main__":
    main()