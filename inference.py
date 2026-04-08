"""
Baseline inference script.

Uses an LLM (via OpenAI-compatible API) to play through all 3 incident
scenarios. The conversation history acts as a soft belief tracker —
the LLM accumulates evidence across steps.

stdout format: [START], [STEP], [END] blocks with exact field names
as required by the OpenEnv automated evaluator.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MAX_STEPS = 20
TEMPERATURE = 0.3


# ------------------------------------------------------------------
# System prompt — Layer 3: the LLM acts as an SRE
# ------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) responding to a production incident.

You are interacting with a simulated microservices infrastructure through an environment API.
Your goal is to:
1. DIAGNOSE the root cause of the incident
2. REMEDIATE the issue (fix it)
3. DECLARE the root cause when confident

## Available Actions
You must respond with a single JSON object containing your chosen action:

DIAGNOSTIC (information gathering):
- {"action_type": "view_alerts"} — See all firing alerts
- {"action_type": "query_logs", "target_service": "<name>", "parameters": {"level": "ERROR"}} — Query logs
- {"action_type": "check_metrics", "target_service": "<name>"} — Get metric timeseries
- {"action_type": "check_dependencies", "target_service": "<name>"} — View dependency graph
- {"action_type": "check_deploy_history", "target_service": "<name>"} — Recent deploys
- {"action_type": "run_health_check", "target_service": "<name>"} — Ping a service

REMEDIATION (fix actions):
- {"action_type": "restart_service", "target_service": "<name>"} — Restart a service
- {"action_type": "rollback_deploy", "target_service": "<name>"} — Rollback to previous deploy
- {"action_type": "scale_service", "target_service": "<name>", "parameters": {"replicas": 5}} — Scale replicas

DECLARATION:
- {"action_type": "declare_root_cause", "parameters": {"root_cause": "<your diagnosis>"}}

## Available services: api_gateway, auth, orders, payment, cache, database, queue

## Strategy
1. Start by viewing alerts to understand the scope
2. Check metrics and logs for the most affected services
3. Check dependency graphs to trace upstream causes
4. Check deploy history for recently changed services
5. Apply remediation to the root cause service FIRST
6. Declare root cause when confident

IMPORTANT: Respond with ONLY a valid JSON object. No explanation, no markdown, just the JSON action.
"""


# ------------------------------------------------------------------
# Environment client (direct HTTP)
# ------------------------------------------------------------------

class EnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def reset(self, task_name: str, seed: int = 42) -> Dict[str, Any]:
        resp = self.session.post(f"{self.base_url}/reset", json={
            "task_name": task_name, "seed": seed})
        resp.raise_for_status()
        return resp.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        resp = self.session.post(f"{self.base_url}/step", json=action)
        resp.raise_for_status()
        return resp.json()

    def state(self) -> Dict[str, Any]:
        resp = self.session.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return resp.json()


# ------------------------------------------------------------------
# LLM agent
# ------------------------------------------------------------------

def create_openai_client() -> OpenAI:
    """Create OpenAI client with appropriate config."""
    api_key = OPENAI_API_KEY or HF_TOKEN or "no-key"
    base_url = os.environ.get("API_BASE_URL")

    # If using HF inference endpoint, set base_url
    if HF_TOKEN and not OPENAI_API_KEY and not base_url:
        base_url = f"https://api-inference.huggingface.co/models/{MODEL_NAME}/v1"

    return OpenAI(api_key=api_key, base_url=base_url)


def parse_llm_action(response_text: str) -> Dict[str, Any]:
    """Extract JSON action from LLM response. Handles markdown wrapping."""
    text = response_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Find JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(text[start:end])

    raise ValueError(f"Could not parse action from: {response_text[:200]}")


def summarize_observation(obs: Dict[str, Any]) -> str:
    """Convert observation dict to a readable string for the LLM context."""
    parts = []
    parts.append(f"Incident: {obs.get('incident_summary', 'N/A')}")
    parts.append(f"Severity: {obs.get('severity', 'N/A')}")
    parts.append(f"Time: {obs.get('time_elapsed_minutes', 0)}/{obs.get('time_budget_minutes', 30)} min")
    parts.append(f"Steps: {obs.get('steps_taken', 0)}/{obs.get('max_steps', 20)}")
    parts.append(f"Reward: {obs.get('current_reward', 0)} (cumulative: {obs.get('cumulative_reward', 0)})")

    statuses = obs.get("service_statuses", {})
    if statuses:
        status_str = ", ".join(f"{k}: {v}" for k, v in statuses.items())
        parts.append(f"Services: {status_str}")

    parts.append(f"Alerts: {obs.get('active_alerts_count', 0)} active")
    parts.append(f"Action result: {obs.get('action_message', 'N/A')}")

    # Include action_result details (truncated)
    action_result = obs.get("action_result", {})
    if action_result:
        result_str = json.dumps(action_result, indent=2, default=str)
        if len(result_str) > 2000:
            result_str = result_str[:2000] + "\n... (truncated)"
        parts.append(f"Data:\n{result_str}")

    return "\n".join(parts)


def run_episode(
    env: EnvClient,
    llm: OpenAI,
    task_name: str,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run a single episode and return results."""

    # --- [START] ---
    print(f"[START] task={task_name}")

    result = env.reset(task_name, seed)
    obs = result["observation"]

    # Conversation history for belief tracking (Layer 3)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"INCIDENT TRIGGERED:\n{summarize_observation(obs)}"},
    ]

    episode_reward = 0.0
    final_info = {}

    for step_num in range(1, MAX_STEPS + 1):
        try:
            # Get LLM action
            completion = llm.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=256,
            )
            llm_response = completion.choices[0].message.content or ""

            # Parse action
            action = parse_llm_action(llm_response)

            # --- [STEP] ---
            print(f"[STEP] step={step_num} action={json.dumps(action)}")

            # Execute in environment
            step_result = env.step(action)
            obs = step_result["observation"]
            reward = step_result.get("reward", 0.0)
            done = step_result.get("done", False)
            info = step_result.get("info", {})
            episode_reward += reward

            # Update conversation history (belief tracker)
            messages.append({"role": "assistant", "content": llm_response})
            messages.append({
                "role": "user",
                "content": f"Step {step_num} result (reward={reward}):\n{summarize_observation(obs)}"
            })

            if done:
                final_info = info
                break

        except Exception as e:
            print(f"[STEP] step={step_num} error={str(e)}", file=sys.stderr)
            # Fallback action: view alerts
            action = {"action_type": "view_alerts"}
            step_result = env.step(action)
            obs = step_result["observation"]
            reward = step_result.get("reward", 0.0)
            done = step_result.get("done", False)
            episode_reward += reward
            if done:
                final_info = step_result.get("info", {})
                break

    # Get final state
    final_state = env.state()
    final_grade = final_info.get("final_grade", 0.0)

    # --- [END] ---
    print(f"[END] task={task_name} "
          f"grade={final_grade:.3f} "
          f"reward={episode_reward:.3f} "
          f"steps={final_state.get('step_count', 0)}")

    return {
        "task_name": task_name,
        "final_grade": final_grade,
        "cumulative_reward": episode_reward,
        "steps": final_state.get("step_count", 0),
        "declared_root_cause": final_state.get("declared_root_cause"),
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    tasks = ["memory_leak", "cascading_failure", "distributed_deadlock"]

    print("=" * 60)
    print("SRE Incident Response — OpenEnv Inference")
    print(f"Model: {MODEL_NAME}")
    print(f"Environment: {ENV_BASE_URL}")
    print("=" * 60)

    env = EnvClient(ENV_BASE_URL)
    llm = create_openai_client()

    results = []
    for task in tasks:
        print(f"\n{'─' * 40}")
        print(f"Task: {task}")
        print(f"{'─' * 40}")

        try:
            result = run_episode(env, llm, task)
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Task {task} failed: {e}", file=sys.stderr)
            traceback.print_exc()
            results.append({
                "task_name": task,
                "final_grade": 0.0,
                "cumulative_reward": 0.0,
                "steps": 0,
                "error": str(e),
            })

    # Summary
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")
    for r in results:
        print(f"  {r['task_name']:30s}  grade={r.get('final_grade', 0):.3f}  "
              f"steps={r.get('steps', 0):2d}  "
              f"root_cause={r.get('declared_root_cause', 'N/A')}")

    avg_grade = sum(r.get("final_grade", 0) for r in results) / len(results)
    print(f"\n  {'AVERAGE':30s}  grade={avg_grade:.3f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
