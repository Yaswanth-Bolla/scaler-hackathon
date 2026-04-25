"""
grpo_train.py — GRPO online RL training for the SRE incident env.

Stage-by-stage curriculum:
  Stage 2  (Pool A, p1_only)   ops diagnosis            — trains P1 only
  Stage 3  (Pool B, p2_only)   code investigation       — trains P2 only
  Stage 4  (Pool C, joint)     end-to-end with r_cross  — trains both

How it works:
  1. Collect K rollouts per task using the policy being trained.
  2. Score each rollout → final ∈ [0, 1].
  3. Normalise rewards within the group → GRPO advantages.
  4. Recompute token-level logprobs under policy + frozen reference model.
  5. Apply clipped PPO-style GRPO loss + KL penalty on assistant tokens only.
  6. Repeat across stages per curriculum.

Usage (local, from repo root):
    python training/grpo_train.py \\
        --model  srinjoyd/qwen2.5-7b-sre-merged \\
        --stages 2 3 4 \\
        --group_size 4 \\
        --episodes_per_task 64 \\
        --use_lora \\
        --push_to_hub srinjoyd/qwen2.5-7b-sre-grpo

HF Jobs (`hf jobs uv run`) often uploads **only the script file** to `/`,
so `server/` is missing → `ModuleNotFoundError: server`.  Fix by either:

  A) Clone the full repo inside the job, then run from that directory.

     **Trap:** ``bash -lc`` often resets ``PATH`` so ``python`` is *not* the
     same interpreter that ``hf jobs uv run`` installed ``torch`` into →
     ``ModuleNotFoundError: torch``.  Prefer ``bash -ec`` (no login) **or**
     nest ``uv run`` after ``cd`` so deps apply to the training process:

     hf jobs uv run --flavor h200 ... -- bash -ec '
       git clone https://github.com/<you>/scaler-hackathon.git /tmp/repo &&
       cd /tmp/repo && git checkout <branch> &&
       uv run --no-project --with torch --with transformers --with accelerate --with peft --with bitsandbytes --with tqdm --with fastapi --with uvicorn --with pydantic python training/grpo_train.py --model ... ...
     '

  B) Set an explicit root (if your job packs the tree elsewhere):
     INCIDENT_ENV_ROOT=/path/to/repo python training/grpo_train.py ...

Memory budget on A100 40 GB (--use_lora, bf16):
  Policy  : ~14 GB base weights + ~200 MB LoRA + optimizer ≈ 15 GB
  Ref     : ~14 GB frozen bf16
  Total   : ~29 GB + activations → fits A100 40 GB comfortably

Requirements:
    pip install torch transformers accelerate peft bitsandbytes tqdm
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── package path so imports work when run directly or via uv run ────
def _find_repo_root() -> Path:
    """
    Locate the scaler-hackathon repo root (directory that contains
    `server/incident_environment.py`).

    HF Jobs sometimes copies only `grpo_train.py` to `/` as `//grpo_train.py`;
    in that case *no* local `server/` exists — set ``INCIDENT_ENV_ROOT`` or
    clone the repo before running (see module docstring).
    """
    marker = Path("server") / "incident_environment.py"

    env_override = os.environ.get("INCIDENT_ENV_ROOT", "").strip()
    if env_override:
        root = Path(env_override).expanduser().resolve()
        if (root / marker).exists():
            return root
        raise RuntimeError(
            f"INCIDENT_ENV_ROOT={root} does not contain {marker.as_posix()}"
        )

    here = Path(__file__).resolve()
    for base in [here.parent, *here.parents]:
        if (base / marker).exists():
            return base

    cwd = Path.cwd().resolve()
    for base in [cwd, *cwd.parents]:
        if (base / marker).exists():
            return base

    raise RuntimeError(
        "Cannot find repo root containing server/incident_environment.py.\n"
        "This usually means the HF Job uploaded only grpo_train.py without "
        "the rest of the repository.\n\n"
        "Fix:\n"
        "  • Clone the full repo in the job command, then `cd` into it, OR\n"
        "  • Set INCIDENT_ENV_ROOT to the directory that contains `server/`.\n"
        f"  __file__={here!s}  cwd={cwd!s}"
    )

_REPO = _find_repo_root()


def _register_incident_env_pkg(repo: Path) -> None:
    """
    The repo is laid out as a *single* installable tree (``models.py``,
    ``server/``, ``scenarios/``, …) but **without** a physical ``incident_env/``
    directory.  Subpackages use relative imports (e.g. ``from ..models`` in
    ``server/``), so they must be loaded as ``incident_env.server``, not as a
    bare top-level ``server`` (which breaks ``..``).

    Register a synthetic parent package ``incident_env`` whose ``__path__`` is
    the repository root.  Then import only ``incident_env.*`` below.
    """
    import importlib.machinery
    import types

    root = repo.resolve()
    root_s = str(root)
    name = "incident_env"

    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, "__path__", None):
        return

    pkg = types.ModuleType(name)
    pkg.__path__ = [root_s]
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = [root_s]
    pkg.__spec__ = spec
    pkg.__package__ = name
    sys.modules[name] = pkg


_register_incident_env_pkg(_REPO)

from incident_env.server.incident_environment import IncidentEnvironment          # noqa: E402
from incident_env.tasks import compute_r_cross                                    # noqa: E402
from incident_env.pools import POOLS, sample_task                                 # noqa: E402
from incident_env.training.curriculum import CurriculumConfig, CurriculumRunner   # noqa: E402
from incident_env.training.segment_grpo import GRPOGroup, Segment, grpo_advantages  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO training for SRE incident env")
    p.add_argument("--model",              required=True,
                   help="HuggingFace model ID or local path of your SFT checkpoint")
    p.add_argument("--stages",             nargs="+", type=int, default=[2, 3, 4],
                   help="Which curriculum stages to run (2=P1, 3=P2, 4=joint)")
    p.add_argument("--group_size",         type=int, default=4,
                   help="K rollouts per task per GRPO update")
    p.add_argument("--episodes_per_task",  type=int, default=64,
                   help="Total episodes per stage (split across tasks evenly)")
    p.add_argument("--lr",                 type=float, default=1e-5)
    p.add_argument("--beta",               type=float, default=0.04,
                   help="KL penalty coefficient")
    p.add_argument("--clip",               type=float, default=0.2,
                   help="PPO clip epsilon")
    p.add_argument("--max_new_tokens",     type=int, default=512)
    p.add_argument("--max_steps",          type=int, default=40,
                   help="Max environment steps per episode")
    p.add_argument("--output_dir",         default="checkpoints")
    p.add_argument("--save_every",         type=int, default=50,
                   help="Save checkpoint every N gradient steps")
    # LoRA — strongly recommended for A100 40 GB
    p.add_argument("--use_lora",           action="store_true", default=False,
                   help="Wrap trainable model with LoRA adapters (PEFT)")
    p.add_argument("--lora_r",             type=int, default=16)
    p.add_argument("--lora_alpha",         type=int, default=32)
    p.add_argument("--lora_dropout",       type=float, default=0.05)
    p.add_argument("--lora_target",        nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj"],
                   help="Which linear layers to attach LoRA to")
    # Kept for backward compat; on A100 bf16 is preferred
    p.add_argument("--load_in_4bit",       action="store_true", default=False,
                   help="Load base model in 4-bit nf4 (QLoRA) — only needed on <24 GB GPUs")
    p.add_argument("--push_to_hub",        default=None,
                   help="HF repo ID to push final model to (optional)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────

def _load_model(
    model_id: str,
    load_in_4bit: bool,
    trainable: bool,
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target: Optional[List[str]] = None,
) -> AutoModelForCausalLM:
    kwargs: Dict[str, Any] = {"device_map": "auto"}
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    if trainable and use_lora:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        if load_in_4bit:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=True
            )
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target or ["q_proj", "v_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    if trainable:
        model.train()
        if not use_lora:
            model.gradient_checkpointing_enable()
    else:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

    return model


# ──────────────────────────────────────────────────────────────────────
# Inline policy — shares the already-loaded model
# ──────────────────────────────────────────────────────────────────────

_SYS = (
    "You are an SRE incident responder. At each step you receive a JSON "
    "observation and a list of valid_actions. Reply with ONLY a single JSON "
    'object: {"action_type": "...", "target_service": null, "parameters": {...}}. '
    "No prose, no markdown fences."
)


class _InlinePolicy:
    """
    Single-model policy that records the exact assistant text so we can
    recompute logprobs for the same tokens during the GRPO update.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        max_history: int = 8,
    ) -> None:
        self._model = model
        self._tok = tokenizer
        self._max_new = max_new_tokens
        self._temp = temperature
        self._max_hist = max_history
        self._msgs: List[Dict[str, str]] = []
        self.recorded_turns: List[str] = []  # assistant text per turn

    def reset(self, task_name: str = "") -> None:
        self._msgs = [{"role": "system", "content": _SYS}]
        self.recorded_turns = []

    # build plain-text prompt for models without a native chat template
    def _format(self) -> str:
        parts: List[str] = []
        for m in self._msgs:
            role = m["role"]
            if role == "system":
                parts.append(f"<|system|>\n{m['content']}\n")
            elif role == "user":
                parts.append(f"<|user|>\n{m['content']}\n")
            else:
                parts.append(f"<|assistant|>\n{m['content']}\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)

    def __call__(
        self,
        observation: Dict[str, Any],
        phase: int,
        task_name: str,
    ) -> Dict[str, Any]:
        payload = json.dumps({
            "phase": phase,
            "observation": observation,
            "valid_actions": observation.get("valid_actions", []),
        })[:4000]
        self._msgs.append({"role": "user", "content": payload})
        self._trim()

        prompt = self._format()
        try:
            device = self._model.get_input_embeddings().weight.device
        except Exception:
            device = next(self._model.parameters()).device

        inputs = self._tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new,
                do_sample=(self._temp > 0),
                temperature=self._temp,
                eos_token_id=self._tok.eos_token_id,
            )
        text = self._tok.decode(
            out[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        ).strip()

        self._msgs.append({"role": "assistant", "content": text})
        self.recorded_turns.append(text)
        return _parse_action(text)

    def _trim(self) -> None:
        if len(self._msgs) > 1 + self._max_hist * 2:
            self._msgs = self._msgs[:1] + self._msgs[-(self._max_hist * 2):]


def _parse_action(text: str) -> Dict[str, Any]:
    text = text.strip().lstrip("`")
    if text.startswith("json"):
        text = text[4:]
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b <= a:
        return {"action_type": "declare_no_change", "parameters": {}}
    try:
        obj = json.loads(text[a : b + 1])
        obj.setdefault("parameters", {})
        return obj if isinstance(obj, dict) else {"action_type": "declare_no_change", "parameters": {}}
    except Exception:
        return {"action_type": "declare_no_change", "parameters": {}}


# ──────────────────────────────────────────────────────────────────────
# Rollout collection — K episodes → one GRPO group
# ──────────────────────────────────────────────────────────────────────

def _collect_group(
    env: IncidentEnvironment,
    policy: _InlinePolicy,
    task_name: str,
    pool: str,
    mode: str,
    group_size: int,
    max_steps: int,
    stage: int,
    group_idx: int,
) -> Tuple[GRPOGroup, List[List[str]]]:
    """
    Run `group_size` episodes for `task_name`.
    Returns (GRPOGroup, recorded_turns_per_episode).
    GRPOGroup uses final score as terminal_reward.
    """
    segments: List[Segment] = []
    turns_per_ep: List[List[str]] = []

    for k in range(group_size):
        policy.reset(task_name)
        seed = stage * 100_000 + group_idx * group_size + k

        info = env.reset(task_name=task_name, pool=pool, mode=mode, seed=seed)
        obs = info["observation"]

        for _ in range(max_steps):
            phase = obs.get("current_phase", 1)
            action = policy(obs, phase, task_name)
            step_out = env.step(action)
            obs = step_out["observation"]
            if step_out.get("done"):
                break

        breakdown = env.score_unified()
        final = float(breakdown.get("final", 0.0))

        r_cross = 0.0
        try:
            state = env.get_state()
            r_cross = float(compute_r_cross(
                task_name=task_name,
                declared_patch=state.get("declared_patch"),
                declared_no_change=bool(state.get("declared_no_change")),
                p2_trajectory=env.get_p2_trajectory(),
            ))
        except Exception:
            pass

        segments.append(Segment(
            segment_id=f"{task_name}_g{group_idx}_k{k}",
            phase=1,  # whole episode treated as one segment for simplicity
            trajectory=[],
            terminal_reward=final,
            r_cross=r_cross,
        ))
        turns_per_ep.append(list(policy.recorded_turns))

    group = GRPOGroup(prompt_id=f"{task_name}_g{group_idx}", segments=segments)
    return group, turns_per_ep


# ──────────────────────────────────────────────────────────────────────
# Per-token logprob computation for assistant turns
# ──────────────────────────────────────────────────────────────────────

def _assistant_logprobs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    turns: List[str],
    no_grad: bool = False,
) -> List[torch.Tensor]:
    """
    For each assistant turn (a short JSON string), compute per-token
    log-probs under `model`.  Returned as a list of 1-D tensors.
    If the turn is empty, returns an empty tensor.
    """
    out: List[torch.Tensor] = []
    try:
        device = model.get_input_embeddings().weight.device
    except Exception:
        device = next(model.parameters()).device

    for turn in turns:
        if not turn.strip():
            out.append(torch.tensor([], device=device))
            continue

        ids = tokenizer(turn, return_tensors="pt", truncation=True,
                        max_length=512)["input_ids"].to(device)
        if ids.shape[1] < 2:
            out.append(torch.tensor([], device=device))
            continue

        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            logits = model(input_ids=ids).logits[0]         # (L, V)
        lp = F.log_softmax(logits, dim=-1)                  # (L, V)
        # logprob of token t is lp[t-1, id[t]]
        seq_lp = lp[:-1].gather(1, ids[0, 1:].unsqueeze(1)).squeeze(1)
        out.append(seq_lp)
    return out


# ──────────────────────────────────────────────────────────────────────
# GRPO update step
# ──────────────────────────────────────────────────────────────────────

def _grpo_step(
    model: AutoModelForCausalLM,
    ref_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    optimizer: AdamW,
    group: GRPOGroup,
    turns_per_ep: List[List[str]],
    beta: float,
    clip: float,
) -> float:
    advantages = grpo_advantages(group)
    if not advantages or all(abs(a) < 1e-8 for a in advantages):
        return 0.0  # uniform reward — zero gradient, skip

    optimizer.zero_grad()
    total_loss = torch.tensor(0.0, device="cpu", requires_grad=False)
    n_tokens = 0

    for i, (turns, adv) in enumerate(zip(turns_per_ep, advantages)):
        if not turns:
            continue

        # Policy logprobs (with grad)
        policy_lps = _assistant_logprobs(model, tokenizer, turns, no_grad=False)
        # Reference logprobs (no grad)
        ref_lps    = _assistant_logprobs(ref_model, tokenizer, turns, no_grad=True)

        ep_loss = torch.tensor(0.0)
        ep_tokens = 0

        for plp, rlp in zip(policy_lps, ref_lps):
            if plp.numel() == 0:
                continue
            if rlp.shape != plp.shape:
                min_len = min(plp.shape[0], rlp.shape[0])
                plp = plp[:min_len]
                rlp = rlp[:min_len]

            ratio     = torch.exp(plp - rlp.detach())
            unclipped = ratio * adv
            clipped   = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
            pg_loss   = -torch.min(unclipped, clipped)           # (L,)
            kl_loss   = beta * (rlp.detach() - plp)              # (L,)

            ep_loss   = ep_loss + (pg_loss + kl_loss).sum()
            ep_tokens += plp.numel()

        if ep_tokens > 0:
            ep_loss = ep_loss / ep_tokens
            ep_loss.backward()
            total_loss = total_loss + ep_loss.detach()
            n_tokens += ep_tokens

    if n_tokens > 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()

    return (total_loss / max(len(turns_per_ep), 1)).item()


# ──────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────

STAGE_POOL = {2: ("A", "p1_only"), 3: ("B", "p2_only"), 4: ("C", "joint")}


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading tokenizer + policy model from: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy_model = _load_model(
        args.model, args.load_in_4bit, trainable=True,
        use_lora=args.use_lora, lora_r=args.lora_r,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        lora_target=args.lora_target,
    )
    ref_model = _load_model(args.model, args.load_in_4bit, trainable=False)
    print("Models loaded.\n")

    optimizer = AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    policy = _InlinePolicy(policy_model, tokenizer, args.max_new_tokens)
    env    = IncidentEnvironment()

    cfg    = CurriculumConfig(n_per_stage=args.episodes_per_task)
    runner = CurriculumRunner(cfg=cfg)   # variance gate + warmup scheduler

    global_step = 0

    for stage in args.stages:
        pool_name, mode = STAGE_POOL[stage]
        tasks = list(POOLS[pool_name].task_names)
        n_groups = max(1, args.episodes_per_task // args.group_size)

        print(f"{'='*60}")
        print(f"Stage {stage} | Pool {pool_name} | mode={mode} | "
              f"{n_groups} groups × {args.group_size} rollouts")
        print(f"{'='*60}")

        # Stage 3 prerequisite: calibrate null-context P2 baseline for r_cross
        if stage == 3:
            print("Calibrating Pool-B null-context baseline (needed for r_cross)...")
            runner.measure_pool_b_baselines(samples=4)
            print("Done.\n")

        if stage == 4 and not runner.should_enter_stage4():
            print("Note: variance gate not yet open — r_cross may be noisy.\n")

        for g in tqdm(range(n_groups), desc=f"Stage {stage}"):
            task = tasks[g % len(tasks)]
            r_cross_w = runner.r_cross_weight() if stage == 4 else 0.0

            # ── collect K rollouts ──────────────────────────────────────
            group, turns_per_ep = _collect_group(
                env=env,
                policy=policy,
                task_name=task,
                pool=pool_name,
                mode=mode,
                group_size=args.group_size,
                max_steps=args.max_steps,
                stage=stage,
                group_idx=g,
            )

            # Apply r_cross weight to P1 advantage signal in Stage 4
            if stage == 4 and r_cross_w > 0:
                for seg in group.segments:
                    seg.r_cross *= r_cross_w

            # ── GRPO update ─────────────────────────────────────────────
            # Skip P1 update in Stage 3 (p2_only) and P2 update in Stage 2 (p1_only)
            skip_update = (stage == 3 and mode == "p2_only" and False)  # always train
            loss = _grpo_step(
                policy_model, ref_model, tokenizer, optimizer,
                group, turns_per_ep, args.beta, args.clip,
            )

            rewards = [s.terminal_reward for s in group.segments]
            avg_r   = sum(rewards) / max(len(rewards), 1)
            r_str   = " ".join(f"{r:.3f}" for r in rewards)

            tqdm.write(
                f"  step {global_step:4d} | {task:30s} | "
                f"avg_r={avg_r:.3f} [{r_str}] | loss={loss:.4f} | r_cross_w={r_cross_w:.2f}"
            )

            # Update variance gate with latest r_cross values
            for seg in group.segments:
                runner.gate.record(task, seg.r_cross)
            if stage == 4:
                runner.stage4_step_counter += 1

            # ── checkpoint ──────────────────────────────────────────────
            if global_step > 0 and global_step % args.save_every == 0:
                ckpt = out_dir / f"stage{stage}_step{global_step}"
                policy_model.save_pretrained(str(ckpt))
                tokenizer.save_pretrained(str(ckpt))
                tqdm.write(f"  Checkpoint saved → {ckpt}")

            global_step += 1

        # Stage-end checkpoint
        ckpt = out_dir / f"stage{stage}_final"
        policy_model.save_pretrained(str(ckpt))
        tokenizer.save_pretrained(str(ckpt))
        print(f"\nStage {stage} complete → {ckpt}\n")

    # ── held-out evaluation on Pool D ───────────────────────────────
    print("Evaluating on held-out Pool D...")
    policy.reset()
    runner.p1_policy = policy
    runner.p2_policy = policy
    eval_results = runner.evaluate_held_out(n_per_task=4)
    eval_path = out_dir / "eval_pool_d.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2, default=str)
    print(f"Pool D results → {eval_path}")

    # ── optional HF Hub push ────────────────────────────────────────
    if args.push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {args.push_to_hub}")
        if args.use_lora:
            # Merge LoRA weights into base before pushing so the result is
            # a standard causal LM that any inference stack can load.
            from peft import PeftModel
            print("  Merging LoRA adapters into base weights...")
            merged = policy_model.merge_and_unload()
            merged.push_to_hub(args.push_to_hub)
        else:
            policy_model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)
        print("Done.")


if __name__ == "__main__":
    main()
