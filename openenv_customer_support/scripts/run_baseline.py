"""Baseline runner that evaluates an OpenEnv agent policy using OpenAI-compatible chat completions via Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean
from typing import Dict

from openai import OpenAI

from openenv_customer_support.env.environment import Action, create_environment

DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_HF_OPENAI_BASE_URL = "https://router.huggingface.co/v1"


def get_client(base_url: str) -> OpenAI:
    """Create OpenAI-compatible client using Hugging Face token from HF_TOKEN.

    Notes:
    - HF_TOKEN must contain a Hugging Face user access token.
    - Requests are sent to the Hugging Face OpenAI-compatible router endpoint.
    """

    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError(
            "HF_TOKEN is not set. Export your Hugging Face User Access Token as HF_TOKEN."
        )

    return OpenAI(api_key=token, base_url=base_url)


def ask_model(client: OpenAI, model_name: str, system_prompt: str, user_prompt: str) -> str:
    """Get model response text."""

    completion = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (completion.choices[0].message.content or "").strip()


def run_task(task_name: str, client: OpenAI, model_name: str) -> float:
    """Run one task episode and return average reward score."""

    env = create_environment(task_name)
    observation = env.reset()
    done = False
    rewards = []
    max_steps = 12
    steps = 0

    if task_name == "email_classification":
        system_prompt = (
            "You are a strict classifier. Respond with exactly one category: "
            "support, refund, complaint, order status, or spam."
        )
    elif task_name == "response_generation":
        system_prompt = (
            "You are a professional customer support agent. Write concise, empathetic, and actionable responses."
        )
    else:
        system_prompt = (
            "You are handling an escalating multi-turn support conversation. "
            "Be consistent, empathetic, specific, and avoid repetition."
        )

    while not done and steps < max_steps:
        model_output = ask_model(client, model_name, system_prompt, observation.prompt)
        observation, reward, done, _ = env.step(Action(content=model_output))
        rewards.append(reward.score)
        steps += 1

    if steps >= max_steps and not done:
        rewards.append(-0.5)  # infinite-loop guard penalty

    return round(mean(rewards), 4) if rewards else 0.0


def run_baseline(model_name: str, base_url: str) -> Dict[str, float]:
    """Run baseline across all tasks and return score dictionary."""

    client = get_client(base_url=base_url)

    task_scores = {
        "Task 1 Score": run_task("email_classification", client, model_name),
        "Task 2 Score": run_task("response_generation", client, model_name),
        "Task 3 Score": run_task("conversation", client, model_name),
    }
    task_scores["Average Score"] = round(mean(task_scores.values()), 4)
    return task_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenEnv customer-support baseline evaluation.")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL), help="Model name")
    parser.add_argument(
        "--base-url",
        default=os.getenv("HF_OPENAI_BASE_URL", DEFAULT_HF_OPENAI_BASE_URL),
        help="OpenAI-compatible endpoint base URL (defaults to Hugging Face router)",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to write scores as JSON (for reproducibility tracking).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = run_baseline(model_name=args.model, base_url=args.base_url)
    for name, score in scores.items():
        print(f"{name}: {score:.2f}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"model": args.model, "base_url": args.base_url, "scores": scores}, indent=2),
            encoding="utf-8",
        )
        print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
