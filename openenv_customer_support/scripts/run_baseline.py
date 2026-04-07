"""Baseline runner that evaluates an OpenAI model across all OpenEnv tasks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean
from typing import Dict

from openai import OpenAI

from openenv_customer_support.env.environment import Action, create_environment

DEFAULT_MODEL = "gpt-4o-mini"


def get_client() -> OpenAI:
    """Create OpenAI client from HF_TOKEN environment variable."""

    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN is not set. Please export HF_TOKEN with your OpenAI API key.")
    return OpenAI(api_key=token)


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


def run_baseline(model_name: str) -> Dict[str, float]:
    """Run baseline across all tasks and return score dictionary."""

    client = get_client()

    task_scores = {
        "Task 1 Score": run_task("email_classification", client, model_name),
        "Task 2 Score": run_task("response_generation", client, model_name),
        "Task 3 Score": run_task("conversation", client, model_name),
    }
    task_scores["Average Score"] = round(mean(task_scores.values()), 4)
    return task_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenEnv customer-support baseline evaluation.")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL), help="OpenAI model name")
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to write scores as JSON (for reproducibility tracking).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = run_baseline(model_name=args.model)
    for name, score in scores.items():
        print(f"{name}: {score:.2f}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"model": args.model, "scores": scores}, indent=2), encoding="utf-8")
        print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
