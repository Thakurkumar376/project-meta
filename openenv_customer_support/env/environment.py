"""OpenEnv-compatible customer support evaluation environment."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from openenv_customer_support.graders.conversation_grader import grade_conversation_quality
from openenv_customer_support.graders.email_grader import grade_email_classification
from openenv_customer_support.graders.response_grader import grade_response_quality
from openenv_customer_support.tasks.conversation_task import CONVERSATION_SCENARIOS
from openenv_customer_support.tasks.email_classification import EMAIL_CLASSIFICATION_DATASET, VALID_EMAIL_CATEGORIES
from openenv_customer_support.tasks.response_generation import RESPONSE_GENERATION_DATASET


class TaskType(str, Enum):
    """Supported tasks."""

    email_classification = "email_classification"
    response_generation = "response_generation"
    conversation = "conversation"


class Observation(BaseModel):
    """Environment observation schema."""

    task_type: TaskType
    prompt: str
    turn_index: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Action(BaseModel):
    """Agent action schema."""

    content: str


class Reward(BaseModel):
    """Reward schema returned by step()."""

    score: float = Field(ge=-1.0, le=1.0)
    reasoning: str


class OpenEnvCustomerSupportEnv:
    """Real-world customer support OpenEnv evaluation environment."""

    def __init__(self, task_type: TaskType):
        self.task_type = task_type
        self.pointer = 0
        self.turn_index = 0
        self.done = False
        self._responses: List[str] = []

    def reset(self) -> Observation:
        """Return the initial observation for selected task."""

        self.pointer = 0
        self.turn_index = 0
        self.done = False
        self._responses = []
        return self._current_observation()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, Dict[str, Any]]:
        """Apply agent action and return OpenEnv tuple (observation, reward, done, info)."""

        if self.done:
            return self._current_observation(), Reward(score=-0.2, reasoning="Episode already completed."), True, {
                "warning": "step_called_after_done"
            }

        content = action.content.strip()
        if not content:
            reward = Reward(score=-0.3, reasoning="Empty action received.")
            return self._current_observation(), reward, False, {"error": "empty_action"}

        info: Dict[str, Any] = {}

        if self.task_type == TaskType.email_classification:
            example = EMAIL_CLASSIFICATION_DATASET[self.pointer]
            base_score = grade_email_classification(content, example.gold_category)
            reward = Reward(
                score=base_score if base_score > 0 else -0.2,
                reasoning=f"Predicted='{content}', Gold='{example.gold_category}', base_score={base_score}",
            )
            info.update({"gold_category": example.gold_category, "prediction": content.lower().strip()})
            self.pointer += 1
            self.done = self.pointer >= len(EMAIL_CLASSIFICATION_DATASET)

        elif self.task_type == TaskType.response_generation:
            example = RESPONSE_GENERATION_DATASET[self.pointer]
            base_score = grade_response_quality(content, example.required_keywords)
            low_quality_penalty = -0.1 if base_score < 0.3 else 0.0
            reward = Reward(
                score=max(-1.0, base_score + low_quality_penalty),
                reasoning=f"response_quality_score={base_score}, low_quality_penalty={low_quality_penalty}",
            )
            info.update({"required_keywords": example.required_keywords})
            self.pointer += 1
            self.done = self.pointer >= len(RESPONSE_GENERATION_DATASET)

        else:
            scenario = CONVERSATION_SCENARIOS[0]
            self._responses.append(content)
            self.turn_index += 1

            if self.turn_index >= len(scenario.turns):
                base_score = grade_conversation_quality(self._responses, scenario.expected_elements)
                reward_value = base_score if base_score >= 0.2 else -0.2
                reward = Reward(
                    score=reward_value,
                    reasoning=f"conversation_quality_score={base_score}, final_reward={reward_value}",
                )
                self.done = True
            else:
                repeated_reply = self._responses.count(content) > 1
                shaping = -0.15 if repeated_reply else 0.05
                reward = Reward(score=shaping, reasoning="Intermediate shaping reward for non-repetition.")
                info["repeated_reply"] = repeated_reply

        next_observation = self._current_observation()
        if self.done:
            info["episode_complete"] = True

        return next_observation, reward, self.done, info

    def state(self) -> Dict[str, Any]:
        """Return serializable internal environment state."""

        return {
            "task_type": self.task_type.value,
            "pointer": self.pointer,
            "turn_index": self.turn_index,
            "done": self.done,
            "responses": list(self._responses),
        }

    def _current_observation(self) -> Observation:
        """Build current observation."""

        if self.task_type == TaskType.email_classification:
            if self.pointer >= len(EMAIL_CLASSIFICATION_DATASET):
                return Observation(task_type=self.task_type, prompt="All email samples evaluated.")
            sample = EMAIL_CLASSIFICATION_DATASET[self.pointer]
            return Observation(
                task_type=self.task_type,
                prompt=sample.prompt,
                turn_index=self.pointer,
                metadata={"valid_categories": VALID_EMAIL_CATEGORIES},
            )

        if self.task_type == TaskType.response_generation:
            if self.pointer >= len(RESPONSE_GENERATION_DATASET):
                return Observation(task_type=self.task_type, prompt="All complaint prompts evaluated.")
            sample = RESPONSE_GENERATION_DATASET[self.pointer]
            return Observation(task_type=self.task_type, prompt=sample.complaint, turn_index=self.pointer)

        scenario = CONVERSATION_SCENARIOS[0]
        if self.turn_index == 0:
            prompt = (
                f"Scenario: {scenario.opening_context}\n"
                f"Customer: {scenario.turns[0].customer_message}"
            )
        elif self.turn_index < len(scenario.turns):
            prompt = f"Customer: {scenario.turns[self.turn_index].customer_message}"
        else:
            prompt = "Conversation complete."

        return Observation(
            task_type=self.task_type,
            prompt=prompt,
            turn_index=self.turn_index,
            metadata={"max_turns": len(scenario.turns), "scenario_id": scenario.scenario_id},
        )


def create_environment(task_name: str) -> OpenEnvCustomerSupportEnv:
    """Factory helper for scripts and external runners."""

    mapping = {
        "email_classification": TaskType.email_classification,
        "response_generation": TaskType.response_generation,
        "conversation": TaskType.conversation,
    }
    if task_name not in mapping:
        raise ValueError(f"Unknown task '{task_name}'. Expected one of: {list(mapping)}")

    return OpenEnvCustomerSupportEnv(task_type=mapping[task_name])
