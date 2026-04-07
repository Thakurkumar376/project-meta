# OpenEnv Customer Support Evaluation Environment

## Project Overview
`openenv_customer_support` is a complete OpenEnv-compatible benchmark project for the **Meta OpenEnv Hackathon**. It evaluates LLM agents on practical customer support workflows with increasing difficulty.

It includes:
- A clean OpenEnv environment API (`reset`, `step`, `state`)
- Three practical tasks (easy → medium → hard)
- Programmatic graders returning scores from `0.0` to `1.0`
- Reward shaping + penalties (including repetition/loop controls)
- A reproducible OpenAI-compatible baseline script routed through Hugging Face
- Docker and Hugging Face Space compatibility (`openenv` tag)

## Motivation
Real support agents must do more than one-shot QA. This benchmark evaluates:
1. Correct routing of inbound customer messages
2. Quality of support writing in a single response
3. Consistency and usefulness through a multi-turn escalation

## OpenEnv API
Implemented in `openenv_customer_support/env/environment.py`.

- `reset() -> Observation`
- `step(action: Action) -> (Observation, Reward, done, info)`
- `state() -> dict`

### Action Space
```python
Action(content: str)
```

### Observation Space
```python
Observation(
  task_type: TaskType,
  prompt: str,
  turn_index: int,
  metadata: dict
)
```

### Reward Space
```python
Reward(score: float, reasoning: str)
```

## Tasks

### Task 1 (Easy): Email Classification
- **Input:** raw email text
- **Output:** one of `support`, `refund`, `complaint`, `order status`, `spam`
- **Primary metric:** exact classification accuracy

### Task 2 (Medium): Customer Support Response Generation
- **Input:** complaint message
- **Output:** professional support response
- **Primary metric:** relevance + empathy + completeness

### Task 3 (Hard): Multi-turn Support Conversation
- **Input:** escalating multi-turn conversation
- **Output:** helpful replies each turn
- **Primary metric:** cross-turn conversation quality (coverage, completion, anti-loop)

## Reward Function
Positive rewards for:
- Correct classification
- Helpful and relevant support responses
- Successfully completing the conversation with quality

Penalties for:
- Wrong classification
- Irrelevant or low-quality responses
- Repetition/infinite-loop behavior
- Invalid usage (e.g., empty action, stepping after done)

## Project Structure
```text
openenv_customer_support/
│
├── env/
│   └── environment.py
├── tasks/
│   ├── email_classification.py
│   ├── response_generation.py
│   └── conversation_task.py
├── graders/
│   ├── email_grader.py
│   ├── response_grader.py
│   └── conversation_grader.py
├── scripts/
│   └── run_baseline.py
├── tests/
│   ├── test_environment.py
│   └── test_graders.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup Instructions
From repository root (`/workspace/project-meta`):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r openenv_customer_support/requirements.txt
```

### Authentication (Important)
Set `HF_TOKEN` to your **Hugging Face User Access Token**. The baseline script uses this token with Hugging Face's OpenAI-compatible router endpoint.

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxx"
```

Optional overrides:
```bash
export OPENAI_MODEL="openai/gpt-4o-mini"
export HF_OPENAI_BASE_URL="https://router.huggingface.co/v1"
```

Run baseline:
```bash
python -m openenv_customer_support.scripts.run_baseline
```

Save JSON results:
```bash
python -m openenv_customer_support.scripts.run_baseline --output-json results/baseline.json
```

Run tests:
```bash
pytest openenv_customer_support/tests -q
```

## Docker
Build:
```bash
docker build -t openenv-customer-support ./openenv_customer_support
```

Run:
```bash
docker run --rm -e HF_TOKEN="$HF_TOKEN" -e OPENAI_MODEL="openai/gpt-4o-mini" openenv-customer-support
```

## Hugging Face Space Deployment
This repo is Docker-compatible and includes OpenEnv metadata in `openenv.yaml`. For Space deployment:
- SDK: Docker
- Secret: `HF_TOKEN`
- Suggested tag: `openenv`

## Baseline Results (Expected Format)
Your exact values depend on model/version and provider availability. Output format:

```text
Task 1 Score: 0.85
Task 2 Score: 0.78
Task 3 Score: 0.66
Average Score: 0.76
```
