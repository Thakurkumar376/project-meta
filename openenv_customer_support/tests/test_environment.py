from openenv_customer_support.env import Action, create_environment


def test_email_task_episode_completes():
    env = create_environment("email_classification")
    obs = env.reset()
    assert "prompt" in obs.model_dump()

    done = False
    while not done:
        obs, reward, done, info = env.step(Action(content="support"))
        assert -1.0 <= reward.score <= 1.0

    assert env.state()["done"] is True


def test_empty_action_penalty():
    env = create_environment("response_generation")
    env.reset()
    _, reward, done, info = env.step(Action(content="   "))

    assert reward.score < 0
    assert done is False
    assert info.get("error") == "empty_action"
