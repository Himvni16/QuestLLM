"""Tests for the Phase 1 configuration surface."""

from questllm.config import APP_NAME, get_settings


def test_settings_use_the_questllm_name() -> None:
    """The configured application name remains consistent."""

    assert APP_NAME == "QuestLLM"
    assert get_settings().app_name == "QuestLLM"
