"""Module that tests the attitude tasks."""

from envs.citation.citation_env import CitationEnv
from tasks.all_tasks import get_task
from tasks.attitude import AttitudeTrain


class TestAttitudeTrain:
    """Test the attitude train task."""

    def test_get_task(self):
        """Test if the correct tasks is returned."""
        task = get_task("att_train")
        assert task == AttitudeTrain

    def test_valid_for_env(self):
        """Test if the task is valid for the environment."""
        env = CitationEnv()
        task = AttitudeTrain(env)
        assert task.valid_for_env(env)

    def test_reference(self):
        """Test the reference signal."""
        env = CitationEnv()
        task = AttitudeTrain(env)
        assert task.reference().shape == (3,)
