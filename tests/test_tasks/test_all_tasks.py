import pytest

from tasks.all_tasks import get_task


class TestAllTasks:
    def test_valid_task(self):
        task = get_task("sin_q")
        assert task.__name__ == "SineQ"

    def test_invalid_task(self):
        with pytest.raises(ValueError):
            task = get_task("sin_qq")
