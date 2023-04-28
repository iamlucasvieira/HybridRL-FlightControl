"""Module that defines the Base class for callbacks."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional


# Only import BaseAgent if type checking
if TYPE_CHECKING:
    from agents.base_agent import BaseAgent


class BaseCallback(ABC):
    """Base class for callbacks."""

    def __init__(self, verbose: int = 0):
        """Initialize the callback."""
        super().__init__()
        self.verbose = verbose
        self.agent = None
        self.env = None
        self.num_steps = 0
        self.n_calls = 0
        self.locals: Dict[str, Any] = {}
        self.globals: Dict[str, Any] = {}

    def init_callback(self, agent: "BaseAgent") -> None:
        """Initialize the callback."""
        self.agent = agent
        self.env = agent.env
        self._init_callback()

    def _init_callback(self) -> None:
        """Initialize the callback."""
        pass

    def on_training_start(
        self, locals_: Dict[str, Any], globals_: Dict[str, Any]
    ) -> None:
        """Method called before training for callbacks."""
        self.locals = locals_
        self.globals = globals_

        self.num_steps = self.agent.num_steps
        self._on_training_start()

    def on_step(self) -> bool:
        """Method called before each step for callbacks."""
        self.n_calls += 1
        self.num_steps = self.agent.num_steps
        return self._on_step()

    def on_training_end(self) -> None:
        """Method called after training for callbacks."""
        self._on_training_end()

    def on_rollout_start(self) -> None:
        """Method called before rollout for callbacks."""
        self._on_rollout_start()

    def on_rollout_end(self) -> None:
        """Method called after rollout for callbacks."""
        self._on_rollout_end()

    def on_episode_end(self, episode_return) -> None:
        """Method called after each episode for callbacks."""
        self._on_episode_end(episode_return)

    def _on_training_start(self) -> None:
        """Method called before training for callbacks."""
        pass

    def _on_rollout_start(self) -> None:
        """Method called before rollout for callbacks."""
        pass

    def _on_rollout_end(self) -> None:
        """Method called after rollout for callbacks."""
        pass

    @abstractmethod
    def _on_step(self) -> bool:
        """Method called before each step for callbacks."""
        return True

    def _on_training_end(self) -> None:
        """Method called after training for callbacks."""
        pass

    def _on_episode_end(self, episode_return) -> None:
        """Method called after each episode for callbacks."""
        pass


class ListCallback(BaseCallback):
    """Class that implements a list of callbacks."""

    def __init__(self, callbacks: Optional[List[BaseCallback]], verbose: int = 0):
        """Initialize the callback list."""
        super().__init__(verbose=verbose)
        self.callbacks = callbacks if callbacks is not None else []

    def _init_callback(self) -> None:
        """Initialize the callback."""
        for callback in self.callbacks:
            callback.init_callback(self.agent)

    def _on_training_start(self) -> None:
        """Method called before training for callbacks."""
        for callback in self.callbacks:
            callback.on_training_start(self.locals, self.globals)

    def _on_rollout_start(self) -> None:
        """Method called before rollout for callbacks."""
        for callback in self.callbacks:
            callback.on_rollout_start()

    def _on_step(self) -> bool:
        """Method called before each step for callbacks."""
        for callback in self.callbacks:
            if not callback.on_step():
                return False
        return True

    def _on_rollout_end(self) -> None:
        """Method called after rollout for callbacks."""
        for callback in self.callbacks:
            callback.on_rollout_end()

    def _on_training_end(self) -> None:
        """Method called after training for callbacks."""
        for callback in self.callbacks:
            callback.on_training_end()

    def _on_episode_end(self, episode_return) -> None:
        """Method called after each episode for callbacks."""
        for callback in self.callbacks:
            callback.on_episode_end(episode_return)
