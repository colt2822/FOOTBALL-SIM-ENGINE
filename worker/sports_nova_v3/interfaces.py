"""Minimal stage interfaces; named modules and detailed semantics are in the plan."""
from typing import Protocol, Any
from .schemas import PregameState
from .game_state import GameState

class GameEnvironment(Protocol):
    def draw(self, pregame: PregameState, parameters: Any, rng: Any) -> Any: ...

class GameScript(Protocol):
    def rates(self, pregame: PregameState, state: GameState, environment: Any) -> Any: ...

class PlayVolume(Protocol):
    def draw(self, state: GameState, rates: Any, rng: Any) -> Any: ...

class PlaySelection(Protocol):
    def draw(self, volume: Any, rates: Any, rng: Any) -> Any: ...

class OpportunityAllocation(Protocol):
    def draw(self, pregame: PregameState, selection: Any, rng: Any) -> Any: ...

class Efficiency(Protocol):
    def draw(self, allocation: Any, state: GameState, rng: Any) -> Any: ...

class Scoring(Protocol):
    def transition(self, state: GameState, production: Any, rng: Any) -> Any: ...

class SimulationEngine(Protocol):
    def simulate_game(self, pregame_state: PregameState, n_sims: int, seed: int, model_version: str) -> Any: ...

class JointQueryEngine(Protocol):
    def query(self, samples: Any, predicates: tuple, given: tuple = ()) -> Any: ...

class Calibration(Protocol):
    def fit(self, prior_oos: Any, cutoff: Any) -> Any: ...

class Validation(Protocol):
    def walk_forward(self, dataset: Any, frozen_protocol: Any) -> Any: ...

class ArtifactFreeze(Protocol):
    def freeze(self, samples: Any, model: Any, runtime: Any) -> Any: ...
