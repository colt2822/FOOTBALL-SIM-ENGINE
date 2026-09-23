from pydantic import Field
from .schemas import Frozen

class GameState(Frozen):
    """Block-entry state; transitions create a new value. Scores drive later script."""
    game_id: str
    simulation_id: int = Field(ge=0)
    block_index: int = Field(ge=0)
    seconds_remaining: int = Field(ge=0)
    home_score: int = Field(ge=0)
    away_score: int = Field(ge=0)
    possession: str
    field_position: int = Field(ge=0, le=100)
    overtime: bool = False

    def advance(self, *, seconds: int, possession: str | None = None,
                field_position: int | None = None, block_index: int | None = None,
                home_score: int | None = None, away_score: int | None = None) -> "GameState":
        return self.model_copy(update={
            "seconds_remaining": max(0, self.seconds_remaining - max(0, int(seconds))),
            "possession": possession or self.possession,
            "field_position": self.field_position if field_position is None else min(100, max(0, int(field_position))),
            "block_index": self.block_index + 1 if block_index is None else max(0, int(block_index)),
            "home_score": self.home_score if home_score is None else max(0, int(home_score)),
            "away_score": self.away_score if away_score is None else max(0, int(away_score)),
        })
