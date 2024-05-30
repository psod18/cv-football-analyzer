from dataclasses import dataclass

import numpy as np


@dataclass
class Pawn:
    track_id: int
    class_id: int
    label: str
    bbox: np.array
    team_id: int


