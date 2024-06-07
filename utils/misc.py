from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Player:
    track_id: int
    class_id: int
    label: str
    bbox: np.array
    team_id: Optional[int] = None


    def __eq__(self, other: 'Player'):
        if self.track_id != other.track_id:
            return False
        if self.class_id != other.class_id:
            return False
        if self.team_id != other.team_id:
            return False
        return True

    @property
    def centroid(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return int(x1+x2/2), int(y1+y2/2)
    
    @property
    def box_dims(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return (x2-x1)//2, (y2-y1)//2

    def distance(self, point: Tuple[int, int]) -> float:
        x0, y0 = self.centroid
        x1, y1 = point
        return ((x0-x1)*(x0-x1) + (y0-y1)*(y0-y1))**(1/2)
    
    def overlaps(self, other: 'Player', th: float = .5) -> bool:
        dist = self.distance(other.centroid)
        if dist < th*(min(self.box_dims) + min(other.box_dims)):
            return True
        return False
