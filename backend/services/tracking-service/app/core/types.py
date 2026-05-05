from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_xyxy(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @classmethod
    def from_xyxy(cls, data: List[float]) -> "BoundingBox":
        if len(data) < 4:
            raise ValueError("BoundingBox requires [x1, y1, x2, y2]")
        return cls(x1=float(data[0]), y1=float(data[1]), x2=float(data[2]), y2=float(data[3]))
