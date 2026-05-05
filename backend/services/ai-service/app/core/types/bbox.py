from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0
    class_id: int = 0
    class_name: str = "person"

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def to_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2, self.confidence]

    @classmethod
    def from_list(cls, data: List[float]) -> "BBox":
        if len(data) < 5:
            raise ValueError("BBox requires at least 5 values [x1, y1, x2, y2, conf]")
        return cls(
            x1=float(data[0]),
            y1=float(data[1]),
            x2=float(data[2]),
            y2=float(data[3]),
            confidence=float(data[4]),
            class_id=int(data[5]) if len(data) > 5 else 0,
        )

    def iou_with(self, other: "BBox") -> float:
        xi1 = max(self.x1, other.x1)
        yi1 = max(self.y1, other.y1)
        xi2 = min(self.x2, other.x2)
        yi2 = min(self.y2, other.y2)

        inter_w = max(0, xi2 - xi1)
        inter_h = max(0, yi2 - yi1)
        inter_area = inter_w * inter_h

        self_area = self.area
        other_area = other.area
        union_area = self_area + other_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0
