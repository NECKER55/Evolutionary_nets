"""Gli agenti di controllo del mondo 3D: non sono cervelli e non imparano. Il
base è il pavimento, il perfetto il soffitto: un cervello si misura tra i due."""

from .base import Base
from .perfect import Perfect

CONTROLS = {"base": Base, "perfetto": Perfect}

__all__ = ["CONTROLS", "Base", "Perfect"]
