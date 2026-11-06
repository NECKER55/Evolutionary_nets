"""Il cervello ragionante (docs/SPEC_CERVELLO_RAGIONANTE.md): un cervello nuovo,
che non usa niente del cervello a sezioni. Per ora esiste solo la sezione 7."""

from .brain import Reasoning
from .motor import BIRTH, HIDDEN, INTENTS, RETUNE, Motor

__all__ = ["BIRTH", "HIDDEN", "INTENTS", "Motor", "RETUNE", "Reasoning"]
