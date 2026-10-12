"""Evoluzione: come nascono i figli (incrocio, mutazione) e il ciclo delle
generazioni. Lavora sui genomi, non sa come la rete calcola."""

from .evolution import Evolution
from .genetics import crossover, mutate, reproduce

__all__ = ["Evolution", "crossover", "mutate", "reproduce"]
