"""Evoluzione: come nascono i figli (incrocio, mutazione) e, più avanti, il
ciclo delle generazioni. Lavora sui genomi, non sa come la rete calcola."""

from .genetics import crossover, mutate, reproduce

__all__ = ["crossover", "mutate", "reproduce"]
