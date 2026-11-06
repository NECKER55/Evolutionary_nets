import os

# JAX di solito si riserva subito il 75% della memoria della scheda video: un
# secondo processo (un'analisi, un'altra corsa) resterebbe senza. Qui la prende
# man mano che serve. Va deciso prima che JAX venga importato; chi vuole il
# comportamento di JAX può impostare la variabile da fuori.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Senza il driver NVIDIA caricato, JAX prova CUDA, fallisce con un errore lungo e
# poi usa comunque la CPU: qui si va diritti sulla CPU. Col driver si usa la scheda.
if not os.path.exists("/proc/driver/nvidia"):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

from .arenas.generational.world import World  # noqa: E402
from .config import Config  # noqa: E402

__all__ = ["Config", "World"]
