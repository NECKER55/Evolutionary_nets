"""Motore di simulazione compilato con JAX (GPU se c'è). Vedi engine.py."""

import os
from pathlib import Path

_ready = None


def available() -> bool:
    """JAX è installato e utilizzabile? La prima volta attiva anche la cache
    delle compilazioni su disco: ogni forma di rete si compila una volta sola,
    anche tra un avvio e l'altro."""
    global _ready
    if _ready is None:
        try:
            import jax

            cache = Path(os.environ.get("EVONETS_JAX_CACHE", Path.home() / ".cache" / "evonets-jax"))
            cache.mkdir(parents=True, exist_ok=True)
            jax.config.update("jax_compilation_cache_dir", str(cache))
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
            _ready = True
        except Exception:
            _ready = False
    return _ready
