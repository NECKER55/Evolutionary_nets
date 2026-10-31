"""Gli agenti dell'arena delle pile: il cervello a sezioni (`sections.py`) e i due agenti
di confronto (`actor_critic.py`, `guided.py`), con ciò che hanno in comune."""

from ...arenas.pile.world import N_FEATURES, N_PROGRAMS
from .actor_critic import ActorCritic
from .attention import priority_map
from .genes import (
    APPROACH,
    BITE,
    DELTA_MAX,
    EXPLORE,
    FAST,
    FOVEA,
    GENE_NAMES,
    IDENTITY,
    INNATE,
    INNATE_CAP,
    LEAVE,
    MAP,
    MOTOR,
    N_A,
    N_BASE,
    N_HEADS,
    WEIGHT_MAX,
    Genes,
    Sizes,
    decode,
    default_genome,
    fovea_of,
    inherit,
    mutate,
    value_of,
)
from .guided import Guided
from .sections import Sections, c_code, teach_of, teach_override, teacher_mask
from .value import o_active, o_read, o_step, o_trace

__all__ = [
    "AGENTS", "APPROACH", "ActorCritic", "BITE", "DELTA_MAX", "EXPLORE", "FAST", "FOVEA", "GENE_NAMES", "Genes",
    "Guided", "IDENTITY", "INNATE", "INNATE_CAP", "LEAVE", "MAP", "MOTOR", "N_A", "N_BASE", "N_FEATURES", "N_HEADS",
    "N_PROGRAMS", "Sections", "Sizes", "WEIGHT_MAX", "brain_kwargs", "c_code", "decode", "default_genome",
    "fovea_of", "inherit", "mutate", "o_active", "o_read", "o_step", "o_trace", "priority_map", "teach_of",
    "teach_override", "teacher_mask", "value_of",
]

AGENTS = {"guidato": Guided, "attore-critico": ActorCritic, "sezioni": Sections}


def brain_kwargs(agent: str, cfg, lesions=()) -> dict:
    """Gli argomenti per costruire il cervello: quanti oggetti può guardare,
    le lesioni del cervello a sezioni, quante pile (per il guidato)."""
    kw = {"objects": cfg.batteries + cfg.beings}
    if agent == "sezioni":
        kw["sizes"] = Sizes(lesions=frozenset(lesions))
    if agent == "guidato":
        kw["batteries"] = cfg.batteries
    return kw
