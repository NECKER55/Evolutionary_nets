"""Il mondo a eventi (docs/SPEC_CERVELLO_RAGIONANTE.md): niente fisica, ogni vita è
una lista di righe "oggetto, azione, esito", generata da una regola nascosta.

Ogni vita ha la sua regola, e la si deve scoprire vivendo. Per non imparare a
memoria le regole, alcune famiglie restano fuori dall'addestramento (`FAMILIES`):
è l'unico modo di sapere se ha imparato a imparare o solo una lista di regole.
"""

import jax
import jax.numpy as jnp
import numpy as np

EMBED = 7  # come `vision.EMBED`: lo stesso formato degli oggetti della percezione

ACTIONS = ("avvicino", "mordo", "ignoro")
OUTCOMES = ("niente", "piacere", "dolore")
NOTHING, PLEASURE, PAIN = 0, 1, 2
APPROACH, BITE, IGNORE = 0, 1, 2
LIFE = 60  # eventi in una vita
BITES = 0.6  # quota di morsi: gli altri eventi non danno esiti
SWAP = (10, 80)  # ogni quanti eventi i ruoli si scambiano: largo, o si impara quando succede
# Una regola è (famiglia, a, b), con a e b valori attesi delle due caratteristiche.
# Caratteristiche di un oggetto: 0 colore, 1 grandezza, 2 vicinanza.
FAMILIES = ("colore", "grandezza", "e", "o", "vicinanza", "soglia")
LEVELS = {"1": ("colore",), "2": ("colore", "grandezza", "e"), "largo": ("soglia",)}  # in addestramento
HELD_OUT = ("o", "vicinanza")  # mai viste in addestramento
COLOUR, SIZE, NEAR = 0, 1, 2
OBJECTS = np.array([[(k >> 2) & 1, (k >> 1) & 1, k & 1] for k in range(8)])  # i tipi di oggetto possibili


def threshold_rules() -> np.ndarray:
    """La famiglia larga "soglia": ogni regola del tipo "somma pesata delle caratteristiche
    sopra una soglia", tranne quelle tenute per l'esame. -> (M, 8) di bit, uno per tipo di oggetto.
    Non è una lista di regole da ricordare: sono 92, e vanno riconosciute dagli esiti."""
    found = set()
    for w in np.ndindex(7, 7, 7):
        score = OBJECTS @ (np.array(w) - 3)
        for edge in np.arange(-9.5, 10.0):
            found.add(tuple((score > edge).astype(int)))
    colour, size, near = OBJECTS.T
    banned = set()
    for a in (0, 1):
        banned.add(tuple((near == a).astype(int)))  # vicinanza
        for b in (0, 1):
            banned.add(tuple(((colour == a) | (size == b)).astype(int)))  # o
    banned |= {tuple(1 - np.array(x)) for x in banned}  # con gli scambi, anche i loro opposti
    keep = [x for x in found if 0 < sum(x) < 8 and x not in banned]
    return np.array(sorted(keep))


TABLES = jnp.asarray(threshold_rules())


def embed(bits, key) -> jnp.ndarray:
    """(N, 3) di bit -> (N, `EMBED`): lo stesso formato di `Percept.embedding`. Angolo a caso."""
    n = bits.shape[0]
    angle = jax.random.uniform(key, (n,), minval=-1.0, maxval=1.0)
    red, yellow = 1.0 - bits[:, COLOUR], bits[:, COLOUR]
    size = 0.48 + 0.32 * bits[:, SIZE]  # 9,6 e 16 cm, diviso 20: le pile piccole e grandi
    near = 0.2 + 0.6 * bits[:, NEAR]
    return jnp.stack([red, yellow, jnp.zeros(n), size, near, jnp.sin(angle), jnp.cos(angle)], -1)


def good(family, a, b, bits, rule=0) -> jnp.ndarray:
    """Quali oggetti `bits` (N, 3) sono buoni per la regola."""
    colour, size, near = bits[:, COLOUR] == a, bits[:, SIZE] == b, bits[:, NEAR] == a
    kind = 4 * bits[:, COLOUR] + 2 * bits[:, SIZE] + bits[:, NEAR]
    return jnp.stack([colour, bits[:, SIZE] == a, colour & size, colour | size, near, TABLES[rule, kind] == 1])[family]


def life(key, families: tuple) -> dict:
    """Una vita: oggetti, azioni, esiti. `families`: indici in `FAMILIES`."""
    k = jax.random.split(key, 8)
    family = jax.random.choice(k[0], jnp.array(families))
    a, b = jax.random.bernoulli(k[1], 0.5, (2,)).astype(int)
    bits = jax.random.bernoulli(k[2], 0.5, (LIFE, 3)).astype(int)
    gaps = jax.random.randint(k[3], (LIFE,), SWAP[0], SWAP[1] + 1)
    swaps = jnp.cumsum(gaps)
    flipped = jnp.searchsorted(swaps, jnp.arange(LIFE), side="right") % 2 == 1  # i ruoli sono scambiati
    rule = jax.random.randint(k[6], (), 0, TABLES.shape[0])
    is_good = good(family, a, b, bits, rule) ^ flipped
    act = jax.random.choice(k[4], 3, (LIFE,), p=jnp.array([(1 - BITES) / 2, BITES, (1 - BITES) / 2]))
    out = jnp.where(act == BITE, jnp.where(is_good, PLEASURE, PAIN), NOTHING)
    gain = jnp.where(out == PLEASURE, 0.25, jnp.where(out == PAIN, -0.15, 0.0))
    # L'energia prima dell'evento: quella dopo conterrebbe già l'esito da prevedere.
    energy = jnp.clip(0.5 + jnp.concatenate([jnp.zeros(1), jnp.cumsum(gain - 0.03)[:-1]]), 0.0, 1.0)
    return {"bits": bits, "obj": embed(bits, k[5]), "act": act, "out": out, "need": 1.0 - energy,
            "family": family, "a": a, "b": b, "rule": rule, "flipped": flipped}


def lives(key, n: int, names: tuple) -> dict:
    """`n` vite, con regole dalle famiglie `names`."""
    families = tuple(FAMILIES.index(f) for f in names)
    return jax.vmap(lambda k: life(k, families))(jax.random.split(key, n))


def table(one: dict) -> list:
    """Una vita come righe leggibili."""
    rows = []
    for t in range(int(one["out"].shape[0])):
        c, s, n = (int(x) for x in one["bits"][t])
        rows.append((t + 1, ("rossa", "gialla")[c], ("piccola", "grande")[s], ("lontana", "vicina")[n],
                     ACTIONS[int(one["act"][t])], OUTCOMES[int(one["out"][t])], bool(one["flipped"][t])))
    return rows


def rule_name(one: dict) -> str:
    f, a, b = FAMILIES[int(one["family"])], int(one["a"]), int(one["b"])
    if f == "soglia":
        kinds = [o for o, on in zip(OBJECTS, np.asarray(TABLES[int(one["rule"])])) if on]
        names = [f"{('rossa', 'gialla')[c]} {('piccola', 'grande')[s]} {('lontana', 'vicina')[n]}" for c, s, n in kinds]
        return "buone: " + ", ".join(names)
    colour, size, near = ("rosse", "gialle")[a], ("piccole", "grandi")[b], ("lontane", "vicine")[a]
    return {"colore": f"buone le {colour}", "grandezza": f"buone le {('piccole', 'grandi')[a]}",
            "e": f"buone solo le {colour} {size}", "o": f"buone le {colour} o le {size}",
            "vicinanza": f"buone le {near}"}[f]


def ideal(data: dict, names: tuple, hazard: float = 1.0 / 45.0) -> np.ndarray:
    """Il limite: un ragionatore perfetto che conosce tutte le regole possibili e tiene una
    probabilità per ognuna. -> (N, T): probabilità di piacere al morso, prima di vedere l'esito."""
    rules = set()
    bits8 = jnp.asarray(OBJECTS)
    for f in names:
        if f == "soglia":
            rules |= {tuple(int(v) for v in x) for x in np.asarray(TABLES)}
        else:
            rules |= {tuple(int(v) for v in good(FAMILIES.index(f), a, b, bits8)) for a in (0, 1) for b in (0, 1)}
    rules |= {tuple(1 - v for v in x) for x in rules}
    order = sorted(rules)
    each = np.array(order, dtype=bool)
    swap_of = np.array([order.index(tuple(1 - v for v in x)) for x in order])
    bits, seen = np.asarray(data["bits"]), np.asarray(data["out"])
    kind = 4 * bits[..., COLOUR] + 2 * bits[..., SIZE] + bits[..., NEAR]
    n, t = seen.shape
    belief = np.full((n, len(order)), 1.0 / len(order))
    out = np.zeros((n, t))
    for k in range(t):
        says = each[:, kind[:, k]].T  # (N, H): per ogni regola, l'oggetto è buono?
        out[:, k] = (belief * says).sum(-1)
        bitten = seen[:, k] != NOTHING
        likely = np.where(says == (seen[:, k] == PLEASURE)[:, None], 1.0, 1e-3)
        belief = np.where(bitten[:, None], belief * likely, belief)
        belief /= belief.sum(-1, keepdims=True)
        belief = (1.0 - hazard) * belief + hazard * belief[:, swap_of]
    return out


def remembered(data: dict) -> np.ndarray:
    """Il riferimento semplice: dice l'ultimo esito di un oggetto identico, 0,5 se mai visto."""
    bits, seen = np.asarray(data["bits"]), np.asarray(data["out"])
    n, t = seen.shape
    out = np.full((n, t), 0.5)
    for i in range(n):
        last = {}
        for k in range(t):
            key = tuple(bits[i, k])
            out[i, k] = last.get(key, 0.5)
            if seen[i, k] != NOTHING:
                last[key] = float(seen[i, k] == PLEASURE)
    return out


def bite_accuracy(p: np.ndarray, data: dict, start: int = 0) -> float:
    """Quanti morsi prevede giusti, da `start` in poi. Il caso fa 0,5."""
    seen = np.asarray(data["out"])[:, start:]
    p = np.asarray(p)[:, start:]
    bitten = seen != NOTHING
    return float(((p > 0.5) == (seen == PLEASURE))[bitten].mean())
