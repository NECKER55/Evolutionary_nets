"""Le mappe del mondo 3D: stanze quadrate unite da porte, generate da un seme.

Ogni stanza ha i muri di un colore, ed è il colore a far riconoscere dove si è; due
stanze possono averlo uguale, come nelle case vere. Le stanze formano un albero più
qualche porta in più (`loops`): così esiste più di una strada. Questa è la verità
del mondo: un cervello non la riceve, la può solo ricostruire girando.
"""

from dataclasses import dataclass

import numpy as np

WALL = 5.0  # spessore del muro di una stanza: tra due stanze sono due, uno per colore
PASS = 40.0  # quanto prima e dopo una porta si passa, per attraversarla dritti


@dataclass(frozen=True)
class Plan:
    cells: int  # stanze per lato
    cell: float  # lato di una stanza
    door: float = 90.0  # larghezza di una porta
    loops: int = 0  # porte in più rispetto a quelle che bastano


PLANS = {"aperta": None, "stanze": Plan(3, 400.0, loops=2), "labirinto": Plan(5, 260.0, door=80.0)}


class Layout:
    """Una mappa concreta. Una stanza è un numero, riga · `cells` + colonna."""

    def __init__(self, plan: Plan | None, seed: int, colours: int):
        self.cells = plan.cells if plan else 1
        self.cell = plan.cell if plan else 0.0
        self.doors = {}  # (a, b) -> (centro della porta, direzione da a a b)
        rects, tints = [], []
        if plan:
            rng = np.random.default_rng([seed, 17])
            self.tint = rng.integers(0, colours, self.cells ** 2)
            for a, b in self._links(plan.loops, rng):
                self._door(a, b, plan, rng)
            for room in range(self.cells ** 2):
                for r in self._sides(room, plan):
                    rects.append(r)
                    tints.append(self.tint[room])
        self.rects = np.array(rects).reshape(-1, 4)  # (M, 4): x0, y0, x1, y1
        self.tints = np.array(tints, dtype=int)
        self._next = self._routes()

    def _near(self, room: int) -> list:
        """Le stanze confinanti."""
        y, x, n = room // self.cells, room % self.cells, self.cells
        return [room + d for d, ok in ((1, x < n - 1), (-1, x > 0), (n, y < n - 1), (-n, y > 0)) if ok]

    def _links(self, loops: int, rng) -> list:
        """Quali stanze si toccano con una porta: un albero a caso, più `loops` porte."""
        seen, stack, links = {0}, [0], []
        while stack:
            free = [r for r in self._near(stack[-1]) if r not in seen]
            if not free:
                stack.pop()
                continue
            r = int(rng.choice(free))
            links.append((stack[-1], r))
            seen.add(r)
            stack.append(r)
        rest = [(a, b) for a in range(self.cells ** 2) for b in self._near(a) if a < b
                and (a, b) not in links and (b, a) not in links]
        rng.shuffle(rest)
        return links + [tuple(int(v) for v in x) for x in rest[:loops]]

    def _door(self, a: int, b: int, plan: Plan, rng) -> None:
        y, x = a // self.cells, a % self.cells
        along = rng.uniform(plan.door, plan.cell - plan.door)
        if abs(a - b) == 1:
            side = 1 if b > a else 0
            centre, way = np.array(((x + side) * plan.cell, y * plan.cell + along)), np.array((np.sign(b - a), 0.0))
        else:
            side = 1 if b > a else 0
            centre, way = np.array((x * plan.cell + along, (y + side) * plan.cell)), np.array((0.0, np.sign(b - a)))
        self.doors[a, b], self.doors[b, a] = (centre, way), (centre, -way)

    def _sides(self, room: int, plan: Plan) -> list:
        """I muri di una stanza, dentro di lei, con i buchi delle porte. Quelli esterni li ha già il recinto."""
        y, x, c, n = room // self.cells, room % self.cells, plan.cell, self.cells
        x0, y0, x1, y1 = x * c, y * c, (x + 1) * c, (y + 1) * c
        out = []
        for other, across, at, inner in ((room - n, False, y0, y0 + WALL), (room + n, False, y1 - WALL, y1),
                                         (room - 1, True, x0, x0 + WALL), (room + 1, True, x1 - WALL, x1)):
            if other not in self._near(room):
                continue
            lo, hi = (y0, y1) if across else (x0, x1)
            door = self.doors.get((room, other))
            mid = None if door is None else door[0][1 if across else 0]
            spans = [(lo, hi)] if mid is None else [(lo, mid - plan.door / 2.0), (mid + plan.door / 2.0, hi)]
            for a, b in spans:
                out.append((at, a, inner, b) if across else (a, at, b, inner))
        return out

    def _routes(self) -> np.ndarray:
        """(R, R): da una stanza, in quale entrare per andare verso un'altra."""
        rooms = self.cells ** 2
        hop = np.full((rooms, rooms), -1)
        for goal in range(rooms):
            hop[goal, goal], todo = goal, [goal]
            for room in todo:
                for other in self._near(room):
                    if (room, other) in self.doors and hop[other, goal] < 0:
                        hop[other, goal] = room
                        todo.append(other)
        return hop

    def room_of(self, pos) -> np.ndarray:
        if not self.cell:
            return np.zeros(np.shape(pos)[:-1], dtype=int)
        at = np.clip((np.asarray(pos) / self.cell).astype(int), 0, self.cells - 1)
        return at[..., 1] * self.cells + at[..., 0]

    def way(self, pos, goal) -> tuple:
        """Dove andare adesso per arrivare a `goal`. -> (punto, è la meta?): le stanze sono
        convesse, quindi basta la prossima porta. Prima ci si mette davanti, poi la si passa."""
        here, there = int(self.room_of(pos)), int(self.room_of(goal))
        if here == there:
            return goal, True
        centre, way = self.doors[here, int(self._next[here, there])]
        off = pos - centre
        lined = abs(off @ np.array((-way[1], way[0]))) < 20.0
        return centre + (PASS if lined else -PASS) * way, False

    def near_wall(self, pos, reach: float) -> np.ndarray:
        """(M, 2): verso i punti dei muri più vicini di `reach`, da `pos`."""
        nearest = np.clip(pos, self.rects[:, :2], self.rects[:, 2:]) - pos
        return nearest[np.hypot(*nearest.T) < reach]

    def hides(self, a, b) -> np.ndarray:
        """Quali segmenti da `a` (..., 2) a `b` (..., 2) tagliano un muro."""
        if not len(self.rects):
            return np.zeros(np.broadcast_shapes(np.shape(a), np.shape(b))[:-1], dtype=bool)
        a, d = np.asarray(a)[..., None, :], (np.asarray(b) - a)[..., None, :]
        d = np.where(np.abs(d) < 1e-9, 1e-9, d)
        lo, hi = (self.rects[:, :2] - a) / d, (self.rects[:, 2:] - a) / d
        enter = np.minimum(lo, hi).max(-1).clip(0.0)
        leave = np.maximum(lo, hi).min(-1).clip(max=1.0)
        return (enter < leave).any(-1)
