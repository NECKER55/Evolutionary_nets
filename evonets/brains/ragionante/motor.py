"""La sezione 7, il motore: traduce un'intenzione (velocità, rotazione, bocca)
in uscite per un corpo che non conosce. Sopra, nessuno sa com'è fatto il corpo.

Tre neuroni d'ingresso, `HIDDEN` nascosti e K d'uscita. Si tara da solo col
"motor babbling": dà comandi, sente che cosa ha fatto il corpo, e impara il
contrario col gradiente: ingresso = il movimento sentito, bersaglio = il comando
dato. All'inizio i comandi sono a caso; poi prova ad andare in direzioni a caso,
perché col solo caso certi corpi non toccano mai i loro estremi, e a stare fermo.

Si tara alla nascita, quando il corpo non obbedisce più (`lost`: confronta ciò
che ha chiesto con ciò che ha sentito) e quando lo chiede qualcuno (`tune`). Il
resto del cervello non ne sa niente.
"""

import numpy as np

INTENTS = ("velocità", "rotazione", "bocca")
HIDDEN = 8  # neuroni nascosti
BIRTH = 60.0  # s di comandi a caso alla nascita
RETUNE = 30.0  # s quando il corpo non obbedisce più
HOLD = 12  # passi per cui si tiene lo stesso comando
SETTLE = 5  # passi di questi in cui il corpo sta ancora cambiando velocità: non insegnano
GOALS = 0.5  # da quale quota della taratura si provano direzioni invece di comandi a caso
NOISE = 0.2  # il rumore su quei comandi
REST = 0.4  # quanti di quei comandi chiedono di stare fermi: per mordere serve fermarsi davvero
MEMORY = 1024  # esempi ricordati per essere, della taratura di adesso
BATCH = 32  # esempi per ogni passo di gradiente
RATE = 0.01  # velocità di apprendimento (Adam)
PATIENCE = 5.0  # s su cui si giudica se il corpo obbedisce
# Sotto queste quote di ciò che si chiede (velocità, rotazione) il corpo non obbedisce più.
# Misurate: la velocità sana non scende sotto 0,59. La rotazione sana arriva fino a 0,
# perché il comandante corregge di continuo: conta solo se gira al contrario.
OBEY = np.array((0.3, -0.1))
ASKED = 0.02  # sotto questa media di intenzione al quadrato non si giudica: si chiede troppo poco


class Motor:
    def __init__(self, shape, outputs: int, dt: float, seed: int = 0):
        self.shape, self.dt = shape, dt
        self.rng = np.random.default_rng([seed, 7])
        self.tunings = np.zeros(shape, dtype=int)  # quante tarature, per essere
        self.resize(outputs)

    def resize(self, outputs: int) -> None:
        """Un corpo con un altro numero di uscite: la rete ricomincia da zero."""
        s, n = self.shape, len(INTENTS)
        self.K = outputs
        self.params = [self.rng.normal(0.0, 0.5, s + (HIDDEN, n)), np.zeros(s + (HIDDEN,)),
                       self.rng.normal(0.0, 0.1, s + (outputs, HIDDEN)), np.zeros(s + (outputs,))]
        self.slow = [np.zeros_like(p) for p in self.params]  # le medie di Adam
        self.fast = [np.zeros_like(p) for p in self.params]
        self.steps = np.zeros(s)
        self.seen_x = np.zeros(s + (MEMORY, n))
        self.seen_u = np.zeros(s + (MEMORY, outputs))
        self.count = np.zeros(s, dtype=int)  # esempi visti in questa taratura
        self.u = np.zeros(s + (outputs,))
        self.intent = np.zeros(s + (n,))
        self.felt = np.zeros(s + (n,))
        self.hidden = np.zeros(s + (HIDDEN,))
        self.left = np.zeros(s)  # s di taratura ancora da fare
        self.span = np.ones(s)  # quanto dura la taratura di adesso
        self.held = np.zeros(s, dtype=int)  # da quanti passi si tiene il comando
        self.error = np.zeros(s)  # quanto sbaglia la rete a spiegare il comando dato (media lenta)
        self.asked = np.zeros(s + (2,))  # medie lente: intenzione al quadrato, per velocità e rotazione
        self.got = np.zeros(s + (2,))  # e intenzione per movimento sentito
        self.watched = np.zeros(s)  # s di queste medie: prima di `PATIENCE` non si giudica
        self.tune(BIRTH)

    def tune(self, seconds: float, mask=None) -> None:
        """Comincia a tararsi, quando si vuole. `mask` (B, n): chi; senza, tutti.
        Gli esempi del corpo di prima non valgono più, la rete resta."""
        mask = np.ones(self.shape, dtype=bool) if mask is None else mask
        self.left[mask] = seconds
        self.span[mask] = seconds
        self.held[mask] = HOLD
        self.count[mask] = 0
        self.asked[mask] = self.got[mask] = 0.0
        self.watched[mask] = 0.0
        self.tunings[mask] += 1

    @property
    def tuning(self) -> np.ndarray:
        return self.left > 0.0

    def _hidden(self, x) -> np.ndarray:
        """`x` (..., N, 3) -> attività dei nascosti (..., N, HIDDEN)."""
        return np.tanh(x @ np.swapaxes(self.params[0], -1, -2) + self.params[1][..., None, :])

    def forward(self, x) -> np.ndarray:
        """`x` (..., N, 3) -> uscite (..., N, K), senza limiti."""
        return self._hidden(x) @ np.swapaxes(self.params[2], -1, -2) + self.params[3][..., None, :]

    def command(self, x) -> np.ndarray:
        """Come `forward`, ma senza intenzione nessun comando: anche un'intenzione
        minima deve contare, per fare gli ultimi centimetri fino a una pila."""
        return self.forward(x) - self.forward(np.zeros((1, len(INTENTS))))

    @property
    def response(self) -> np.ndarray:
        """(..., K, 3): di quanto cambia ogni uscita chiedendo 1 di ogni intenzione."""
        return np.swapaxes(self.command(np.eye(len(INTENTS))), -1, -2)

    @property
    def lost(self) -> np.ndarray:
        """Chi chiede un movimento e non lo sente: il corpo non obbedisce più."""
        return (self.watched >= PATIENCE) & ((self.asked > ASKED) & (self.got < OBEY * self.asked)).any(-1)

    def _train(self, on) -> None:
        """Un passo di gradiente sugli esempi ricordati, per chi è `on`."""
        pick = self.rng.integers(0, np.minimum(self.count, MEMORY)[..., None].clip(1), self.shape + (BATCH,))
        x = np.take_along_axis(self.seen_x, pick[..., None], -2)
        h = self._hidden(x)
        miss = h @ np.swapaxes(self.params[2], -1, -2) + self.params[3][..., None, :] \
            - np.take_along_axis(self.seen_u, pick[..., None], -2)
        back = (miss @ self.params[2]) * (1.0 - h * h)
        grads = [np.swapaxes(back, -1, -2) @ x / BATCH, back.mean(-2),
                 np.swapaxes(miss, -1, -2) @ h / BATCH, miss.mean(-2)]
        self.steps = self.steps + on
        for p, slow, fast, g in zip(self.params, self.slow, self.fast, grads):
            at = on.reshape(on.shape + (1,) * (g.ndim - on.ndim))
            slow += at * 0.1 * (g - slow)
            fast += at * 0.001 * (g * g - fast)
            t = self.steps.reshape(at.shape)
            p -= at * RATE * (slow / (1.0 - 0.9 ** t.clip(1))) \
                / (np.sqrt(fast / (1.0 - 0.999 ** t.clip(1))) + 1e-8)

    def _learn(self, felt, touched) -> None:
        """Il ricordo: in un urto il corpo non dice la verità sui comandi, e non si impara."""
        on = self.tuning & (self.held >= SETTLE) & ~touched
        if not on.any():
            return
        miss = self.u - self.forward(felt[..., None, :])[..., 0, :]
        self.error = np.where(on, 0.98 * self.error + 0.02 * np.abs(miss).mean(-1), self.error)
        at = np.nonzero(on)
        slot = self.count[at] % MEMORY
        self.seen_x[(*at, slot)] = felt[at]
        self.seen_u[(*at, slot)] = self.u[at]
        self.count[at] += 1
        self._train(on)

    def _command(self, new) -> np.ndarray:
        """Il prossimo comando di taratura: a caso, poi la rete che prova una direzione a caso."""
        guess = self.rng.uniform(-2.0, 2.0, self.shape + (self.K,)).clip(-1.0, 1.0)  # metà agli estremi
        aim = self.rng.uniform(-1.0, 1.0, self.shape + (len(INTENTS),))
        aim[..., 2] = self.rng.integers(0, 2, self.shape)
        aim[..., :2] *= self.rng.random(self.shape + (1,)) > REST
        tried = self.command(aim[..., None, :])[..., 0, :] + NOISE * self.rng.normal(size=guess.shape)
        goals = self.left < (1.0 - GOALS) * self.span
        return np.where(new[..., None], np.where(goals[..., None], tried.clip(-1.0, 1.0), guess), self.u)

    def _judge(self, felt, touched) -> None:
        """Confronta la velocità e la rotazione chieste col passo prima con quelle sentite."""
        on = (~self.tuning & ~touched)[..., None]
        fade = np.where(on, self.dt / PATIENCE, 0.0)
        self.watched = self.watched + on[..., 0] * self.dt
        wanted = self.intent[..., :2]
        self.asked = self.asked + fade * (wanted * wanted - self.asked)
        self.got = self.got + fade * (wanted * felt[..., :2] - self.got)

    def step(self, intent, felt, touched) -> np.ndarray:
        """`intent` (B, n, 3); `felt` (B, n, 3): che cosa ha fatto il corpo col
        comando di prima; `touched` (B, n): il corpo non è libero di muoversi (urta
        qualcosa, sta mangiando). -> uscite (B, n, K)."""
        self._learn(felt, touched)
        self._judge(felt, touched)
        self.intent, self.felt = intent, felt
        self.left = np.maximum(self.left - self.dt, 0.0)
        new = self.tuning & (self.held >= HOLD)
        if new.any():
            self.u = self._command(new)
        self.held = np.where(new, 0, self.held + 1)
        self.hidden = self._hidden(intent[..., None, :])[..., 0, :]
        ruled = self.command(intent[..., None, :])[..., 0, :].clip(-1.0, 1.0)
        self.u = np.where(self.tuning[..., None], self.u, ruled)
        return self.u
