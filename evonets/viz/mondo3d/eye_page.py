"""L'occhio a tutto schermo, mentre la prova gira: l'ultimo sguardo delle reti pronte.
Ogni cosa trovata da FastSAM ha il suo colore, lo stesso finché il tracker la segue,
con il riquadro, il numero e ciò che P ne sa. Gli oggetti veri che nessuna maschera
copre almeno a metà sono segnati in rosso: sono quelli che P non vede. Passando
sopra a una cosa si vede il ritaglio a scala fissa che legge DINOv2."""

import numpy as np
import pygame

from ...brains.ragionante.vision import KINDS
from ..render import BORDER, DIM, PLAYER, TEXT

MISSED = (255, 70, 60)
FOUND = (90, 220, 120)
SEEN = 0.5  # quanto una maschera deve coprire un oggetto vero perché sia visto


def tint(who: int) -> tuple:
    """Un colore fisso per ogni numero di oggetto: lo stesso finché il tracker lo segue."""
    rng = np.random.default_rng(who + 1)
    c = rng.uniform(60, 255, 3)
    return tuple(int(x) for x in c * 255 / c.max())


def covered(truth: list, masks: list) -> list:
    """Per ogni oggetto vero, quanto lo copre la maschera migliore (intersezione su unione)."""
    return [max(((m & t).sum() / (m | t).sum() for m in masks), default=0.0) for t, _ in truth]


class EyePage:
    def __init__(self, detector):
        self.detector = detector
        detector.truth = True  # da ora ogni sguardo registra anche gli oggetti veri
        self.cache = None  # (sguardo, immagine colorata)
        self.picked = None

    def _paint(self, look) -> np.ndarray:
        """L'immagine con ogni maschera colorata del suo colore."""
        out = look["rgb"].astype(np.float32)
        for mask, who in zip(look["masks"], look["ids"]):
            out[mask] = 0.55 * out[mask] + 0.45 * np.array(tint(who))
        return out.astype(np.uint8)

    def draw(self, s, app, run, b, i, mouse, paused) -> None:
        from .view import surface

        W, H = s.get_size()
        look = self.detector.looks[b][i]
        title = f"L'occhio dell'essere {i + 1}, recinto {b + 1}   t {run.t / 60:.1f} min" + ("   PAUSA" if paused else "")
        s.blit(app.font.render(title, True, PLAYER), (16, 10))
        if look is None:
            s.blit(app.font.render("ancora nessuno sguardo", True, DIM), (16, 50))
            return
        if self.cache is None or self.cache[0] is not look:
            self.cache = (look, surface(self._paint(look)))
        h, w = look["rgb"].shape[:2]
        side = min(W * 68 // 100, (H - 110) * w // h)
        pic = pygame.Rect(16, 44, side, side * h // w)
        s.blit(pygame.transform.smoothscale(self.cache[1], pic.size), pic)
        pygame.draw.rect(s, BORDER, pic, 1)
        scale = pic.w / w
        truth = look["truth"] or []
        cover = covered(truth, look["masks"])
        named = sum(k >= 0 for k in look["kind"])
        lost = sum(c < SEEN for c in cover)
        info = (f"sguardo di {run.t - look['t']:.1f} s fa · FastSAM: {len(look['masks'])} cose, {named} con un nome · "
                + (f"oggetti veri {len(truth)}, non visti {lost}" if look["truth"] is not None else "oggetti veri: dal prossimo sguardo"))
        s.blit(app.small.render(info, True, TEXT), (16, pic.bottom + 6))
        s.blit(app.small.render("colore = una cosa seguita dal tracker; rosso tratteggiato = oggetto vero non visto; "
                                "il nome serve solo al comandante finto, P non lo passa", True, DIM), (16, pic.bottom + 24))
        self.picked = None
        for k, (mask, who) in enumerate(zip(look["masks"], look["ids"])):
            rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
            rect = pygame.Rect(pic.x + cols[0] * scale, pic.y + rows[0] * scale,
                               max(2, (cols[-1] + 1 - cols[0]) * scale), max(2, (rows[-1] + 1 - rows[0]) * scale))
            if mouse and rect.collidepoint(mouse) and (self.picked is None or rect.w * rect.h < self.picked[1].w * self.picked[1].h):
                self.picked = (k, rect)
            pygame.draw.rect(s, tint(who), rect, 1)
            s.blit(app.small.render(f"#{who}", True, tint(who)), (rect.x + 2, max(pic.y, rect.y - 14)))
        for (mask, name), c in zip(truth, cover):
            if c >= SEEN:
                continue
            rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
            rect = pygame.Rect(pic.x + cols[0] * scale - 3, pic.y + rows[0] * scale - 3,
                               (cols[-1] + 1 - cols[0]) * scale + 6, (rows[-1] + 1 - rows[0]) * scale + 6)
            for x in range(rect.x, rect.right, 8):
                pygame.draw.line(s, MISSED, (x, rect.y), (min(x + 4, rect.right), rect.y), 2)
                pygame.draw.line(s, MISSED, (x, rect.bottom), (min(x + 4, rect.right), rect.bottom), 2)
            for y in range(rect.y, rect.bottom, 8):
                pygame.draw.line(s, MISSED, (rect.x, y), (rect.x, min(y + 4, rect.bottom)), 2)
                pygame.draw.line(s, MISSED, (rect.right, y), (rect.right, min(y + 4, rect.bottom)), 2)
            s.blit(app.small.render(f"non visto: {name}", True, MISSED), (rect.x, rect.bottom + 2))
        if self.picked:
            pygame.draw.rect(s, PLAYER, self.picked[1], 2)
        self._list(s, app, look, truth, cover, pic.right + 20, 44, W - pic.right - 36)

    def _list(self, s, app, look, truth, cover, x, y, wide) -> None:
        s.blit(app.font.render("Che cosa esce da P", True, PLAYER), (x, y))
        s.blit(app.small.render("#  nome (somiglianza)   distanza  larghezza", True, DIM), (x, y + 24))
        y += 42
        for k, who in enumerate(look["ids"]):
            if y > s.get_height() - 260:
                s.blit(app.small.render(f"... altre {len(look['ids']) - k}", True, DIM), (x, y))
                y += 16
                break
            kind = look["kind"][k]
            name = KINDS[kind] if kind >= 0 else "—"
            text = f"#{who:<3} {name} ({look['score'][k]:.2f})   {look['dist'][k]:.0f} cm   {look['size'][k]:.0f} cm"
            pygame.draw.rect(s, tint(who), (x, y + 3, 10, 10))
            colour = PLAYER if self.picked and self.picked[0] == k else TEXT if kind >= 0 else DIM
            s.blit(app.small.render(text, True, colour), (x + 16, y))
            y += 16
        if look["truth"] is not None:
            y += 8
            s.blit(app.font.render("Oggetti veri (dal simulatore)", True, PLAYER), (x, y))
            y += 22
            for (_, name), c in zip(truth, cover):
                s.blit(app.small.render(f"{name}: {'visto' if c >= SEEN else 'NON visto'} ({c:.2f})", True,
                                        FOUND if c >= SEEN else MISSED), (x, y))
                y += 16
        if self.picked:
            from ...brains.ragionante.ready import fixed
            from .view import surface

            k = self.picked[0]
            crop = fixed(look["rgb"], look["masks"][k], look["depth"])
            side = min(wide, 200)
            top = s.get_height() - side - 40
            s.blit(app.small.render(f"#{look['ids'][k]}: ciò che legge DINOv2 (40 cm veri)", True, TEXT), (x, top - 18))
            box = pygame.Rect(x, top, side, side)
            s.blit(pygame.transform.smoothscale(surface(crop), box.size), box)
            pygame.draw.rect(s, BORDER, box, 1)
