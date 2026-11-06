"""Controlli minimi per i moduli dell'interfaccia: scelta tra opzioni, numero,
testo libero e pulsante. Si usano con mouse e tastiera."""

import pygame

from .render import BG, BORDER, DIM, IDLE, PLAYER, TEXT

ROW_H = 34


def wrap(font, text: str, width: int) -> list:
    """Il testo a righe larghe al massimo `width` pixel."""
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and font.size(trial)[0] > width:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + [line] if line else lines


class Field:
    """Una riga del modulo: etichetta a sinistra, valore in un riquadro."""

    def __init__(self, label: str, help: str = ""):
        self.label, self.help = label, help
        self.enabled = True
        self.rect = pygame.Rect(0, 0, 0, 0)  # il riquadro del valore, fissato da `draw`

    def text(self) -> str:
        raise NotImplementedError

    def detail(self) -> str:
        """Che cosa vuol dire il valore di adesso, se serve dirlo."""
        return ""

    def explain(self) -> str:
        """Che cosa significa la riga, col valore di adesso."""
        return f"{self.detail()} {self.help}".strip()

    def change(self, step: int) -> None:
        """Freccia sinistra (-1) o destra (+1)."""

    def type(self, ev) -> None:
        """Un tasto premuto mentre la riga è selezionata."""

    def click(self, pos) -> None:
        """Clic sul riquadro: sulla metà sinistra indietro, sulla destra avanti."""
        self.change(-1 if pos[0] < self.rect.centerx else 1)

    def draw(self, surface, font, x: int, y: int, label_w: int, box_w: int, focused: bool) -> None:
        color = TEXT if self.enabled else DIM
        surface.blit(font.render(self.label, True, color), (x, y + 8))
        self.rect = pygame.Rect(x + label_w, y, box_w, ROW_H - 6)
        pygame.draw.rect(surface, BG if self.enabled else (20, 22, 28), self.rect)
        pygame.draw.rect(surface, PLAYER if focused and self.enabled else BORDER, self.rect, 1)
        text = self.text()
        while len(text) > 4 and font.size(text)[0] > box_w - 20:  # troppo lungo: se ne mostra la fine
            text = "..." + text[4:]
        surface.blit(font.render(text, True, color), (self.rect.x + 10, y + 6))


class Choice(Field):
    """Una tra più opzioni: `options` = [(valore, testo), ...]."""

    def __init__(self, label, options, index: int = 0, help: str = "", on_change=None, details=None):
        super().__init__(label, help)
        self.options, self.index, self.on_change = list(options), index, on_change
        self.details = details or {}  # valore -> che cosa vuol dire sceglierlo

    @property
    def value(self):
        return self.options[self.index][0]

    def select(self, value) -> None:
        self.index = [v for v, _ in self.options].index(value)

    def text(self) -> str:
        return f"< {self.options[self.index][1]} >"

    def detail(self) -> str:
        return self.details.get(self.value, "")

    def change(self, step: int) -> None:
        if self.enabled:
            self.index = (self.index + step) % len(self.options)
            if self.on_change:
                self.on_change(self.value)


class Number(Field):
    """Intero non negativo: frecce per passi di `step`, oppure lo si scrive."""

    def __init__(self, label, value: int, step: int = 1, low: int = 0, help: str = ""):
        super().__init__(label, help)
        self.value, self.step, self.low = value, step, low

    def text(self) -> str:
        return f"< {self.value} >"

    def change(self, step: int) -> None:
        if self.enabled:
            self.value = max(self.low, self.value + step * self.step)

    def type(self, ev) -> None:
        if not self.enabled:
            return
        if ev.key == pygame.K_BACKSPACE:
            self.value //= 10
        elif ev.unicode.isdigit():
            self.value = min(self.value * 10 + int(ev.unicode), 10**7)

    def fixed(self) -> int:
        """Il valore, riportato al minimo se scritto più basso."""
        self.value = max(self.low, self.value)
        return self.value


class Text(Field):
    """Testo libero (per esempio un percorso)."""

    def __init__(self, label, value: str = "", help: str = ""):
        super().__init__(label, help)
        self.value = value
        self.edited = False  # True dopo che l'utente l'ha toccato

    def text(self) -> str:
        return self.value + ("_" if pygame.time.get_ticks() // 500 % 2 else " ")

    def click(self, pos) -> None:
        pass

    def type(self, ev) -> None:
        if ev.key == pygame.K_BACKSPACE:
            self.value = self.value[:-1]
            self.edited = True
        elif ev.unicode and ev.unicode.isprintable():
            self.value += ev.unicode
            self.edited = True


class Picker(Field):
    """Una scelta fatta in un elenco a parte: la riga mostra la scelta attuale,
    e Invio, le frecce o un clic aprono l'elenco (`on_open`)."""

    def __init__(self, label, on_open, help: str = ""):
        super().__init__(label, help)
        self.on_open = on_open
        self.value = None
        self.shown = ""  # come si mostra la scelta attuale

    def text(self) -> str:
        return f"{self.shown}  [scegli...]"

    def change(self, step: int) -> None:
        if self.enabled:
            self.on_open()

    def click(self, pos) -> None:
        self.change(1)


class Button:
    def __init__(self, label: str, action, primary: bool = False):
        self.label, self.action, self.primary = label, action, primary
        self.enabled = True
        self.rect = pygame.Rect(0, 0, 0, 0)

    def draw(self, surface, font, x: int, y: int, w: int = 0) -> int:
        """Disegna e restituisce la larghezza occupata."""
        img = font.render(self.label, True, TEXT if self.enabled else DIM)
        self.rect = pygame.Rect(x, y, w or img.get_width() + 32, ROW_H)
        hover = self.enabled and self.rect.collidepoint(pygame.mouse.get_pos())
        fill = (70, 58, 24) if self.primary and self.enabled else IDLE if hover else (32, 35, 46)
        pygame.draw.rect(surface, fill, self.rect, border_radius=4)
        pygame.draw.rect(surface, PLAYER if self.primary and self.enabled else BORDER, self.rect, 1, border_radius=4)
        surface.blit(img, img.get_rect(center=self.rect.center))
        return self.rect.w

    def click(self, pos) -> bool:
        if self.enabled and self.rect.collidepoint(pos):
            self.action()
            return True
        return False
