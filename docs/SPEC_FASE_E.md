# Specifica: fase E — imparare durante la vita

Stato: **logica dei neuroni implementata** (2026-10-14). L'ambiente che rende
utile imparare e il compito non sono ancora decisi: per scelta dell'utente si
preparano dopo.

## Riferimenti biologici

- **Regola a tre fattori.** Una sinapsi cambia se i due neuroni sono attivi
  insieme (Hebb, 1949) *e* se arriva un segnale modulatore (dopamina,
  octopamina) che dice "è andata bene" o "male". Così imparano la mosca della
  frutta (corpi a fungo, dopamina), l'ape (il neurone VUMmx1, octopamina) e i
  mammiferi.
- **Plasticità evoluta.** Niv, Joel, Meilijson e Ruppin (2002): api virtuali
  in cui l'apprendimento per rinforzo nasce per evoluzione. Soltoggio et al.
  (2008): reti con neuroni modulatori e regola evolvibile, la stessa forma
  usata qui.
- **Tracce.** Nelle reti a impulsi il premio arriva dopo l'azione: una traccia
  dell'attività recente permette di rinforzare ciò che l'ha causata
  (Izhikevich, 2007).
- **Trasmissione di volume.** La dopamina non viaggia su un filo: si diffonde
  nel tessuto e agisce su ciò che sta vicino.
- **Barriera di Weismann.** Ciò che si impara non si eredita; si eredita la
  capacità di imparare. Nel tempo, ciò che si impara sempre può finire nei geni
  (effetto Baldwin).

## Il modello

**Geni** (in coda ai geni propri di ogni neurone interno o motore; riguardano
le sinapsi che il neurone *riceve*):

| Gene | Significato |
|---|---|
| η | velocità di apprendimento: η = η0 · e^gene (η0 = 0,001, `plasticity_initial_rate`) |
| A, B, C, D | la regola: Δw ∝ A·pre·post + B·pre + C·post + D |
| mod | se > 0, un neurone interno è **modulatore** |

**Modulatori.** Non trasmettono: i loro collegamenti in uscita non esistono e
non costano cablaggio. La loro attività rilascia M, che arriva a ogni neurone
calando con la distanza nel cervello:

```
M(t) = Σ_modulatori  attività_m · segno_m · exp(−d(m, t) / 0,2)
```

(`modulation_range`). Nelle reti continue l'attività ha già un segno; in quelle
a impulsi l'attività è la traccia degli impulsi e il segno viene dal tipo del
modulatore: eccitatorio = M positivo ("bene"), inibitorio = M negativo
("male").

**Regola.** A ogni passo, per ogni collegamento effettivo verso il neurone t:

```
Δw = η_t · M_t · (A_t · pre · post + B_t · pre + C_t · post + D_t)
```

- reti continue: pre = attività della sorgente al passo prima, post = attività
  del neurone dopo il passo;
- reti a impulsi: pre e post sono le tracce recenti degli impulsi (e i livelli
  dei motori).

**Limiti fisici.**

- forza massima di una sinapsi ±4 (`synapse_max`): le sinapsi vere saturano;
- nelle reti a impulsi un collegamento da un neurone eccitatorio resta
  eccitatorio, e viceversa (legge di Dale), anche imparando.

**Vita.** A ogni nascita e a ogni arena i pesi ripartono da quelli del genoma.
Il genoma non viene mai modificato da ciò che si impara.

**Chi impara.** Solo chi ha almeno un modulatore acceso: senza M non cambia
nulla (niente Hebb puro).

**Partenza.** Plasticità debole e diversa tra individui (gene di η ~ N(0; 0,5),
A–D ~ N(0; 0,5)); circa il 10% dei neuroni nasce con il gene modulatore acceso
(`modulatory_initial_share`). Tutti questi geni mutano e si incrociano con il
loro neurone, come gli altri geni propri.

**Interruttore.** `plasticity = False` spegne l'apprendimento (per confronti).

## Cosa non c'è ancora

- **L'ambiente che rende utile imparare** (proposte: fiori di due colori con
  valore sorteggiato a ogni arena; inversione a metà vita) e il **compito**
  collegato (bersaglio con il cibo per vivere; nutrirsi; sopravvivere). Sono
  cambi d'ambiente: li decide l'utente.
- **Il segnale di "bene/male" dal mondo** c'è già in potenza: un recettore
  interno sensibile all'energia con adattamento sottrattivo dà un picco quando
  l'energia sale (cibo), uno chimico sente la tossina. Ma oggi l'energia non
  sale mai: senza cibo, la ricompensa non esiste.
- **Costo della plasticità e dell'attività**: imparare e scaricare impulsi non
  costano energia.
- **Duplicazione**: alla nascita resta neutra; durante la vita no, perché le
  due copie imparano ciascuna per conto suo.

## Formato dei salvataggi

Formato 5. I salvataggi precedenti si aprono senza plasticità (η quasi nullo)
e senza modulatori; il registro lo annota.
