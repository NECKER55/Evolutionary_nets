# Traguardi: arena delle pile e cervello a sezioni

Serve a tenere traccia dei passi principali e a capire se il sistema migliora
davvero, o se migliorando una cosa ne peggiora un'altra. Ogni versione è
misurata con **lo stesso banco di prova** (`evonets/evo/pile_bench.py`): le
versioni salvate in git si rimisurano da un `git worktree` (istruzioni nella
docstring del modulo).

## Il banco di prova

Cervello a sezioni, fase A (nessuna evoluzione: si misura l'apprendimento
dentro la vita), 200 recinti da 6 esseri, 15 minuti di mondo, pile buone per
colore, semi 0 e 1 (la tabella riporta la media: tra i due semi le misure
differiscono di meno di 1 punto). Le misure:

- **buone**: quota di pile buone tra quelle prese (la scelta);
- **pile/min** e **buone/min**: quante pile prende un essere al minuto;
- **energia/min**: energia netta dalle pile, buone × 25 − cattive × 8 (quanto
  la scelta *e* la quantità nutrono davvero);
- **morti/min**: morti di fame per essere al minuto (la sopravvivenza);
- **inversione**: quota di buone nelle finestre −30..0, 0..15, 15..30, 30..60,
  60..90 s intorno a ogni inversione (prima: quanto si è imparato; dopo:
  quanto in fretta ci si riadatta).

Nessuna misura da sola dice se si va meglio: una versione può scegliere
meglio e mangiare meno, o il contrario.

## Le versioni

| versione | buone | pile/min | buone/min | energia/min | morti/min | inversione |
|---|---|---|---|---|---|---|
| 17b0806 prima versione delle sezioni | 53,1% | 2,79 | 1,48 | 26,6 | **0,102** | 0,61 · 0,35 · 0,37 · 0,41 · 0,44 |
| 27dca3b attenzione, tracce strette, soglia sulla sorpresa; C legge l'oggetto, D ricorda | 54,0% | 2,63 | 1,42 | 25,8 | 0,180 | 0,61 · 0,38 · 0,41 · 0,43 · 0,47 |
| e498d67 corpo nuovo: sensi elementari, morso deciso da E, F come circuito di neuroni | 57,2% | 2,76 | 1,58 | 30,1 | 0,198 | 0,66 · 0,38 · 0,41 · 0,44 · 0,48 |
| a3b4edd arbitraggio con U, genoma con pesi e connessioni, salvataggi | 56,0% | 3,45 | 1,94 | 36,2 | 0,192 | 0,63 · 0,39 · 0,42 · 0,45 · 0,50 |
| e09d770 costanti come geni, critico legato al corpo, attenzione a mappa di priorità, maestro, O | 57,9% | 3,14 | 1,82 | 34,8 | 0,178 | 0,63 · 0,43 · 0,45 · 0,50 · 0,54 |
| O con dolore pesato (×2) e volatilità | 59,8% | 3,11 | 1,86 | 36,5 | 0,183 | 0,64 · 0,45 · 0,49 · 0,53 · **0,58** |
| **attuale**: + C come cellule di Kenyon, scala della sorpresa adattata, inerzia legata agli oggetti | 58,9% | **3,54** | **2,09** | **40,5** | 0,145 | 0,64 · 0,38 · 0,43 · 0,46 · 0,50 |
| variante: senza la scala adattata (C nuovo e inerzia sì) | **66,0%** | 2,46 | 1,62 | 33,9 | 0,167 | **0,77** · 0,29 · 0,37 · 0,44 · 0,52 |
| variante: solo l'inerzia legata agli oggetti | 62,2% | 2,71 | 1,68 | 33,9 | 0,193 | 0,70 · 0,39 · 0,47 · 0,52 · 0,58 |

Riferimenti, sulla versione attuale del mondo:

| | buone | pile/min | buone/min | energia/min | morti/min | inversione |
|---|---|---|---|---|---|---|
| guidato a mano (sa qual è la buona) | 94,7% | 2,35 | 2,23 | 54,7 | 0,000 | 0,94 · 0,98 · 0,94 · 0,95 · 0,93 |
| attore-critico classico (senza sezioni) | 50,4% | 3,54 | 1,78 | 30,6 | 0,102 | 0,51 · 0,49 · 0,50 · 0,50 · 0,50 |

Attenzione: tra una versione e l'altra è cambiato anche il mondo (per
esempio il morso deciso da E, che costa energia, e le pile grandi o piccole).
Le differenze tra versioni sono quindi di tutto il sistema, non solo del
cervello.

## Che cosa ha portato ogni passo

- **Prima versione (17b0806)**: le sezioni A, B, C, D, E, L, U, V; il
  riflesso dei muri corretto. Sceglie poco meglio del caso (53%), ma muore di
  fame poco.
- **Attenzione e C sull'oggetto (27dca3b)**: un oggetto alla volta, C legge
  che cosa si guarda. La scelta cambia poco; **le morti di fame quasi
  raddoppiano** (0,10 → 0,18) e non torneranno più al livello di prima.
- **Corpo nuovo (e498d67)**: il morso diventa una decisione (e costa); la
  scelta sale al 57%, le morti restano alte.
- **Arbitraggio e genoma (a3b4edd)**: molte più pile al minuto (3,45),
  scelta un po' più bassa.
- **Critico, mappa di priorità, maestro, O (e09d770)**: il riadattamento
  dopo l'inversione migliora (0,50 → 0,54 a 60-90 s).
- **O con dolore e volatilità**: la scelta sale al 59,8%, il riadattamento è
  il migliore (0,58).
- **C come cellule di Kenyon, scala adattata, inerzia sugli oggetti
  (attuale)**: più cibo e più energia di sempre, morti di fame al minimo
  dalla seconda versione; ma il riadattamento dopo l'inversione torna
  indietro (0,58 → 0,50).

## Migliorando una cosa se ne peggiora un'altra?

Sì, in tre punti:

1. **Scelta contro quantità.** La variante senza la scala adattata sceglie
   meglio di tutte (66%, 77% prima di un'inversione) ma mangia poco (2,46
   pile al minuto): è più selettiva e più lenta. Con la scala adattata si
   sceglie meno bene (59%) ma si mangia di più, e alla fine si prende più
   energia (40,5 contro 33,9) e si muore meno (0,145 contro 0,167). Il tetto
   della scala adattata schiaccia le sorprese grandi e rare (il dolore di un
   morso cattivo), e C impara meno a evitare le cattive.
2. **Memoria contro riadattamento.** Con i codici di C distinti per oggetto
   ciò che si impara su un oggetto resta più a lungo: si sceglie meglio prima
   dell'inversione e si sbaglia più a lungo dopo (0..15 s: 0,29-0,38 contro
   0,45).
3. **Sopravvivenza.** Dalla seconda versione il cervello a sezioni muore di
   fame più dell'attore-critico classico (0,15-0,20 contro 0,10), che pure
   sceglie a caso. Una causa misurata: la bocca resta aperta a vuoto (senza
   niente davanti si morde il 60% delle volte) e mordere costa. Il riflesso
   del morso (proposto, non fatto) la toglierebbe.

## Il critico che insegna, sulla versione attuale (2026-10-29)

| peso del critico | buone | pile/min | energia/min | morti/min |
|---|---|---|---|---|
| 0 (attuale) | 58,9% | 3,54 | 40,5 | 0,145 |
| 0,25 | 58,4% | 2,49 | 28,0 | 0,192 |
| 0,5 | 58,0% | 2,36 | 26,3 | 0,200 |
| 1 | 57,6% | 2,30 | 25,3 | 0,201 |

Peggiora ancora, anche ora che legge O. Quando un oggetto entra nella fovea
la sua previsione scende invece di salire (pila buona −0,13, compagno −0,35):
quei momenti arrivano spesso subito dopo aver finito di mangiare, e il
critico prevede il valore della situazione con questo comportamento, non
quello dell'oggetto.

## K, i contesti (2026-10-29)

Prima versione: un neurone per contesto (4), un cassetto di O per contesto
(è il passo iniziale; l'obiettivo è un contesto sparso su più neuroni).

| | buone | pile/min | buone/min | energia/min | morti/min | inversione |
|---|---|---|---|---|---|---|
| con K | 58,7% | 3,58 | 2,10 | 40,7 | 0,144 | 0,64 · 0,39 · 0,45 · 0,47 · 0,51 |

Sul banco non cambia niente, perché pochi esseri tornano a un regime già
conosciuto, che è il caso in cui K serve. La causa principale è la fame: in
15 minuti le morti sono 2680 di fame e 366 di vecchiaia (12%), e la vita
media è intorno ai 400 s. Alla seconda inversione (600 s) solo il 35% degli
esseri ha vissuto la prima, e più di metà di loro (il 19% del totale, chi è
nato all'inizio) muore di vecchiaia proprio in quel momento: ne resta circa il
16%. Prima di morire, 107 esseri su 360 avevano già il cassetto pronto
(preferenza +0,87 per il colore che sarebbe tornato buono).

Con una vita di un'ora (prova, non il mondo di partenza), tra chi ha già
vissuto un'inversione:

| | torna al vecchio contesto | scelta 0..30 s dopo | 30..120 s dopo |
|---|---|---|---|
| con K | 41% | 0,554 | 0,597 |
| senza K | — | 0,528 | 0,582 |

Il meccanismo funziona (e un test lo verifica in condizioni pulite), ma
l'effetto sulla scelta è piccolo: anche con il cassetto giusto la scelta
resta intorno al 60%, il limite del resto della catena (E usa poco O). Il 27%
degli esseri riempie tutti e 4 i contesti: si recluta anche per rumore.

## Ancora lontano

Il guidato prende 54,7 di energia al minuto e non muore mai; il cervello a
sezioni 40,5 e muore 0,145 volte al minuto. I limiti misurati: E usa poco O
(i pesi dai neuroni di O restano quasi a zero); O si riadatta in minuti, non
in secondi; il morso a vuoto.
