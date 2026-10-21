# Specifica: un mondo nuovo, dal basico al complesso

Stato: **approvata** (2026-10-15). Fatti i punti 1–5 dell'ordine di lavoro:
mondo nei due motori con i test di equivalenza; scala adattiva
(`evo/ladder.py`) e controlli nello stesso lotto (`--task cibo --ladder
--controls`); esame a livelli (versione 4), lesioni (`--lesion`) e misura
della fortuna (`luck` nelle statistiche); interfaccia. Mancano taratura e
primo esperimento; il compito predefinito resta `bersaglio` fino ad allora.

Scelte dell'utente:

- **compito**: si riproduce chi, nella sua vita, mangia almeno N cibi buoni;
- **progressione adattiva** lungo una scala di difficoltà, senza mai far
  estinguere la popolazione;
- **in cima alla scala**: prede che scappano e predatori;
- **due popolazioni di controllo**: una sempre nel mondo complesso, una sempre
  nel mondo basico.

Motivo: nel mondo di oggi il compito è toccare il bersaglio una volta, quindi
chi nasce vicino vince per fortuna e "sporca" la selezione. E il mondo si
risolve con un riflesso (un occhio che gira, esame 46%): un cervello più grande
non serve, quindi non nasce.

---

## 1. Il compito: nutrirsi

- Nell'arena c'è **cibo**: piccoli dischi di un colore (rosso, 0,85 sull'asse
  della luce) che, toccati, vengono **mangiati**. Danno energia
  (`food_energy`, proposta: 40, così nel mondo basico nutrirsi conviene
  chiaramente) e contano per il compito.
- Il cibo mangiato **ricompare** in un punto a caso dell'arena, lontano
  dall'agente che l'ha mangiato.
- **Si riproduce chi mangia almeno N cibi** nella vita (N lungo la scala, da 1
  a 5). Arrivato a N l'agente è in salvo, come oggi chi tocca il bersaglio.
- Gli agenti nascono ad almeno 20 unità da qualsiasi cibo (`food_min_dist`) e
  40 dai predatori.
- La vita ha un tetto (`life_max`, proposta: 120 s), perché mangiando l'energia
  può durare a lungo.

**Perché toglie la fortuna:** una volta si può essere fortunati, cinque volte
di fila no. In natura il successo riproduttivo dipende da quanto ci si nutre in
tutta la vita, non da un incontro fortunato.

**Ognuno vive la propria prova**, come oggi. Ogni agente ha la **sua copia** di
cibo, prede e predatori, che partono uguali per tutti all'inizio della
generazione. Se un agente mangia un cibo, sparisce solo nella sua copia. Niente
competizione tra agenti: è un'altra scelta d'ambiente, che si potrà fare più
avanti.

## 2. Le manopole della difficoltà

Ogni manopola va da 0 (basico) a 1 (complesso). I valori indicati sono
proposte, da tarare con le misure.

| Manopola | 0 (basico) | 1 (complesso) | Cosa richiede |
|---|---|---|---|
| **abbondanza** | 12 cibi, raggio 4 | 4 cibi, raggio 2 (era 3: vedi sezione 9) | cercare in modo efficiente |
| **cibi necessari** (N) | 1 | 5 | affidabilità |
| **prede** | cibo fermo | il cibo si accorge dell'agente entro 15 u e fugge a 12 u/s, fermandosi contro le pareti (la deriva lenta quando l'agente è lontano è rimandata alla taratura) | inseguire, prevedere, aggirare |
| **predatori** | nessuno | 3 predatori che inseguono l'agente entro 40 u a 15 u/s (lui arriva a 30 a spinta piena, ma consuma molto); a contatto tolgono 60 energia al secondo | accorgersi in tempo, fuggire, scegliere tra mangiare e scappare |

**Percepire prede e predatori** non richiede sensi nuovi:

- il **cibo** è rosso (0,85) per la vista;
- i **predatori** hanno un loro colore (proposta: 0,05, quasi blu) e
  **vibrano** muovendosi (frequenza 0,4): si possono vedere e anche sentire,
  utile nell'acqua torbida;
- il veleno resta **spento** in tutta la scala (non scelto per la cima); le
  costanti ci sono, se un giorno vorrai riaccenderlo.

## 3. La progressione adattiva

- Le manopole salgono **una alla volta**, in quest'ordine: abbondanza e cibi
  necessari (insieme), poi prede, poi predatori.
- A ogni generazione si guarda la quota di chi completa il compito, mediata
  sulle ultime 10 generazioni:
  - **sopra il 30%** la manopola corrente sale di 0,02, cioè almeno 50
    generazioni per percorrerla tutta;
  - **sotto il 10%** scende di 0,02;
  - tra i due resta ferma;
  - se in una generazione **nessuno** completa scende di 0,05, e la
    generazione dopo riparte dagli ultimi genitori (come oggi): non ci si
    estingue mai.
- Le soglie sono una nostra scelta: servono solo a non far estinguere la
  popolazione e a non lasciarla in un mondo troppo facile.
- Ogni movimento delle manopole finisce nel **registro dell'ambiente** con la
  sua generazione. L'interfaccia mostra le manopole e il loro andamento.

## 4. Come si misura l'intelligenza

1. **Esame a livelli** (esame versione 4). 8 livelli fissi lungo la scala (da
   tutte le manopole a 0 a tutte a 1), con arene e semi fissi. Ogni esame dà un
   **profilo**: quanto la popolazione riesce a ogni livello. Diventare più
   intelligenti vuol dire migliorare sui livelli difficili **senza perdere**
   quelli facili.
2. **Due popolazioni di controllo**, nello stesso lotto sulla GPU, con lo stesso
   seme di partenza:
   - **sempre complesso**: tutte le manopole a 1 dall'inizio (con la sola regola
     anti-estinzione). Dice se la scala serve davvero;
   - **sempre basico**: tutte a 0. Dice quanto conta il tempo da solo.
3. **Prove di lesione** su richiesta (riga di comando e interfaccia): togliere
   recettori, neuroni interni o la plasticità e rimisurare il profilo. Dicono
   cosa gli agenti usano davvero.
4. **Fortuna sotto controllo**: per ogni generazione si registra se completare
   dipende dalla distanza iniziale dal cibo. Se la dipendenza è forte, la
   fortuna sta ancora pesando.

## 5. Cosa cambia nel codice

- **Il mondo** guadagna oggetti per agente: cibo (posizione, attivo o mangiato,
  velocità), predatori (posizione, velocità), contatori dei cibi mangiati. La
  fisica di prede e predatori è semplice: fuga o inseguimento in linea retta,
  rimbalzo sulle pareti, velocità massima.
- **I sensi**: la luce di cibo e predatori (dischi colorati, uno per agente: la
  stessa geometria del bersaglio di oggi), le vibrazioni dei predatori. Nessun
  gene nuovo nei recettori.
- **Il motore GPU e quello NumPy**: entrambi, con i test di equivalenza come per
  la migrazione.
- **L'evoluzione**: la progressione adattiva, e più popolazioni (scala +
  controlli) nello stesso lotto.
- **L'esame versione 4**, a livelli.
- **L'interfaccia**: cibo, prede e predatori nell'arena; le manopole e il
  profilo dell'esame nella pagina principale; la scia del percorso
  dell'agente seguito.
- Il mondo di oggi (bersaglio e veleno) **resta disponibile** come compito
  separato, per riaprire le corse vecchie.

## 6. Ordine di lavoro

1. Mondo nuovo nel motore NumPy (cibo, prede, predatori, compito), con i test.
2. Lo stesso nel motore GPU, con i test di equivalenza.
3. Scala e progressione adattiva; più popolazioni nello stesso lotto.
4. Esame a livelli, prove di lesione, misura della fortuna.
5. Interfaccia.
6. Taratura: con una popolazione primordiale nel mondo basico, deve completare
   qualcuno ma non troppi (proposta: tra il 5% e il 30%), poi si regolano i
   valori della tabella.
7. Primo esperimento: scala adattiva più i due controlli, qualche migliaio di
   generazioni, 4 semi.

## 7. Limiti noti

- **Ognuno ha la sua copia del mondo**: niente competizione per il cibo né
  vita di gruppo. È una semplificazione voluta, per ora.
- **Prede e predatori sono "stupidi"**: regole fisse, non evolvono. La
  coevoluzione (predatori che evolvono anch'essi) è possibile più avanti.
- **Le soglie della progressione** sono strumenti di addestramento, non natura.

---

## 8. Stato dei lavori (aggiornato al 2026-10-20)

### Fatto

**Punto 1: il mondo nel motore NumPy** (commit `80dc71b`)

- `sim/tasks.py`, classe `Foraging`. Il cibo mangiato dà energia, fino al
  massimo `energy_init`, e ricompare da una coda di posizioni estratta
  all'inizio dell'arena: durante la vita non servono numeri casuali.
- Le prede fuggono dall'agente vicino. I predatori lo inseguono e a contatto
  tolgono energia.
- Ogni agente ha la sua copia di cibo e predatori.
- `sim/signals.py` e `net/receptors.py`: il cibo e i predatori si vedono
  (dischi opachi del loro colore) e i predatori si sentono (vibrazioni).
- `sim/world.py`: il compito si sceglie con `cfg.task` (`"bersaglio"` o
  `"cibo"`). Il compito agisce sull'energia prima che si contino i vivi.
- Corretto un difetto: chi muore per un morso non può mangiare nello stesso
  passo e tornare in vita.
- Numeri decisi strada facendo:
  - un cibo vale 40 di energia, così nel mondo basico nutrirsi conviene
    chiaramente;
  - la vita ha un tetto di 120 s;
  - si nasce ad almeno 20 u dal cibo e 40 dai predatori.

**Punto 2: il mondo nel motore GPU** (commit `80dc71b`)

- `gpu/engine.py` e `gpu/sim.py` fanno la stessa fisica e gli stessi sensi.
- Cibo e predatori entrano nello stato GPU solo nel mondo del cibo: il mondo
  del bersaglio non rallenta.
- Le manopole sono parametri di ogni mondo, quindi popolazioni con manopole
  diverse girano nello stesso lotto.
- I test di equivalenza con il motore NumPy coprono sensi, fisica con prede e
  predatori, passi interi con entrambi i tipi di rete e una generazione
  intera. La prova di fisica comprende migliaia di contatti con i predatori,
  prede in fuga, cibi mangiati e morti.
- Velocità invariata: con 1.000 agenti una generazione dura circa 0,4 s.

**Punto 3: scala adattiva e più popolazioni** (commit `6ffcff4`)

- `evo/ladder.py`: tutte le manopole si riassumono in una **posizione** x da
  0 a 3, una unità per stadio. Lo stadio i ha le sue manopole a
  clip(x − i, 0, 1).
- Gli stadi, nell'ordine: abbondanza e cibi necessari insieme, poi prede,
  poi predatori.
- La posizione si muove con le soglie della sezione 3 (`ladder_*` nella
  configurazione).
- Ogni movimento va nel registro dell'ambiente (evento `scala`) con il motivo
  e la quota che l'ha deciso. Le manopole si salvano con la configurazione.
- I modi (`cfg.ladder`) sono quattro:
  - `"scala"`: la progressione adattiva;
  - `"basico"` e `"complesso"`: i due controlli, con le manopole ferme;
  - `""`: nessuna scala.
- Le statistiche di ogni generazione registrano le manopole, la posizione e
  i cibi mangiati in media.
- Riga di comando:
  - `--task cibo` spegne il veleno;
  - `--ladder` attiva la scala;
  - `--controls` aggiunge, per ogni seme, le due popolazioni di controllo
    nello stesso lotto, con lo stesso seme di partenza
    (file `pop-s<seme>-<modo>.npz`).

**Punto 4: misure** (commit `6ffcff4`)

- **Esame a livelli** (versione 4, `evo/exam.py`): 8 livelli alle posizioni
  0, 3/7, …, 3 della scala, con le manopole scritte nell'esame. Le corse nel
  mondo del bersaglio continuano a fare l'esame versione 3.
- **Lesioni** (`--exam FILE --lesion X`): si può togliere un senso (la
  sensibilità a quel fenomeno viene azzerata in tutti i recettori), i neuroni
  interni o la plasticità. Il risultato è registrato come esame con
  lesione. Gli originali non cambiano.
- **Fortuna**: per ogni generazione, `luck` = quota di chi completa tra il
  quarto nato più vicino al cibo (`near`) e il quarto nato più lontano
  (`far`).

Test: 165, tutti superati (12 nuovi in `tests/test_ladder.py`, 10 in
`tests/test_forage.py`, 5 nuovi in `tests/test_gpu.py`).

**Punto 5: interfaccia**

- **Arena** (`viz/render.py`): cibo (rosso), predatori (blu) con l'area in cui
  inseguono, anello di fuga delle prede quando sono accese, scia del percorso
  dell'agente seguito. Si disegna la copia di mondo dell'agente seguito. Nel
  riquadro in alto: manopole, cibi mangiati su quelli necessari, distanza
  dal predatore più vicino. Lo stesso nel banco di prova.
- **Pagina principale** (`viz/app.py`): grafico della scala (posizione 0–3 con
  gli stadi, cibi mangiati medi, manopole di adesso, fortuna vicini/lontani);
  riquadro dei cibi medi; esame a livelli con l'andamento della media e il
  profilo dell'ultimo esame (tacca bianca = primo esame, punto viola = ultima
  lesione). Gli esami con lesione restano fuori dalla media e non bloccano
  l'esame automatico.
- **Lesioni dall'interfaccia**: tasto `L`, poi un numero per scegliere cosa
  togliere.
- **Menu**: righe "Mondo" (bersaglio o cibo) e "Scala" (adattiva, ferma al
  basico, ferma al complesso). I due controlli insieme alla scala nello
  stesso lotto restano dal terminale (`--controls`): l'interfaccia segue una
  popolazione sola.

### Prima prova: 150 generazioni, seme 7, scala e controlli

Esame alla generazione 150 (i 20 più anziani di ogni popolazione):

| Popolazione | L0 | L1 | L2 | L3 | L4 | L5–L7 |
|---|---|---|---|---|---|---|
| scala | 100% | 90% | 31% | 9% | 6% | 0% |
| sempre basico | 96% | 52% | 7% | 0% | 0% | 0% |
| sempre complesso | 4% | 0% | 0% | 0% | 0% | 0% |

- **La scala serve**: la popolazione della scala supera il controllo basico a
  ogni livello, senza perdere i livelli facili. Quella sempre nel mondo
  complesso non trova mai genitori e riparte ogni volta da reti casuali.
- **La fortuna non pesa più**:

  | Riuscita | quarto più vicino | quarto più lontano |
  |---|---|---|
  | prime generazioni | 51% | 38% |
  | scala, generazioni 135–150 | 27% | 30% |

  Era il problema da cui siamo partiti.
- **Lesioni sulla popolazione della scala** (media dell'esame, intatta: 29%):

  | Tolto | Media | L0 |
  |---|---|---|
  | neuroni interni | 29% (invariata) | 100% |
  | vista | 5% | 40% |
  | senso interno | 25% | 99% |

  Gli agenti usano la vista e quasi nient'altro, e non hanno neuroni interni:
  il comportamento è ancora un riflesso.

### Problemi aperti, da affrontare nella taratura

1. **La scala si blocca alla fine del primo stadio.** Sale fino a 0,92–0,97 e
   poi oscilla, con generazioni quasi senza vincitori (una volta 7 su 1.000).
   Il passo verso "5 cibi da mangiare con 3 cibi in arena" è forse troppo
   duro per l'energia disponibile. Prede e predatori non arrivano mai.
2. **Il mondo non chiede ancora più cervello**: in media da 0 a 0,1 neuroni
   interni.
3. **Il mondo basico si impara subito** (dal 7% al 59% in 15 generazioni).
   Le soglie e il passo della scala sono da rivedere.
4. **Il controllo "sempre complesso" dice poco**: mostra solo che la scala
   serve, perché la popolazione non si avvia mai.
5. **La deriva lenta delle prede** quando l'agente è lontano non è ancora
   fatta.

### Da fare

- **Punto 6, taratura**, partendo dai problemi aperti qui sopra.
- **Punto 7, primo esperimento**: scala e controlli, qualche migliaio di
  generazioni, 4 semi. Poi il compito predefinito passa da `bersaglio` a
  `cibo`.

---

## 9. Taratura 1: il gradino dei cibi e gli odori (2026-10-20)

### Cosa non quadrava

La corsa `corsa-20261020-130439` (rete a impulsi, 700 individui, scala
adattiva, 1.537 generazioni) si è fermata per 1.300 generazioni attorno alla
posizione 0,94, e l'esame non migliorava più dalla generazione 200.

| Cibi in arena | Da mangiare | Generazioni | Riescono |
|---|---|---|---|
| 5 | 4 | 39 | 26% |
| 4 | 5 | 769 | 25% |
| 3 | 5 | 510 | 14% |
| 3, prede accese | 5 | 175 | 15% |

- **Un gradino, non una salita**: togliere un cibo su 4 tagliava la riuscita
  di quasi metà. La scala saliva con 4 cibi e ricadeva con 3.
- **Il cibo non si vedeva**: con l'attenuazione della luce a 12 u il cibo è
  visibile entro 20–30 u, ma con 3 cibi il più vicino è in media a 54 u (entro
  20 u solo nel 12% dei casi). La ricerca era quasi alla cieca, quindi un
  cervello più grande non serviva.
- Il corpo era rimasto minimo (un recettore usato, generalista; 1–2 neuroni) e
  i tassi di mutazione erano scesi (ritocco dei pesi da 10% a 0,6%).

### Scelte dell'utente

- **Il primo stadio si ferma a 4 cibi** (`food_count_hard` = 4).
- **Odori**: il cibo rilascia una molecola sua (0,6), i predatori un'altra
  (0,35). Nube stabile `forza · e^(−d/λ)`, d dal bordo, λ = 25 u
  (`odor_range`); le nubi si sommano. È la forma di equilibrio tra diffusione
  e degradazione (come la "ficosfera" attorno alle alghe).
- **La nube che sfugge è una difficoltà** (`forage_scent`, nello stadio 2
  insieme alle prede): a 0 la nube è sempre in equilibrio; salendo, la nube di
  un cibo appena comparso si forma come 1 − e^(−t/τ) e quella di un cibo
  mangiato resta dov'era e svanisce come e^(−t/τ), con τ = manopola × 3 s
  (`scent_time_max`).
- Nessun gene nuovo: un recettore sensibile alle sostanze con l'adattamento
  sottrattivo (gene già esistente) risponde ai cambiamenti, come il naso di un
  batterio.

### Cosa cambia nel codice

- `sim/tasks.py`: `Foraging.odors` (le nubi nei punti del corpo di ogni
  agente, nella sua copia di mondo) e `Foraging.scent` (forza di nube e
  traccia); per ogni posto di cibo l'età, la traccia e da quanto è stata
  lasciata.
- `sim/signals.py`, `net/receptors.py`: le sostanze ricevono anche gli odori
  del compito.
- `gpu/engine.py`, `gpu/sim.py`: lo stesso, con i test di equivalenza.
- `evo/ladder.py`: lo stadio 2 è "prede e odore che sfugge".
- `evo/exam.py`: **esame a livelli versione 5** (4 cibi al massimo, odori
  accesi, nube che sfugge nei livelli dello stadio 2). I punteggi della
  versione 4 non si confrontano: la panoramica confronta solo esami della
  stessa versione.
- Salvataggi fatti prima degli odori: oggi riprendono con le regole più
  recenti (vedi sezione 12).
- Interfaccia: aloni d'odore e tracce nell'arena; la nube nello stato della
  scala.

### Da fare

- Misurare, con una popolazione primordiale, quanto aiuta l'odore nel mondo
  difficile (prima di lanciare un esperimento lungo).

---

## 10. Taratura 2: dare peso alla bravura (2026-10-20)

### Cosa non quadrava

Nella corsa `corsa-20261020-144156` (con odori, 2.701 generazioni) la scala si
è fermata a 1,7 per 2.000 generazioni e l'esame non migliorava. Misurando ogni
individuo su 30 arene (bravura vera):

- la bravura vera variava di appena ±5% tra individui, ma la difficoltà
  dell'arena di una generazione andava dal 2% al 53%;
- in una prova sola i vincitori erano più bravi dei perdenti di 3 punti: la
  selezione vedeva quasi solo la fortuna, e i figli perdevano per mutazione
  quello che i genitori guadagnavano (equilibrio, e tassi di mutazione che
  crollano perché mutare fa quasi solo danni);
- l'odore non era usato (esame senza sostanze identico), un solo recettore
  (un occhio a sensibilità 60), 0–1 neuroni.

### Scelte dell'utente e misure

| Cambio | Misura sulla popolazione di quella corsa |
|---|---|
| **un'arena per agente** (`forage_own_arena`) | dev. std della riuscita tra arene di una generazione: da 12,6% a 1,6% |
| **fascia della scala 30–50%** (era 10–30%) | correlazione tra esito di una prova e bravura vera: 0,13 al 42% di riuscita contro 0,07 al 19% |
| **vite più lunghe, fino a 10 cibi in 240 s** | da sola nessun effetto: si moriva di fame verso i 50 s (2,6 cibi in 120 s come in 240 s) |
| **cibo da 100 di energia** (era 40) | 5,3 cibi in una vita; con 10 cibi necessari il 20% ce la fa: la cima è raggiungibile |

Limite: la prova singola non diventa più informativa con vite più lunghe (0,14
a parità di riuscita) finché la popolazione è quasi uniforme. Il collo di
bottiglia è anche la poca varietà.

Esame a livelli versione 6 (fino a 10 cibi in 240 s, cibo da 100, arena per
agente). Le corse salvate prima riprendono con le regole più recenti (sezione 12).

---

## 11. Recettori nuovi come geni duplicati (2026-10-20)

Nella corsa `corsa-20261020-200723` (prima corsa con la taratura 2) gli agenti
hanno sviluppato un naso: un recettore delle sostanze sintonizzato sulla
molecola del cibo (0,58), sensibilità circa 60, adattamento sottrattivo 0,71
(chemiotassi). Esame senza sostanze: 0%; senza luce o senza neuroni: invariato.
Non nascevano altri recettori: in circa 400 duplicazioni la copia intera di un
naso costava subito il 20% del metabolismo di base e veniva eliminata.

Scelta dell'utente: duplicare "il più possibile come in natura". Si duplica il
gene, non l'organo (`receptor_dup_split`, regole in docs/REGOLE.md, punto 15):
la copia prende una cellula e le sue sinapsi, l'originale le perde. Prima
versione con i pesi interi: non neutrale (dal 46% all'11%), perché un
recettore con adattamento divisivo risponde al contrasto e una cellula sola
manda un segnale grande quanto l'organo, solo più rumoroso. Con le sinapsi che
seguono le cellule: 44,8% contro 46,0%.

---

## 12. Ogni ripresa con le regole più recenti (2026-10-20)

Scelta dell'utente: ogni salvataggio caricato gioca con le regole più recenti,
anche in futuro, così ogni corsa si può sempre continuare. Dal file vengono
genomi, storia e lo stato della corsa (`RUN_STATE`: rete, popolazione, posti,
mondo scelto, modo e posizione della scala, frequenza dell'esame); tutto il
resto dal codice. Le regole cambiate sono nel registro della ripresa (campo
`rules`, [valore salvato, valore di adesso]). Prima, al contrario, si
riprendeva con la configurazione salvata e le regole nuove restavano spente.

Prova sui quattro salvataggi in `runs/` (una generazione ciascuno): tutti
ripartono. La corsa `corsa-20261020-130439` (adattata al mondo vecchio, scala
a 0,94) completa 1 su 700: con le regole nuove la stessa posizione vuol dire 9
cibi da mangiare invece di 5, in un'arena per agente. La scala scenderà da
sola.

---

## 13. Incrocio tra omologhi e sesso come gene (2026-10-20)

Scelte dell'utente, dopo la revisione di cosa restava innaturale:

- **Incrocio tra omologhi**: ogni gene di recettore e di neurone porta identità
  e famiglia (formato dei salvataggi 8; i vecchi ricevono l'identità del
  posto). Si appaiano i geni con la stessa identità; i collegamenti seguono
  l'identità delle sorgenti; un gene di un solo genitore passa con
  probabilità 1/2. Prima si abbinavano i posti per numero, e una copia nata in
  una linea poteva essere incrociata con un gene estraneo.
- **Scambio ineguale** ("il più simile possibile alla natura"): tra copie della
  stessa famiglia, al più uno per famiglia e per meiosi, con probabilità tasso
  di duplicazione × (n − 1) × somiglianza; una copia in più o in meno. Una
  prima versione con uno scambio per ogni coppia di copie faceva esplodere i
  cervelli anche senza selezione (71 neuroni in media invece di meno di 4).
- **Sesso come gene della madre** (`sex`, partenza 0,5), senza costi per ora.

Sulla popolazione di `corsa-20261020-200723`: 400 figli in 0,04 s.

## 14. Quarto stadio: le rocce (2026-10-21)

Richiesta dell'utente: aggiungere ostacoli al mondo incrementale, nello
stadio ritenuto migliore. Scelta: **un quarto stadio dopo i predatori**
(`forage_obstacles`, posizione della scala da 0 a 4).

- Perché dopo i predatori: le corse in corso (al più allo stadio dei
  predatori) si riaprono senza cambi improvvisi (con la scala da 0 a 3 le
  loro posizioni restano le stesse); in natura le rocce danno senso ai
  predatori (rifugi: un predatore che insegue dritto ci si blocca contro);
  nascondono il cibo alla vista, quindi l'olfatto (che le scavalca) e la
  memoria diventano utili; un mondo con punti di riferimento fissi per
  tutta la vita è il primo passo verso un mondo in cui imparare conviene.
- Fisica: dischi fermi, 8 al massimo, raggio 6–16, che non si sovrappongono
  e lasciano un passaggio di almeno 10 u (`rock_gap`); bloccano agenti, prede
  e predatori (spinta sul bordo, rimbalzo come le pareti), premono sul lato
  che le tocca, sono opache alla luce (grigie, 0,45, albedo 0,4). Il cibo non
  nasce né ricompare dentro una roccia. Estratte solo quando ci sono: senza
  rocce il mondo è identico a prima.
- Nei due motori (NumPy e GPU), con il confronto passo per passo nei test.
- Esame a livelli: versione 7 (i livelli coprono la scala da 0 a 4).

## 15. Quinto stadio: i frutti che cambiano da una vita all'altra (2026-10-21)

Proposta dell'assistente, scelte dell'utente: due frutti che si distinguono
per colore e odore; il cattivo dà dolore e toglie poca energia.

- Perché: in natura l'apprendimento si evolve dove c'è qualcosa di stabile
  dentro una vita ma diverso tra una vita e l'altra (Stephens 1991, effetto
  Baldwin). Con un mondo sempre uguale basta un riflesso, e così era finora.
  Qui nessun riflesso fisso può sapere quale colore mangiare: va imparato
  ogni volta, dal gusto e dal dolore attraverso il sistema di valore, come le
  api imparano il colore dei fiori che danno nettare.
- Fisica: rosso (0,85, molecola 0,6) e giallo (0,65, molecola 0,5), stessa
  forma; in ogni vita, per ogni agente, il colore buono è scelto a caso
  (`good`), e una quota dei posti (0,5 · manopola) ha il frutto cattivo, di
  colore opposto. Il buono nutre (100), dà gusto, conta per riprodursi; il
  cattivo non nutre, dà dolore, toglie 30 di energia. Due occhi gialli nel
  set di sensi fisso. Nei due motori, con il confronto passo per passo.
- Statistica nuova: frutti cattivi mangiati (`bad_eaten`): se gli agenti
  imparano, a parità di frutti buoni gli errori calano.
- Esame a livelli: versione 8 (scala da 0 a 5).

