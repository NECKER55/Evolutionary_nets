# Specifica: l'arena continua

Stato: **prima versione implementata** (2026-10-23): `sim/arena.py` (il
mondo), `evo/ecology.py` (nascite, morti, misure, salvataggi),
`viz/arena_view.py` (la schermata), `gpu/arena.py` (la GPU), `uv run main.py
--task arena`. Con il motore "gpu" (quello di partenza) sensi, cervello,
plasticità, movimento, urti, pasti e fame girano sulla scheda video, un
secondo di mondo per volta; il motore "numpy" resta come riferimento (i test
confrontano i due). Le modalità a generazioni non cambiano: stesso seme,
stessi numeri di prima.

Una nuova modalità del mondo, accanto a quella a generazioni (non al suo
posto): tutti gli esseri vivono insieme in un'unica arena grande, il tempo
scorre senza generazioni, si nasce quando si ha abbastanza energia e si muore
di fame o di vecchiaia. Nessun compito da completare: si riproduce chi
sopravvive e accumula energia, come in natura.

## 1. Perché

Nelle corse a generazioni il mondo premia solo i riflessi: togliendo la
plasticità l'esame non cambia (23% contro 24%), e la sola fase che chiede di
imparare (i frutti) è un salto, non una salita, e nessuna popolazione ci
arriva. In un mondo condiviso gli altri esseri, che evolvono, e il cibo che si
sposta rendono l'ambiente imprevedibile: ciò che conviene fare non si può
scrivere tutto nei geni, e imparare durante la vita diventa utile. La
coevoluzione sposta di continuo ciò che conviene: le valli di oggi possono
diventare discese domani senza che le progettiamo noi.

## 2. Scelte concordate

| tema | scelta |
|---|---|
| bordi | nessuno: mondo toroidale, chi esce da un lato rientra dall'altro |
| tempo | continuo, senza generazioni né vite a tempo |
| riproduzione | da soli quando l'energia basta; con un partner se un compatibile è vicino |
| interazioni (primo passo) | si vedono, si annusano, si urtano, si contendono il cibo |
| cibo | ricresce a macchie fertili che si spostano e cambiano con le stagioni |
| individui iniziali | dal menu (riga "Individui"), come oggi |
| tetto | dal menu (riga nuova "Tetto"), per evitare l'esplosione |
| sensi, rete, genoma | dal menu, come oggi; il primo esperimento con il genoma diretto |

## 3. Spazio

- **Arena 2000 × 1400** (100 volte quella di oggi), senza bordi.
- **Perché questa misura.** La luce si attenua come exp(−d / 12): oltre circa
  70 u non si vede quasi più niente, l'odore arriva a circa 25 u. Con 1000
  esseri ognuno ha in media circa 2800 u² (un quadrato di 53 u di lato): nel
  suo raggio di vista ci sono di solito 1–4 altri esseri. Abbastanza per
  incontrarsi e contendersi il cibo, non tanti da vivere ammassati.
- **Griglia dei vicini**: celle di 80 u (più del raggio di vista). Vista,
  olfatto, urti e ricerca del partner guardano solo le celle vicine: il costo
  cresce con i vicini, non con tutti gli oggetti del mondo.
- **Bordi toroidali nei sensi**: distanze e raggi di luce si calcolano "da
  vicino" attraverso il bordo (gli oggetti entro 80 u da un bordo compaiono
  anche dall'altro lato per chi guarda).

## 4. Energia, vita e morte

Restano le regole di oggi: energia iniziale 150, riserve fino a 450,
metabolismo di base 2,5/s, movimento fino a 10/s in più, costo di sensi e
neuroni, cibo che dà 100.

- **Fame**: energia a zero, si muore.
- **Vecchiaia** (legge di Gompertz, come negli animali): la probabilità di
  morire cresce con l'età, h(età) = h₀ · e^(età / τ), con τ = 150 s e
  h₀ = 8,6 · 10⁻⁵ /s. Vita mediana per vecchiaia circa **600 s** (10 minuti,
  2,5 volte la vita di oggi); entro 300 s muore per vecchiaia circa l'8%,
  entro 750 s circa l'85%, entro 900 s quasi tutti. Niente tetto netto.
- **Perché 600 s**: abbastanza per imparare e sfruttare ciò che si impara
  (oggi una vita è 240 s), abbastanza breve perché le generazioni si
  susseguano (circa 6 per ora di mondo).

## 5. Riproduzione

- **Quando**: energia ≥ **250** (era 300, sezione 13), età
  ≥ **30 s** (maturità), e almeno **30 s** dall'ultimo figlio.
- **Costo**: il figlio riceve **120**; il genitore ne paga **160** (erano 150
  e 200): in natura fare un figlio costa più dell'energia che il figlio
  riceve. Dopo il parto il genitore resta con circa 90: deve tornare a
  mangiare.
- **Dove**: il figlio nasce accanto al genitore (a 3 u, direzione a caso).
- **Partner**: se entro **20 u** c'è un essere maturo e compatibile, con la
  probabilità del gene `sex` del genitore il figlio è un incrocio dei due
  (l'incrocio tra omologhi di oggi); il partner non paga energia. Altrimenti
  è una copia mutata. **Compatibile**: i due condividono almeno metà delle
  identità dei geni (la discendenza c'è già nel genoma): se due gruppi
  divergono troppo smettono di incrociarsi, come specie diverse.
- **Mutazioni**: quelle di oggi, alla nascita.
- **Tetto**: se tutti i posti sono occupati il figlio non nasce (il genitore
  non paga). Il tetto è una sicurezza tecnica: la popolazione vera la decide
  il cibo (sezione 6).
- **Cervello alla nascita**: come oggi; con lo sviluppo il figlio passa dal
  grembo.

### Colonizzatori (aggiunti nell'implementazione)

Partendo da reti nuove quasi nessuno mangia abbastanza per fare un figlio
(nel primo minuto di prova 717 morti di fame su 750): senza aiuto la
popolazione si estingue prima di imparare. Come in un'isola vicino a un
continente (MacArthur e Wilson), finché i vivi sono meno di
`arena_min_alive` (100) arrivano **colonizzatori**: copie mutate dei migliori
già vissuti (archivio dei 50 che hanno mangiato di più, scelti con
probabilità proporzionale al cibo mangiato), o reti nuove se nessuno ha
ancora mangiato. Nascono nelle macchie con l'energia iniziale. Quando la
popolazione si mantiene da sola smettono di arrivare. `arena_min_alive` = 0
li spegne.

## 6. Cibo a macchie

- **Macchie fertili**: **8** zone, ognuna una campana di raggio tipico
  (σ) **160 u** (era 120, sezione 13). Il cibo ricresce in un punto a caso con probabilità
  proporzionale alla fertilità lì: dentro una macchia si trova molto cibo,
  fuori quasi niente (un fondo dello 0,05 della fertilità massima, per non
  lasciare deserti assoluti).
- **Si spostano**: ogni macchia deriva a **0,1 u/s** in una direzione che
  cambia piano (passeggiata casuale lenta). In una vita mediana si sposta di
  circa mezza macchia (60 u): ciò che un essere impara su dove c'è cibo resta
  utile, ma va aggiornato.
- **Stagioni**: la fertilità di ogni macchia oscilla tra 0,3 e 1 con un
  periodo di **1200 s** (circa due vite), ogni macchia con una fase sua: le
  zone ricche cambiano nel tempo, e chi resta fermo in un posto lo vede
  impoverirsi.
- **Quanto cibo**: ricrescono **45 cibi al secondo** in tutto il mondo, con un
  massimo di **1500** cibi presenti insieme.
  - Il calcolo: un essere che cerca spende in media circa 4–5 energia/s (le
    vite di oggi: circa 8 cibi in 240 s più le 150 iniziali) e, in
    equilibrio, deve fare in media un figlio per vita (200 in 600 s, circa
    0,3/s). Ogni essere consuma quindi circa 0,045 cibi al secondo, e
    45 cibi/s mantengono circa **1000 esseri**: un terzo del tetto di
    partenza (3000), con spazio per oscillazioni.
  - Il massimo di cibi presenti impedisce che, con pochi esseri, il mondo si
    riempia di cibo e la fame sparisca.
- **Cibo**: rosso, raggio 3, profumo come oggi (la nube d'odore che si forma
  e svanisce resta).

## 7. Gli altri esseri

- **Si vedono**: ogni essere è un disco opaco (raggio 1) con un colore suo,
  0,25 sull'asse della luce (lontano dal rosso del cibo, 0,85, dal giallo dei
  frutti, 0,65, e dal blu dei predatori, 0,05).
- **Si annusano**: ogni essere emette una molecola sua, 0,45, debole
  (forza 0,3 rispetto al cibo), con la stessa legge dell'odore del cibo.
- **Si urtano**: due esseri non si sovrappongono, si spingono come dischi
  morbidi.
- **Si contendono il cibo**: chi lo tocca per primo lo mangia; se due lo
  toccano nello stesso passo, lo mangia il più vicino.
- **Non ancora**: predazione tra esseri, segnali, predatori, rocce, frutti.
  Si aggiungono uno alla volta, dopo aver visto il primo passo.

## 8. Menu

- **Mondo**: una scelta nuova, "arena continua".
- **Individui**: quanti esseri all'inizio (come oggi), sparsi nelle macchie
  con energia iniziale.
- **Tetto**: riga nuova, numero massimo di esseri (proposta **3000**). Visibile
  solo con l'arena continua.
- **Simulazione, genoma, rete, seme**: come oggi.
- **Scala**: disattivata (la difficoltà la danno il cibo e gli altri).
- **Generazioni**: diventa "Minuti di mondo" (dopo quanti fermarsi e salvare;
  0 = senza limite).
- **Esame ogni**: in minuti di mondo (proposta **10**).

## 9. Misure

Ogni **60 s di mondo** una riga di storia:
- esseri vivi, nati, morti di fame e di vecchiaia, figli con partner;
- età media e massima, energia media;
- cibo presente, cibo mangiato al minuto;
- sensi, neuroni, collegamenti medi, geni delle mutazioni (come oggi);
- varietà: numero di gruppi compatibili (quasi specie).

**Esame**: ogni 10 minuti di mondo l'esame fisso di oggi su 20 esseri adulti
presi a caso (età ≥ 60 s): il punteggio si confronta con le corse a
generazioni. Più avanti la **prova dell'apprendimento**: lo stesso esame con e
senza plasticità, e una prova in cui la macchia di cibo si sposta a metà vita.

**Campioni**: ogni 60 s il più anziano vivo, per la scheda delle reti.

## 10. Interfaccia

- **Vista che segue** un essere, con zoom (rotella) e la scia.
- **Mappa intera** in piccolo: esseri come punti, macchie fertili come
  ombre, il riquadro della vista.
- Panoramica e scheda reti come oggi, con le misure della sezione 9 al posto
  di quelle per generazione.

## 11. Motore

**Come è fatto (2026-10-23).** Sulla GPU (`gpu/arena.py`, JAX) gira un
secondo di mondo intero (60 passi) per chiamata: griglia dei vicini rifatta
a ogni passo (al più 16 oggetti per cella e per tipo), sensi, cervello e
plasticità (le stesse funzioni del motore a generazioni), movimento sul
toro, urti, pasti, morti di fame. Sul processore, una volta al secondo:
vecchiaia, ricrescita del cibo e macchie, nascite e colonizzatori; i
neonati arrivano sulla GPU riga per riga, senza ricompilare. Con 750 esseri
a impulsi su 3000 posti (RTX 5070): circa 70 ms di GPU e 30 ms di
processore per secondo di mondo, **circa 11 volte il tempo reale**, contro
0,7 volte sul processore. I sensi della GPU coincidono con quelli del
processore a meno di qualche per cento in rari recettori: la GPU cerca i
vicini attorno al centro dell'essere, il processore attorno al recettore.

### Il progetto iniziale

- **Posti fissi** sulla GPU (quanti il tetto) con una maschera dei vivi:
  nascere è occupare un posto libero, morire è liberarlo. Le dimensioni non
  cambiano mai, così il codice compilato resta lo stesso.
- Il passo del mondo resta 1/60 s.
- Nascite, morti e mutazioni a ogni secondo di mondo (non a ogni passo), in
  NumPy; il resto sulla GPU.
- Il motore NumPy di riferimento fa lo stesso, e i test confrontano i due come
  oggi.

## 12. Prima misura (2026-10-23)

Rete continua, genoma diretto, sensi che evolvono, 750 esseri all'inizio,
30 minuti di mondo, sul processore circa 5 volte il tempo reale con 100 vivi:

| minuto | vivi | nati/min | colonizzatori/min | morti di fame/min | sensi usati | esame |
|---|---|---|---|---|---|---|
| 1 | 100 | 21 | 44 | 713 | 1,16 | |
| 10 | 100 | ~60 | ~40 | ~100 | 0,5 | 6% (L0 50%) |
| 30 | 100 | 54 | 51 | 105 | 0,42 | 6% (minuto 20) |

La popolazione non si regge ancora da sola: resta al minimo dei
colonizzatori. Le nascite salgono da 21 a circa 60 al minuto e poi si
fermano. Il cibo è sempre al massimo (1500): non manca, ma gli esseri
corrono a tutta velocità (costa 5 volte il riposo) e muoiono di fame in
circa un minuto, prima di arrivare all'energia per un figlio (300). I sensi
si perdono: dentro le macchie il cibo è così fitto che correre alla cieca
basta, e i sensi costano. Da ricalibrare (vedi sotto).

## 13. Ricalibrazione (2026-10-23)

Sulla GPU, rete a impulsi, genoma diretto, 750 esseri all'inizio, seme 11.
Prima tornata, 20 minuti, energia del cibo al secondo uguale per tutti salvo
dove indicato:

| variante | si regge da sola? | sensi usati, min. 20 |
|---|---|---|
| base (300/150/200, macchie 120) | quasi: colonizzatori da 66 a 0–17, vivi ~100 | 1,67 |
| metà cibi da 200 | no (~60 colonizzatori/min) | 1,86 |
| metà cibi da 200, macchie 200 | quasi, lenta | 0,87 |
| figlio a 250 (120/160) | **sì dal minuto 18: 220 vivi** | 0,56 |
| le due insieme | no | 1,52 |
| un terzo dei cibi da 300, macchie 200 | no | 1,32 |

Seconda tornata, 40 minuti, tutte con il figlio a 250:

| variante | vivi (min. 20 → 24 → 40) | colonizzatori | sensi usati, min. 40 |
|---|---|---|---|
| macchie 120 | 220 → 344 → 143 (esplosione e crollo) | 0 dal min. 18 | 0,31 |
| **macchie 160** | 142 → 146 → 214 (crescita regolare) | 0 dal min. 20 | 0,44 |
| macchie 120, 30 cibi/s | ~100 sempre | ~60/min | 1,36 |

Il cibo è sempre al massimo: non lo limita la quantità ma l'abilità di
trovarlo. Rendere il cibo più raro peggiora l'avvio. Nuovi valori: figlio a
250 (120 al figlio, 160 al genitore), macchie di raggio 160. Resta un
problema: quando la popolazione decolla vincono le famiglie che corrono alla
cieca dove il cibo è fitto e i sensi calano. Il mondo da solo non chiede di
più: va complicato (proposte in docs/NOTE.md o nella prossima specifica).

## 15. Il mondo che si complica (2026-10-23)

### Perché

Con la popolazione che si regge da sola vincono le famiglie che corrono alla
cieca dove il cibo è fitto, e i sensi si perdono: il mondo non chiede altro.
In natura l'intelligenza cresce dove le risorse sono sparse e cambiano (nei
primati la dieta prevede la grandezza del cervello meglio del gruppo,
DeCasien 2017), dove ci sono altri da prevedere, dove il cibo va ottenuto in
più passi e dove c'è tempo per imparare. Le novità, una per ciascuna di
queste pressioni, si accendono da sole e per gradi.

### Il corpo (solo nell'arena)

- **Taglia**, un gene (in scala logaritmica, tra 0,3 e 4): raggio × √taglia,
  metabolismo × taglia^0,75 (Kleiber), riserva di energia × taglia, spinta
  × taglia^-0,25. Più grande: morso più forte, più riserva, ma più lento e
  più caro.
- **Quattro azioni**, ciascuna comandata da un neurone motore in più (le
  reti dell'arena hanno 6 o 8 motori; le altre arene non cambiano):
  **mordere**, **luce** (del colore del proprio gene), **odore** (della
  propria molecola), **nutrire**. Mordere costa 0,5/s, luce e odore 0,3/s
  ciascuno a intensità piena.
- I geni del corpo si ereditano (ognuno da uno dei genitori) e mutano con la
  probabilità di mutazione del genitore.

### Le dieci manopole, nell'ordine

| | manopola | cosa fa | che cosa chiede |
|---|---|---|---|
| 0 | `arena_scarcity` | il cibo diventa raro: ricrescita fino al 40% (18 al secondo) e al più il 35% presenti insieme (525) | arrivare al cibo prima degli altri, vederlo da lontano, non sprecare energia |
| 1 | `arena_thirst` | sete: la traspirazione sale dal 25% (senza bere ~560 s fermi) al 100% (~140 s) | fare la spola tra cibo e acqua, scegliere quale bisogno prima |
| 2 | `arena_pool_distance` | le pozze nuove nascono più lontane dal cibo: dal centro della macchia più vicina da 230–300 u fino a 260–450 u | andare a bere lontano e tornare, orientarsi, ricordare dov'è l'acqua |
| 3 | `arena_depletion` | ogni cibo mangiato vicino a una macchia ne toglie l'1% della salute (per la manopola); la salute torna in 500 s; la fertilità è la salute | cercare un'altra macchia, ricordare dove erano |
| 4 | `arena_fruits` | fino a metà del cibo nuovo è un frutto cattivo, del colore che nella sua macchia non è buono (toglie 30, fa male) | distinguere i colori ed evitare |
| 5 | `arena_reversal` | ogni 10 minuti ogni macchia ha il colore buono giallo invece di rosso con probabilità fino a ½ | imparare durante la vita quale colore è buono qui |
| 6 | `arena_predation` | si morde un essere toccato davanti (entro 60°) e almeno 1,2 volte più piccolo: gli si tolgono 40 · morso · taglia di energia al secondo, chi morde ne prende il 60% | cacciare, fuggire, prevedere l'altro |
| 7 | `arena_extraction` | fino a metà del cibo nuovo è da aprire: **duro** (va morso per 1 s a morso pieno, vale il doppio) o **grande** (va morso da almeno due insieme, vale 6 volte, diviso tra loro) | azioni in più passi, coordinarsi |
| 8 | `arena_night` | un giorno e una notte ogni 10 minuti: la luce del mondo cala fino a 1 − 0,85; la luce dei segnali no | olfatto, memoria, segnali |
| 9 | `arena_drought` | siccità: le pozze evaporano (piene, in 600 s con la manopola al massimo) | trovare pozze nuove, ricordare dove sono |

Il cibo nasce **solo nelle macchie** (`arena_background` = 0 dal 2026-10-23; era
0,05). Il cibo raro fin dall'inizio non regge: con 450, 550 o 700 cibi al
massimo e senza fondo la popolazione non si è mai retta da sola in 27 minuti
(0–8 nati al minuto, 180–290 colonizzatori al minuto, il cibo sempre tutto lì
da trovare). Per questo è la prima manopola della scala, non un valore fisso.

### Il cibo a ondate

Il cibo non ricresce un po' ogni secondo: matura 30 al secondo e nasce tutto
insieme ogni 60 s (`arena_food_every`), fino al massimo di 1500, solo nelle
macchie. Tra un'ondata e l'altra si consuma: conviene esserci quando ricresce.
Calibrazione provvisoria (prove interrotte a 5–6 minuti di mondo, popolazione
ancora tenuta dai colonizzatori): ogni 60 s con 30/s e 1500 ha avuto meno
colonizzatori (90 al minuto) di 30 s / 30 / 900 (153), 60 s / 20 / 1000 (141),
30 s / 20 / 600 (202).

### Neuroni nuovi già collegati

Nell'arena (`arena_wire_new`, genoma diretto) un neurone interno che compare in
un figlio nasce collegato: in entrata e in uscita, con la probabilità della
nascita (cala con la distanza nel cervello e con la portata dell'individuo).
Prima nasceva senza collegamenti e doveva aspettarli dalle mutazioni (1% per
collegamento): in 40 minuti i vivi avevano 0,05–0,2 neuroni interni.

### Esseri primordiali con 2 neuroni interni

Gli esseri primordiali (all'inizio e i colonizzatori nuovi quando l'archivio
è vuoto) nascono con 2 neuroni interni accesi e collegati con la regola della
nascita (`arena_hidden_initial`), invece di nessuno. Serve a capire se il
limite è la mancanza di neuroni o la mancanza di motivi per usarli: se il
mondo non li premia, la selezione li spegne.

### Il clima

Una temperatura da 0 a 1: una fascia calda al centro del mondo, fresca verso
i bordi (in alto e in basso, che sul toro si toccano), 0,5 ± 0,4, più ± 0,1
con le stagioni (`arena_temperature`, `arena_temp_season`).
- **Metabolismo**: tutto il consumo per 3^(T − 0,5) (Q10 degli ectotermi): da
  0,58 volte al freddo a 1,73 al caldo (`arena_temp_q`).
- **Cibo**: la fertilità di una macchia per 1 + 1,5 · (T − 0,5) dove si trova
  (`arena_temp_food`): da 0,25 a 1,75 volte. Le macchie si spostano,
  attraversano zone calde e fredde.
- **Sensi**: i recettori della temperatura sentono quella del punto dove
  stanno.
Al caldo c'è più cibo ma si brucia di più; al freddo si risparmia ma si mangia
poco. Il clima c'è sempre (non è una manopola della scala), solo con il motore
JAX.

### L'acqua

Una seconda riserva del corpo, accanto all'energia (solo motore JAX,
`arena_water`). Scelte concordate il 2026-10-23: sete subito ma leggera, pozze
ferme, in posti diversi dal cibo ma non troppo lontani, consumate piano piano,
che riappaiono quando sono poche; un senso interno nuovo per la sete.
- **Riserva**: 100 · taglia. Cala di 0,7/s (per la manopola della sete: da
  un quarto a tutto) · taglia^0,75 · (1 + 0,5 · |spinta|)
  · lo stesso Q10 del clima: senza bere si resiste ~140 s fermi, ~55 s correndo
  al caldo. A zero si muore di sete (contati a parte da fame e predazione).
- **Pozze**: 6 all'inizio, su 10 posti. Ferme. Nascono con il centro della
  macchia di cibo più vicina tra 230 e 300 u (appena fuori dal bordo della
  macchia, che con la pozza fa 220) e lontane da altre pozze; la manopola
  "pozze lontane" porta le nuove fino a 260–450 u (a velocità piena 9–15 s). Volume 60000; raggio 60 · √(volume / pieno): si
  restringono bevendo. Vuote spariscono; con meno di 4 pozze ne nasce una
  nuova (in media dopo 60 s). Con la siccità evaporano anche da sole.
- **Bere**: stando dentro una pozza, 25/s fino al pieno; se in tanti chiedono
  più di quanto resta, si divide.
- **Figli**: il genitore paga 40 d'acqua e deve restargliene il 10% della
  riserva; il figlio nasce con la riserva piena (idratato dal genitore, come
  con il latte o il tuorlo).

Prima versione (pozze a 260–450 u, figlio con 40 d'acqua, servivano 40 + 20%):
in 40 minuti di mondo la popolazione non si è mai retta da sola (80–96% di
colonizzatori, nessun nipote, nessun adulto pronto a riprodursi, 30–60% dei
morti senz'acqua); la stessa corsa senz'acqua si reggeva dal minuto 12 con
29 generazioni. Analisi del 2026-10-24.
- **Sensi**: le pozze si vedono (blu, 0,1 sull'asse della luce; piatte, non
  nascondono niente; dentro si vede acqua tutt'attorno), si annusano
  (umidità, molecola 0,15, portata 100 u contro i 25 del cibo), e la riserva
  si sente da dentro: una grandezza nuova sull'asse dell'interno, a 0,11, tra
  l'energia (0,05) e il gusto (0,17). Un recettore interno generalista la
  sente insieme all'energia; specializzandosi distingue fame e sete.

Sempre presenti, come capacità: i **segnali** (luce e odore di ciascuno, che
gli altri vedono e annusano) e le **cure** (chi nutre passa 20 energia/s a un
proprio figlio toccato non ancora adulto, che ne riceve l'80%).

### La scala del mondo

Ogni minuto: se negli ultimi 3 minuti non è servito nessun colonizzatore e i
vivi sono almeno il minimo, la prima manopola non ancora al massimo sale di
0,05; se in ognuno degli ultimi 3 minuti sono serviti colonizzatori, l'ultima
manopola alzata scende di 0,05. Al più un cambio ogni 3 minuti, tutto nel
registro (evento "mondo"). Le manopole sono stato del mondo: vengono dal
salvataggio. `arena_world_ladder` = False le lascia ferme.

### Misure in più

Morti predati, frutti cattivi mangiati, energia data ai figli, taglia media e
dispersione, livelli medi di morso, luce e odore, salute media delle macchie,
manopole. Dopo ogni esame fisso, la **prova dell'apprendimento**: gli stessi
20 senza plasticità; la differenza è quanto imparano durante la vita.

### Motore

Solo sul motore JAX ("gpu", che gira anche senza scheda video): il motore
NumPy resta il riferimento del mondo base e lì le manopole restano a 0. Con
750 esseri e il mondo complesso circa 10 volte il tempo reale: la GPU calcola
solo i posti fino all'ultimo vivo (arrotondati a 256), e le nascite occupano i
posti liberi più bassi.

## 16. Da verificare

- Che la popolazione si stabilizzi (né estinzione né tetto pieno) con i
  numeri della sezione 6; altrimenti si ritocca la ricrescita del cibo.
- Che la vita mediana effettiva sia vicina a 600 s e che si muoia sia di fame
  sia di vecchiaia.
- Quanto costa un secondo di mondo con 1000 e con 3000 esseri.
- Che gli esseri si incontrino davvero (vicini nel raggio di vista).
