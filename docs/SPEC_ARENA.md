# Specifica: l'arena continua

Stato: **prima versione implementata** (2026-10-23): `sim/arena.py` (il
mondo), `evo/ecology.py` (nascite, morti, misure, salvataggi),
`viz/arena_view.py` (la schermata), `uv run main.py --task arena`. Gira sul
processore (NumPy e numba); il motore GPU, costruito attorno alle
generazioni, non la fa ancora (sezione 11). Le modalità a generazioni non
cambiano: stesso seme, stessi numeri di prima.

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

- **Quando**: energia ≥ **300** (il doppio dell'energia iniziale), età
  ≥ **30 s** (maturità), e almeno **30 s** dall'ultimo figlio.
- **Costo**: il figlio riceve **150** (l'energia iniziale di oggi); il
  genitore ne paga **200**: in natura fare un figlio costa più dell'energia
  che il figlio riceve. Dopo il parto il genitore resta con circa 100: deve
  tornare a mangiare.
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
  (σ) **120 u**. Il cibo ricresce in un punto a caso con probabilità
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

## 13. Da verificare

- Che la popolazione si stabilizzi (né estinzione né tetto pieno) con i
  numeri della sezione 6; altrimenti si ritocca la ricrescita del cibo.
- Che la vita mediana effettiva sia vicina a 600 s e che si muoia sia di fame
  sia di vecchiaia.
- Quanto costa un secondo di mondo con 1000 e con 3000 esseri.
- Che gli esseri si incontrino davvero (vicini nel raggio di vista).
