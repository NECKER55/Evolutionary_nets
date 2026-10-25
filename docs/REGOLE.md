# Le regole della simulazione

Come funzionano il mondo, gli agenti, la riproduzione e le mutazioni. L'interfaccia è descritta in
[INTERFACCIA.md](INTERFACCIA.md), il mondo del cibo in [SPEC_MONDO.md](SPEC_MONDO.md).

Tutti i numeri qui sotto sono i valori di default. Tra parentesi c'è il nome
della costante in `evonets/config.py` che li controlla.

## 0. Due tipi di simulazione

Si sceglie con `sim_mode` (riga "Simulazione" del menu, `--mode` nel terminale);
è parte dello stato della corsa e si conserva riprendendola.

- **evoluzione** (predefinita): evolvono sensi e cervello, come descritto in
  tutto il resto di questo documento.
- **cervello**: i **sensi sono fissi e decisi da noi**, uguali per tutti
  (`net/senses.py`, set `sensor_set`): non mutano, non si duplicano, non si
  rompono, e il nervo entra nel cervello sempre nello stesso punto. **Evolve
  solo il cervello**, con tutte le altre regole (neuroni, collegamenti, pesi,
  plasticità, duplicazioni, incrocio, sesso, fecondità, riserve). Lo scopo è
  adattare un cervello ai sensi dati per sopravvivere nel mondo scelto. Il
  cervello parte primordiale; i sensi costano energia come sempre (circa 0,64
  energia/s in tutto).

Il set "naturale" (12 sensi, ognuno specializzato su un solo fenomeno; le
corse "solo cervello" salvate prima del 2026-10-21 restano con i loro 10):

| Senso | Sente | Sensibilità | Dove sta / dove guarda | Apertura | Adattamento (τ, sottr., div.) |
|---|---|---|---|---|---|
| naso sinistro / destro | odore del cibo (0,6; larghezza 0,1) | 15 | muso, ±30° | — | 0,3 s; 0,8; 0,75 (come il naso evoluto) |
| occhio rosso sinistro / destro | colore del cibo (0,85) | 10 | ±15°, guarda a ±25° | 60° | 1 s; 0,2; 0,5 |
| occhio blu sinistro / destro | colore dei predatori (0,05) | 6 | fianchi, ±90° | 120° | 1 s; 0,2; 0,5 |
| vibrazioni | vibrazioni dei predatori (0,4) | 6 | centro | 360° | 1 s; 0,5; 0 |
| contatto | pressione delle pareti | 2 | muso | 180° | nessuno |
| energia | energia del corpo (rispetto alle riserve massime) | 4 | interno | — | nessuno |
| velocità | velocità in avanti | 2 | interno | — | nessuno |
| gusto | il cibo appena mangiato | 4 | interno | — | nessuno |
| dolore | veleno e morsi | 4 | interno | — | nessuno |

Nella simulazione "solo cervello" anche il **sistema di valore** è dato
(forza di partenza, mai mutata): vedi "Imparare durante la vita".

## 1. Il tempo

- La simulazione avanza a passi di 1/60 di secondo (`dt`).
- A ogni passo, nell'ordine: ogni agente legge i propri sensi, la sua rete
  calcola i comandi, la fisica muove tutti, si aggiorna l'energia, si controlla
  chi è morto e chi ha completato il compito.

## 2. L'arena e gli agenti

- **Arena**: rettangolo di 200 × 140 unità (`arena_w`, `arena_h`) chiuso da pareti.
- **Agente**: un cerchio di raggio 1 (`agent_radius`) con una direzione (il muso).
- **Nascita**: posizione casuale, orientamento casuale, fermo, ad almeno 4 unità
  dal bordo di ogni zona di veleno (`spawn_margin`).
- **Gli agenti non si toccano tra loro**: si attraversano. Ognuno affronta la
  stessa arena come se fosse solo.

## 3. Il movimento

La rete produce due comandi, entrambi tra -1 e 1.

| Comando | Effetto |
|---|---|
| **Spinta** | accelera lungo il muso: fino a 60 u/s² in avanti (`thrust_accel`), la metà in retromarcia (`reverse_factor`) |
| **Sterzata** | accelerazione di rotazione fino a 40 rad/s² (`turn_accel`); positivo = verso destra |

- **Attrito in avanti** (`lin_drag` = 2): limita la velocità a 30 u/s in avanti e
  15 u/s in retromarcia. L'arena si attraversa in circa 7 secondi.
- **Attrito di traverso** (`lat_drag` = 8): quattro volte più forte. L'agente
  tende ad andare dove punta il muso, ma in curva derapa un po'.
- **Attrito di rotazione** (`ang_drag` = 6): limita la rotazione a circa un giro
  al secondo. La rotazione ha inerzia: non si ferma all'istante.
- **Pareti**: l'agente rimbalza perdendo il 70% della velocità
  (`wall_restitution` = 0,3). Urtare una parete non fa danno.

## 4. L'energia e la morte

- Ogni agente nasce con **150 di energia** (`energy_init`). Nel mondo del bersaglio
  **non si ricarica mai**; nel mondo del cibo ogni cibo mangiato ne restituisce 100
  (`food_energy`). Si possono accumulare **riserve** fino a 450
  (`energy_max`), come il grasso: l'energia oltre quella di partenza pesa, e
  il movimento costa di più (con le riserve piene il doppio, `reserve_mass`).
  Così chi sa cercare supera i periodi sfortunati; chi cerca male muore di
  fame. Misurato: senza riserve la bravura spiegava il 17% del successo in una
  vita, con le riserve il 36%.
- **Si muore in un solo modo: quando l'energia arriva a zero.** Non esiste morte
  istantanea.
- L'energia si consuma per quattro motivi, che si sommano:

| Causa | Consumo al secondo |
|---|---|
| Esistere | 2,5 (`metabolism_basal`) |
| Spingere | fino a 10 in più, proporzionale al **quadrato** del comando di spinta, uguale in avanti e indietro (`metabolism_move`, `metabolism_exponent`) |
| Sterzare | fino a 2 in più, proporzionale al quadrato del comando di sterzata (`metabolism_turn`) |
| Veleno | da 0 a 60, secondo la profondità nella zona (vedi sotto) |
| Cablaggio del cervello | 0,004 per ogni unità di lunghezza dei collegamenti effettivi (`wire_cost`) |
| Neuroni interni | 0,01 per ogni neurone acceso (`neuron_cost`): un neurone consuma anche a riposo |
| Recettori | 0,002 per ogni recettore acceso (`receptor_cell_cost`), più 0,00005 per ogni unità di sensibilità (`receptor_sens_cost`): un recettore preciso costa |

- Si paga lo **sforzo comandato**, non la velocità raggiunta: spingere contro una
  parete costa come correre.
- Conseguenze: un agente fermo vive circa 60 secondi (un po' meno, per il
  costo di corpo e cervello); uno sempre a spinta piena circa 12 secondi (circa
  360 unità percorse); uno a metà spinta circa 30 secondi.
- **Perché il movimento costa così.** A spinta piena il consumo totale è 5
  volte quello a riposo: è lo sforzo massimo sostenibile tipico di un pesce
  (l'"aerobic scope", da 2–3 a circa 10, con valori tipici 4–5). Il costo cresce
  con il quadrato del comando perché, nella fisica dell'agente, la velocità è
  proporzionale alla spinta e la potenza è forza per velocità. Andare piano
  costa poco, scattare molto, e il costo per unità di strada è minimo a velocità
  moderata, come per i pesci veri. Vagare alla cieca a tutta velocità non è più
  quasi gratis.
- Perché 150: con 100 anche una rete buona fallisce spesso per sfortuna (non
  trova il bersaglio in tempo) e viene eliminata, quindi la selezione è
  rumorosa. In un confronto su 4 semi e 120 generazioni, le reti evolute con 150
  completano il 51% delle prove anche quando le si mette alla prova con 100,
  contro il 39% di quelle evolute con 100 (dettagli in [NOTE.md](NOTE.md)).
- Cablaggio, neuroni e recettori si pagano anche da fermi, come il cervello
  vero che consuma energia sempre: un cervello grande o tanti sensi precisi
  accorciano la vita. Al posto di un numero massimo c'è un prezzo, e la
  selezione tiene un neurone, un recettore o un collegamento lungo solo se
  vale quello che costa. I costi sono ancora da calibrare con le misure (vedi
  [SPEC_FASE_C.md](SPEC_FASE_C.md), sezione 10).
- **Un agente morto** resta fermo dov'è, diventa grigio e non può più completare
  il compito.

## 5. Il veleno

- **12 zone circolari** (`n_hazards`), raggio tra 5 e 12 unità
  (`hazard_radius_min`, `hazard_radius_max`).
- **Si muovono**: ognuna oscilla attorno a un proprio punto fisso, con
  un'oscillazione indipendente in orizzontale e in verticale (un giro completo
  ogni 8–31 secondi). Non escono mai dall'arena.
- **Pulsano**: il raggio cresce e cala del 15% (`hazard_pulse`).
- **Il danno è graduale**: zero sul bordo, massimo al centro. Vale
  60 × profondità² al secondo (`hazard_drain`, `hazard_falloff`), dove la
  profondità è 0 sul bordo e 1 al centro.

| Dove si trova l'agente | Energia persa al secondo |
|---|---|
| sul bordo | 0 |
| a un quarto del raggio dal bordo | circa 4 |
| a metà raggio | 15 |
| al centro | 60 (morte in meno di 2 secondi) |

- Conta la posizione del **centro** dell'agente.
- Se due zone si sovrappongono, i danni si sommano.
- `hazard_speed` moltiplica la velocità di tutte le zone: è la leva prevista per
  rendere il compito più arduo.

## 6. Il compito

Ci sono due mondi, scelti con `task` (riga "Mondo" del menu, `--task` nel
terminale). Il predefinito è ancora il bersaglio.

### Mondo del bersaglio (`bersaglio`)

- Il bersaglio è un cerchio di raggio 3 (`target_radius`) in una posizione
  casuale. Può capitare anche dentro o vicino a una zona di veleno.
- È uno solo per tutti e non si consuma.
- **Completa il compito** l'agente vivo che lo tocca, cioè il cui centro arriva a
  meno di 4 unità dal centro del bersaglio.
- Chi completa è **in salvo**: si ferma, diventa bianco, non consuma più energia
  e il veleno non gli fa più nulla.
- Non conta in quanto tempo, con quanta energia residua o per quale strada:
  **completato è completato**.

### Mondo del cibo (`cibo`)

Il progetto completo è in [SPEC_MONDO.md](SPEC_MONDO.md).

- **Completa il compito** chi mangia almeno **N cibi** nella sua vita (N da 1 a
  10, vita fino a 240 s). Una volta si può essere fortunati, dieci volte di
  fila no.
- Il cibo è un disco rosso (0,85 sull'asse della luce). Toccato, viene mangiato,
  dà 100 di energia e ricompare altrove, lontano dall'agente.
- **Ognuno ha la sua arena**: cibo, prede, predatori e punti in cui il cibo
  ricompare sono diversi per ogni agente (`forage_own_arena`). Così una
  generazione non dipende dalla fortuna di un'arena sola. Non c'è competizione
  per il cibo.
- Il veleno è spento. Si nasce ad almeno 20 u dal cibo e 40 dai predatori.
- **Le manopole della difficoltà**, da 0 (basico) a 1 (complesso):

| Manopola | 0 | 1 |
|---|---|---|
| abbondanza (`forage_abundance`) | 12 cibi di raggio 4 | 4 cibi di raggio 2 |
| cibi necessari (`forage_need`) | 1 | 10 |
| prede (`forage_prey`) | il cibo sta fermo | il cibo fugge a 12 u/s dall'agente entro 15 u |
| nube d'odore (`forage_scent`) | sempre formata | si forma in 3 s quando il cibo compare; quella di un cibo mangiato resta dov'era e svanisce in 3 s |
| predatori (`forage_predators`) | nessuno | 3 predatori (quasi blu, vibrano) che inseguono entro 40 u a 15 u/s e a contatto tolgono 60 di energia al secondo |
| rocce (`forage_obstacles`) | nessuna | 8 rocce grigie di raggio 6–16, ferme, che non si toccano (tra due c'è sempre un passaggio di almeno 10 u) |
| frutti (`forage_variety`) | un frutto solo, rosso e buono | metà dei frutti cattivi: in ogni vita un colore (rosso o giallo) è buono e l'altro cattivo, a caso |

- **I frutti** (quinto stadio): un mondo in cui imparare conviene, perché
  cambia da una vita all'altra ma resta stabile dentro una vita. Ci sono due
  frutti: rosso (0,85 sulla luce, molecola 0,6) e giallo (0,65, molecola
  0,5). In ogni vita, per ogni agente, uno dei due colori è buono e l'altro
  cattivo, scelto a caso; una quota dei posti (fino a metà, con la manopola)
  ha il frutto cattivo, e ogni posto tiene il suo tipo quando il frutto
  ricompare. Il frutto buono nutre (100), dà gusto e conta per riprodursi;
  quello cattivo non nutre, dà dolore e toglie 30 di energia
  (`bad_fruit_damage`): uno sbaglio insegna, non uccide. Con gusto, dolore e
  il sistema di valore il cervello può imparare in pochi bocconi quale
  colore mangiare in questa vita; nessun riflesso fisso può saperlo prima.
  Il set di sensi "naturale" ha due occhi gialli (14 sensi).

- **Le rocce** (quarto stadio): sono opache e solide. Fermano l'agente come
  una parete (rimbalzo anelastico, e il lato che le tocca sente la
  pressione), le prede e i predatori: un predatore che insegue dritto si
  blocca contro una roccia, che diventa un **rifugio**. Nascondono alla vista
  ciò che sta dietro (anche il cibo); si vedono grigie e spente (0,45
  sull'asse della luce, albedo 0,4). Odori e vibrazioni le scavalcano: dietro
  una roccia il cibo si sente ma non si vede. Il cibo non nasce e non ricompare
  mai dentro una roccia, e nessuno nasce dentro una roccia. Ogni agente ha le
  sue rocce, come il suo cibo.

- **Gli odori**: il cibo rilascia la sua molecola (0,6 sull'asse delle
  molecole), i predatori un'altra (0,35). Diffusione e degradazione fanno
  attorno a ognuno una nube stabile, `e^(−d/25)` con d dal bordo
  (`odor_range`); le nubi si sommano. Una cellula normale sintonizzata sul
  cibo raccoglie circa 130 quanti a 10 u, 75 a 25 u, 23 a 54 u: da lontano
  il segnale si distingue solo muovendosi (confronto nel tempo), da vicino
  anche tra un lato e l'altro del corpo.
- **La scala** (`ladder`): con `scala` le manopole salgono una alla volta
  (abbondanza e cibi necessari insieme, poi prede e nube d'odore insieme,
  poi predatori, poi rocce, poi frutti) se negli
  ultime 10 generazioni completa più del 50%, scendono sotto il 30%, e
  scendono di più se in una generazione non completa nessuno. Con `basico` o
  `complesso` restano ferme a 0 o a 1: sono le popolazioni di controllo. Ogni
  movimento finisce nel registro dell'ambiente.

## 7. I sensi: i recettori universali

L'agente non conosce la propria posizione, né dov'è il bersaglio, né il proprio
orientamento nell'arena. Sa solo ciò che i suoi recettori captano, e i
recettori sono a loro volta il risultato dell'evoluzione. Il modello completo,
con le formule, è in [SPEC_FASE_C.md](SPEC_FASE_C.md).

**La fisica: sei fenomeni.** Lo decide il mondo, non l'agente. Ognuno ha un
asse della "qualità" da 0 a 1:

| Fenomeno | Come arriva | Qualità | Nel mondo, di default |
|---|---|---|---|
| luce | in linea retta; il mezzo è torbido e la luce si attenua come e^(−d/12) | colore | pareti grigie (0,5), bersaglio rosso (0,85), veleno verde (0,25) che brilla come una nube e non nasconde ciò che sta dietro |
| sostanze | concentrazione nel punto del corpo | molecola | la tossina del veleno (0,15): dentro la zona è uguale al danno; fuori il veleno è inodore |
| meccanica | pressione da contatto, vibrazioni a distanza | frequenza | la pressione di una parete toccata (frequenza 0); nessuna sorgente di vibrazioni |
| temperatura | campo nel punto del corpo | la temperatura | uniforme (0,5), nessuna sorgente di calore |
| magnetismo | campo uniforme, come quello di un pianeta | — | nessun campo |
| interno | grandezze del corpo | quale grandezza | energia, gusto, dolore, velocità avanti e indietro, derapata e rotazione a destra e a sinistra |

Cosa emette ogni oggetto è una costante della configurazione: dare un suono al
bersaglio (`target_vibration`), un odore al veleno (`hazard_smell`) o un campo
magnetico al mondo (`magnetic_field`) è cambiare un numero, senza toccare il
codice dei sensi.

**Il recettore.** Un solo tipo di cellula, i cui geni decidono tutto:

- per ciascuno dei 6 fenomeni una **sensibilità** g e una **sintonia** (una
  campana sull'asse della qualità: centro e larghezza);
- dove sta sul **corpo**, dove **guarda** e quanto è larga la sua **apertura**
  (da 1° a 360°);
- dove entra il suo nervo nel **cervello** (sul bordo sinistro).

**La risposta.** Il recettore conta "quanti" (fotoni, molecole) e il conteggio
è rumoroso, come nella retina: n = Σ scala × g × segnale, e la risposta è
(n + rumore) / 300, tra 0 e 1, con un rumore che vale √(n + 1).

- **scala** è la fisica del fenomeno: quanti quanti raccoglie in un passo una
  cellula normale da un segnale 1. È tarata perché uno stimolo tipico dia circa
  100 quanti, come un fotorecettore alla luce del giorno: luce 300, sostanze
  200, meccanica, temperatura e magnetismo 100, interno 70
  (`receptor_scale_*`). Non va ritoccata per aggiustare un esperimento: è come
  la fisica dei fotoni. Se cambi l'ambiente, esprimi le emissioni nella stessa
  unità (1 = uno stimolo normale per quel fenomeno).
- **g** è il gene della sensibilità: un moltiplicatore, 1 = cellula normale.
- Misure con una cellula normale specializzata sul colore del bersaglio
  (apertura 10°): bersaglio a 2 u circa 220 quanti, a 10 u circa 110 (rapporto
  segnale/rumore 10), a 20 u circa 50, a 40 u circa 8 (si vede appena).

Di conseguenza:

- più sensibilità vuol dire meno rumore relativo, ma anche saturazione prima e
  più costo: il compromesso lo trova l'evoluzione;
- un recettore sensibile a più fenomeni li somma in un numero solo, ambiguo:
  conviene specializzarsi;
- la sensibilità non è un doppione dei pesi sinaptici: un peso grande
  amplifica anche il rumore, solo la sensibilità lo riduce.

**L'adattamento sensoriale.** Come tutti i recettori veri, ogni recettore
tiene una media recente m dei quanti che riceve (con un tempo caratteristico
τ) e la usa per regolarsi:

    risposta = (quanti − sott × m) / (100 + div × m)

- **sottrattivo** (`sott` da 0 a 1, come nei batteri): a segnale costante la
  risposta torna verso zero; risponde ai **cambiamenti**. Muovendosi, un
  recettore così sente da che parte un segnale cresce;
- **divisivo** (`div` da 0 in su, come nella retina): divide per il livello
  medio; risponde al **contrasto**, cioè a quanto il segnale è più forte o più
  debole del solito, e non satura anche con segnali forti.

τ, sott e div sono geni di ogni recettore. Alla partenza sono deboli e diversi
da individuo a individuo (τ intorno a 1 s, sott e div tra 0 e 0,2): un
adattamento rudimentale, che l'evoluzione può rafforzare o spegnere. La media
riparte dalla prima lettura a ogni arena.

**La partenza è primordiale.** Ogni individuo nasce con 2 recettori
**generalisti** (sensibili un po' a tutto: moltiplicatore 1 su ogni fenomeno,
sintonia larga), a ventaglio davanti al muso con un'apertura di 120°. Sono come
le prime macchie fotosensibili: un bersaglio a 10 u si distingue (rapporto
segnale/rumore circa 4 per la sola luce), ma non si sa bene dove sia, e il
segnale si mescola con quello degli altri fenomeni.
Occhi, nasi o orecchie non esistono nel codice: possono nascere solo per
duplicazione e specializzazione dei recettori (punto 15).

- In più si può accendere un disturbo uguale per tutti gli ingressi
  (`sensor_noise`, 0 di default); lo usa la prova "rumore" dell'esame fisso.

## 8. La rete: struttura comune ai due tipi

**I neuroni.** Tre gruppi:

| Gruppo | Quanti | Esistono sempre? |
|---|---|---|
| recettori | 2 alla partenza, poi quanti ne nascono per duplicazione | no: ogni posto ha un gene che dice se esiste |
| interni | 0 alla partenza, poi quanti ne nascono | no: come sopra |
| motori | 2 (reti continue) o 4 (reti a impulsi) | sì |

**Nessun numero massimo.** Il genoma ha dei **posti** per recettori e neuroni
interni (all'inizio 4 e 8, `receptor_slots`, `net_hidden_slots`). Quando
qualcuno ne usa più dell'80% (`pool_fill`), tutta la popolazione riceve 8
posti nuovi spenti (`pool_step`). È solo memoria: a frenare la crescita è il
costo in energia di ogni neurone e recettore, non un tetto.

- Nelle generazioni successive il numero cambia per eredità e mutazione
  (accensione, spegnimento, duplicazione). Non c'è un minimo né un obiettivo:
  lo decide la selezione.
- Un neurone o recettore spento resta a zero e i suoi collegamenti sono
  ignorati. I suoi geni però restano scritti nel genoma (come uno pseudogene):
  se una mutazione lo accende, nasce già con i suoi geni.
- Il calcolo usa matrici piene: il tempo per passo cresce con il cervello più
  grande della popolazione.

**Il cervello ha una geometria.** È un foglio quadrato di lato 1, come un
pezzo di corteccia:

- i **recettori** entrano dal bordo sinistro, all'altezza data da un loro gene:
  alla nascita segue la direzione in cui guardano (una mappa topografica, come
  la retina), poi deriva con le mutazioni;
- i **motori** stanno in posizioni fisse sul bordo destro. Con i quattro motori
  della rete a impulsi il cervello è **bilaterale** come il corpo: lo sterzo
  verso sinistra nella metà sinistra (y = 0,125, dove entrano i nervi dei
  recettori che guardano a sinistra), quello verso destra nella metà destra
  (y = 0,875), avanti e indietro vicino alla linea mediana (0,45 e 0,55);
- ogni **neurone interno** sta dove dice il suo gene di posizione. Alla nascita
  di una popolazione casuale le posizioni sono sparse a caso; poi si ereditano
  e si spostano di poco con le mutazioni.

**I collegamenti.** Non ci sono strati.

- Un collegamento va da una **sorgente** (un recettore, un neurone interno o un
  motore) a un **destinatario** (un neurone interno o un motore). I recettori
  non ricevono collegamenti.
- Sono ammessi tutti i casi: recettore → motore diretto, interno → interno,
  anelli, un neurone verso se stesso, un motore che rimanda il proprio valore a
  un interno.
- **Come nel cervello vero, due neuroni si collegano tanto più facilmente
  quanto più sono vicini.** Un collegamento compare con probabilità
  0,75 × e^(−d / λ), dove d è la distanza tra i due neuroni e λ, la portata
  tipica dei collegamenti, è un **gene** di ogni individuo (parte da 0,3,
  `net_conn_range`). È la regola esponenziale della distanza
  osservata nella corteccia: i collegamenti locali sono la maggioranza, quelli
  lunghi rari ma possibili.

| Distanza | Probabilità con λ = 0,3 |
|---|---|
| 0,1 (vicinissimi) | 54% |
| 0,3 | 28% |
| 0,5 | 14% |
| 1 (da un bordo all'altro) | 3% |

- Un neurone collegato a se stesso fa eccezione: in natura è raro, e ha
  probabilità 5% (`net_self_prob`).
- Un collegamento diretto recettore → motore attraversa tutto il cervello ed è
  raro (3%): alla partenza primordiale quasi nessun individuo ne ha, quindi
  all'inizio gli agenti sono di fatto ciechi e si muovono secondo le soglie dei
  motori. Gli strati non sono imposti, ma la geometria li favorisce.
- Contano solo i collegamenti tra neuroni esistenti. Un recettore può restare
  senza alcun collegamento: per quell'individuo è come non averlo, ma lo paga.
- Ogni collegamento ha un **peso**, sorteggiato attorno a zero (per lo più tra
  -2 e 2, `net_weight_std`).

**Il tempo nella rete.**

- Tutti i neuroni si aggiornano nello stesso istante, usando i valori del passo
  precedente. Il segnale fa un salto per passo: da recettore a motore diretto arriva
  subito, passando per un neurone interno arriva un passo dopo.
- Gli anelli trattengono informazione da un passo all'altro: sono la **memoria**
  dell'agente. Non esiste altra memoria.
- Alla nascita tutti i neuroni partono da zero.
- **I pesi cambiano durante la vita** (hebbiano di base, omeostasi e, se ci
  sono neuroni modulatori, plasticità modulata): vedi "Imparare durante la
  vita" qui sotto. Ciò che si impara non si eredita.

**Imparare durante la vita** (dettagli in [SPEC_FASE_E.md](SPEC_FASE_E.md)).

- **La plasticità cambia la forza, mai il tipo**: ogni sinapsi nasce
  eccitatoria o inibitoria (per la legge di Dale nei neuroni a impulsi, dal
  segno del gene per i sensi e nelle reti continue) e lo resta tutta la vita;
  la sua forza |w| va da 0 a `synapse_max`. Le attività contano come
  frequenze (|attività|).
- **Hebbiano di base, sempre attivo** (η_h gene del neurone che riceve,
  velocità di partenza 1e-4, `hebb_initial_rate`: neutrale; l'evoluzione può
  alzarla o spegnerla). Non serve un modulatore. Scrive una **traccia
  provvisoria** (come l'early-LTP), che svanisce da sola con il tempo τ del
  neurone (gene "persist", partenza 30 s, `early_initial_tau`; in natura 1–3
  ore). Diventa stabile solo se, mentre c'è, arriva un modulatore
  (*tagging and capture*, Frey e Morris 1997): una quota
  1 − exp(−`capture_rate` · |M| · dt) a passo, con M buono o cattivo (si
  ricorda sia ciò che premia sia ciò che spaventa). Così l'attività di tutti i
  giorni non sovrascrive i circuiti: resta solo ciò che accade insieme a un
  segnale importante.
  - Sinapsi eccitatorie, regola BCM: Δ|w| = η_h · pre · post · (post − θ),
    con la soglia mobile θ = ā² / a*. Sopra la soglia la sinapsi si rafforza,
    sotto si indebolisce; se il neurone è troppo attivo la soglia sale. A
    regime l'attività del neurone va verso a*.
  - Sinapsi inibitorie, regola di Vogels: Δ|w| = η_h · pre · (post − a*).
    L'inibizione cresce dove il neurone è più attivo del desiderato e cala
    dove lo è meno: tiene in equilibrio eccitazione e inibizione.
- **Frequenze**: le regole di Hebb e l'omeostasi lavorano sulle frequenze,
  sempre ≥ 0. Nella rete a impulsi l'attività è già una frequenza; nella rete
  continua x = tanh(...) è lo scarto dalla frequenza spontanea, e la frequenza
  è (1 + x)/2 (x = 0: il neurone spara al suo ritmo di riposo).
- **Scaling omeostatico, sempre attivo**: ogni neurone interno e ogni motore
  tiene la media ā della propria frequenza su un tempo lungo (120 s,
  `homeo_tau`: molto più di qualsiasi comportamento, come in natura dove
  sono ore) e moltiplica le sinapsi eccitatorie che riceve per
  exp(h · dt · (a* − ā)) e quelle inibitorie per l'inverso (come ha misurato
  Turrigiano). Geni del neurone: l'attività desiderata a* = sigmoide(gene) e
  la forza h = `homeo_initial_rate` · e^gene (partenza 1e-4 per secondo:
  neutrale). Chi è troppo attivo si indebolisce gli ingressi, chi è troppo
  zitto se li rafforza. Si nasce in equilibrio (ā = a*).
- **Omeostasi dell'eccitabilità** (Desai 1999), un meccanismo distinto con
  un gene proprio ("intrinsic", stessa partenza): la corrente di fondo si
  sposta di h_i · dt · (a* − ā), cioè cresce se il neurone è troppo zitto.
- **Sistema di valore innato**: come in tutti gli animali, si nasce già
  sapendo cosa è buono e cosa fa male. Due nuclei, come l'area tegmentale
  ventrale (dopamina) e il locus coeruleus (noradrenalina), ricevono il
  **gusto** (un impulso a ogni cibo mangiato, che svanisce in 0,5 s,
  `taste_tau`) e il **dolore** (veleno e morsi) e proiettano su tutto il
  cervello: a M di ogni neurone si aggiunge forza_gusto · gusto −
  forza_dolore · dolore. Così ciò che il cervello ha appena fatto (la
  traccia provvisoria) diventa stabile quando si mangia o ci si fa male, e
  la plasticità modulata prende il verso giusto. Le due forze sono geni
  dell'individuo (`Genome.value`, partenza `value_initial` = 1, in scala
  logaritmica): passano interi da un genitore e mutano come gli altri geni,
  quindi l'evoluzione può rafforzarli o spegnerli; nella simulazione "solo
  cervello" sono fissi. Gusto e dolore sono anche grandezze del corpo
  (fenomeno "interno"): la rete può evolvere recettori che li sentono. Le
  popolazioni salvate prima (formato < 16) si aprono con i nuclei spenti.
- **Periodo critico**: come in natura, l'hebbiano è fortissimo nello
  sviluppo e si chiude con l'età. Per ogni neurone
  η_h(età) = η_h · (1 + (B − 1) · e^(−età/τ_c)), dove η_h è la velocità
  dell'adulto, B quante volte è più forte all'inizio (partenza 100,
  `critical_initial_boost`) e τ_c in quanti secondi si chiude (partenza 2 s,
  `critical_initial_tau`): B e τ_c sono geni del neurone ("boost",
  "closure"), così neuroni diversi possono avere finestre diverse, come le
  aree della corteccia. L'età si conta in secondi dall'inizio del grembo
  (senza grembo, dalla nascita); un sopravvissuto ha già vissuto una vita
  intera per ogni generazione superata, quindi non torna bambino in
  un'arena nuova. Con i valori di partenza il periodo critico cade per lo
  più nel grembo: lì il cablaggio si rifinisce (circa il 10% dei pesi),
  alla nascita l'hebbiano è circa 9 volte quello dell'adulto e dopo 15 s è
  quello dell'adulto. La plasticità modulata (la dopamina) resta aperta per
  tutta la vita. Le popolazioni salvate prima (formato < 15) si aprono con
  il periodo critico spento (B = 1): sono evolute senza.
- **Saturazione graduale** (van Rossum 2000): ogni rafforzamento è
  moltiplicato per 1 − |w| / `synapse_max`: una sinapsi grande cresce sempre
  meno, invece di sbattere contro un tetto.
- Come le altre plasticità, ciò che cambia durante la vita non si eredita.
- **Salvataggi di prima del formato 10** (senza a*): aprendoli, una vita di
  prova di 10 s misura la frequenza media di ogni neurone, che diventa il suo
  a* (la popolazione era in equilibrio); costa circa mezzo minuto, una volta.

- Ogni neurone ha, tra i suoi geni, una **regola di plasticità** per le
  sinapsi che riceve: Δw = η · M · (A·pre·post + B·pre + C·post + D). η, A, B,
  C e D sono geni: l'evoluzione sceglie *come* imparare, non *cosa*.
- Un gene può rendere un neurone interno **modulatore** (come i neuroni a
  dopamina): non trasmette, ma rilascia M, che si diffonde nel cervello e
  arriva ai neuroni vicini calando con la distanza. M dice alle sinapsi quando
  cambiare e in che verso. Senza modulatori non si impara nulla.
- Una sinapsi non supera ±4 e, nelle reti a impulsi, non cambia segno.
- A ogni nascita e a ogni arena si riparte dai pesi del genoma.
- Alla partenza circa il 10% dei neuroni è modulatore e la plasticità è debole.
  Nel mondo di oggi non c'è niente da imparare (tutto ciò che serve si vede, e
  l'energia non sale mai): l'ambiente che renderà utile imparare è ancora da
  decidere. Nel viewer i modulatori hanno l'anello viola; toccandone uno, linee
  viola mostrano i neuroni che raggiunge.

## 9. Rete continua (`--net rate`, predefinita)

- Ogni neurone ha un valore tra -1 e 1.
- A ogni passo: somma i valori delle sue sorgenti, ciascuno moltiplicato per il
  peso del collegamento, aggiunge la propria **soglia** (un numero suo, anche
  negativo) e schiaccia il risultato tra -1 e 1 (funzione tanh).
- **Segno libero**: ogni collegamento ha il suo. Peso positivo = eccita,
  negativo = inibisce. Lo stesso neurone può eccitare un vicino e inibirne un
  altro. Non esistono neuroni "eccitatori" o "inibitori".
- **2 motori**: il valore del primo è la spinta, quello del secondo la sterzata.
- Geni propri di ogni neurone: 1, la soglia.

## 10. Rete a impulsi con neuroni eccitatori o inibitori (`--net spike-ei`)

Stessa struttura, ma i neuroni interni comunicano a impulsi e hanno un tipo.

**Neuroni interni.**

- Ognuno ha una **carica**. A ogni passo di rete la carica perde una parte di sé
  e ne guadagna in proporzione alla corrente in arrivo (la somma pesata delle
  sorgenti più la corrente di fondo del neurone).
- Quando la carica raggiunge **1**, il neurone emette un **impulso** e torna a 0.
- Agli altri neuroni arriva solo l'impulso: 1 se ha scattato, 0 altrimenti.
- La carica non scende sotto -1.
- La rete fa **4 passi per ogni passo di fisica** (`spike_substeps`), cioè 240 al
  secondo: un neurone può scattare al massimo 240 volte al secondo.
- La corrente verso i neuroni interni è moltiplicata per 3 (`spike_gain`).

**Eccitatori e inibitori.**

- Ogni neurone interno è **tutto eccitatorio o tutto inibitorio**: il tipo vale
  per tutti i suoi collegamenti in uscita.
- Il peso di un collegamento che parte da un neurone indica solo la **forza**
  (si usa il valore senza segno). Il segno lo dà il tipo di chi trasmette: tra
  neuroni non esistono pesi negativi.
- Il tipo è un gene del neurone. Alla nascita di una popolazione casuale il
  **20%** dei neuroni è inibitorio (`spike_inhibitory_share`).
- Non c'è una quota da rispettare dopo la nascita: la proporzione è libera di
  cambiare con l'evoluzione.
- Il tipo si eredita col neurone e può invertirsi per mutazione, ma di rado: il
  gene deve derivare fino a cambiare segno.

**Eccezioni alla regola del segno.**

| Chi trasmette | Può eccitare | Può inibire |
|---|---|---|
| neurone interno eccitatorio | sì | no |
| neurone interno inibitorio | no | sì |
| motore | sì | no |
| recettore | sì | sì (il segno è del collegamento) |

- I recettori sono liberi per scelta. Con `sensors_excitatory_only` possono solo
  eccitare, come gli altri.
- La **corrente di fondo** di un neurone non è un collegamento e può essere
  positiva (neurone che tende a scattare da solo) o negativa (neurone difficile
  da accendere).

**Recettori e motori non emettono impulsi.**

- I recettori entrano come corrente continua: un valore più alto carica più in
  fretta i neuroni a cui sono collegati.
- I **4 motori** lavorano a coppie opposte, come i muscoli: avanti/indietro e
  destra/sinistra. Ciascuno accumula nel tempo ciò che riceve, senza soglia, e
  il suo livello va da 0 a 1: non può andare sotto zero, quindi inibire un
  motore lo spegne ma non lo fa "tirare al contrario".
- Spinta = avanti − indietro. Sterzata = destra − sinistra.

**Geni propri di ogni neurone: 3.**

| Gene | Effetto |
|---|---|
| corrente di fondo | quanto il neurone è attivo di suo |
| perdita | quanta carica trattiene da un passo all'altro: dal 50% (neurone rapido, dimentica subito) al 97% (lento, somma a lungo), tipicamente l'82% |
| tipo | eccitatorio o inibitorio (usato solo dai neuroni interni) |

## 11. Il genoma

Il genoma di un individuo è tutto ciò che serve a costruire il suo corpo
sensoriale e la sua rete:

- per ciascun posto di **recettore**: se esiste e tutti i suoi geni (punto 7);
- per ciascun posto di **neurone interno**: se esiste e dove sta nel cervello;
- per ciascun collegamento possibile, se **esiste** e il suo **peso**;
- per ciascun neurone interno o motore, i suoi **geni propri**;
- il **meccanismo di mutazione e di riproduzione**: i tassi di mutazione, la
  portata λ dei collegamenti, quanto spesso l'individuo, come madre, fa figli
  con un partner (gene `sex`, punto 14) e, con lo sviluppo, la precisione con
  cui il cervello si costruisce (gene `canal`);
- per ogni gene di recettore e di neurone interno, la sua **discendenza**:
  un'identità (quale gene è) e una famiglia (da quale gene antenato discendono
  lui e le sue copie). I geni dei primi individui hanno l'identità del loro
  posto, perché discendono tutti dagli stessi antenati; una copia nata da una
  duplicazione riceve un'identità nuova e la famiglia dell'originale.

È organizzato a **blocchi**: uno per neurone (interno o motore: se esiste, dove
sta, i geni propri e tutti i collegamenti *in ingresso*), uno per recettore,
uno per il meccanismo di mutazione. Nell'incrocio ogni blocco arriva intero da
un genitore.

### Genoma diretto o sviluppo (`genome`)

Con il genoma **diretto** (predefinito, e quello di tutte le corse di prima)
il genoma scrive ogni collegamento e ogni peso, come descritto sopra. Con lo
**sviluppo** (`genome = "sviluppo"`, riga "Genoma" del menu; dettagli in
[SPEC_SVILUPPO.md](SPEC_SVILUPPO.md)) scrive invece le regole per costruire il
cervello, come in natura (chemoaffinità di Sperry):

- **Marcatori**: molecole di superficie. Ogni recettore, neurone interno e
  motore esprime ciascun marcatore in una quantità tra 0 e 1, che dipende da
  un gene della cellula e, per i marcatori a **gradiente**, dalla sua
  posizione nel cervello (come Eph ed efrine, che costruiscono le mappe
  topografiche: due geni per marcatore, lungo x e lungo y). Si parte con 4
  marcatori su 6 posti; i posti crescono come quelli dei neuroni.
- **Asse sinistra-destra del fondatore**: i primi 2 marcatori di partenza sono una
  coppia di gradienti opposti lungo y (pendenza `marker_map_gradient` 12),
  ciascuno legato a se stesso (`marker_map_affinity` 2), con un po' di
  variazione tra individui (`marker_map_std` 0,3). Ogni cellula si collega
  soprattutto alla sua stessa metà: il recettore che guarda a destra allo
  sterzo verso destra, come la mappa del tetto ottico. Gli altri 2 marcatori
  partono a caso. Alla partenza il 52% degli individui nasce con un riflesso
  "gira verso ciò che vedi", il 13% con quello opposto, gli altri senza.
- **Tabella di affinità** (gene dell'individuo): quanto un marcatore
  sull'assone si lega a un marcatore sul bersaglio, con il segno.
- **Alla nascita**: affinità(s → t) = Σ espressione_s[a] · tabella[a, b] ·
  espressione_t[b]. L'attrazione che arriva al bersaglio si attenua con la
  distanza: |affinità| · e^(−d/λ), con λ la portata dell'individuo. Il
  collegamento nasce con probabilità sigmoide(k · (attrazione − 1)), spostata
  perché senza attrazione sia 0 (`develop_threshold` 1). È la
  **canalizzazione**: le vie attratte con forza nascono quasi sempre, le deboli
  quasi mai. k, la **precisione dello sviluppo**, è un gene dell'individuo
  (`canal`, partenza `develop_precision` 8). Un neurone con se stesso nasce
  con probabilità `net_self_prob` · (1 − e^(−|affinità|)). La sinapsi nasce
  **giovane, cioè debole**, come in natura: la sua forza cresce da zero con
  quanto l'attrazione supera la soglia (più contatti dove l'assone è attratto
  di più), `synapse_young` (1) · (attrazione − 1) · e^guadagno, con il
  **guadagno** gene del neurone che riceve (quanti recettori postsinaptici
  ha), al più ±`synapse_max`; il segno è quello dell'affinità, e nelle reti a
  impulsi viene dal tipo di chi trasmette. La guida decide con chi ci si
  collega; la forza la costruiscono poi il grembo, Hebb e l'omeostasi.
- Lo sviluppo è in parte **casuale**, ma solo vicino alla soglia: fratelli con
  lo stesso genoma hanno le stesse vie principali (alla partenza un clone
  ritrova l'88% dei collegamenti del genitore) e differiscono in quelle al
  limite. Il cervello sviluppato appartiene
  all'individuo: un genitore che sopravvive tiene il suo, ogni figlio sviluppa
  il proprio.
- **Mutazioni**: i pesi e i collegamenti non sono geni e non mutano; mutano le
  espressioni, i gradienti, la tabella e il guadagno (come gli altri geni). Un
  marcatore si **duplica** (la copia è espressa uguale ovunque e riga e
  colonna della tabella si dividono a metà: nessun cambiamento finché non
  diverge), si rompe o si ripara. Il gene di un marcatore ha lunghezza fissa
  7 (esistenza, gradiente, dominio di legame): come in natura non diventa più
  fragile perché esistono altri marcatori;
  tra copie della stessa famiglia vale lo scambio ineguale. Un neurone
  duplicato porta con sé i suoi marcatori: i suoi collegamenti nascono dallo
  sviluppo.
- **Incrocio**: ogni marcatore (esistenza, gradiente, riga della tabella)
  passa intero da un genitore, abbinato per identità; quello di un solo
  genitore passa con probabilità 1/2. Una cellula esprime ogni marcatore come
  nel genitore da cui viene (o, se lui non lo ha, come nell'altro).
- **Solo cervello**: anche i geni dei marcatori dei sensi sono fissi, uguali
  per tutti, mai mutati. Come le cellule della retina, però, i sensi seguono i
  gradienti secondo l'altezza a cui entra il loro nervo: l'occhio sinistro e
  il destro hanno un'etichetta di lato e l'asse sinistra-destra vale anche per
  loro (per i due marcatori dell'asse la parte propria dei sensi è nulla: il
  lato lo dice solo la posizione). I due sensi di una coppia (sinistro e
  destro) sono lo stesso organo: hanno gli stessi marcatori propri e
  differiscono solo per la posizione del nervo.
- **Il grembo** (net/womb.py): ogni neonato, prima di entrare nell'arena,
  vive `prenatal_time` (5 s) di **attività spontanea**. I sensi ricevono onde
  che scorrono lungo l'ordine dei nervi, come le onde della retina: sensi
  vicini sono attivi insieme, lontani no (più un rumore di fondo). La rete
  gira e agiscono Hebb, l'omeostasi delle sinapsi e della soglia; la
  plasticità modulata no (nel grembo non c'è ricompensa); il corpo non si
  muove. ā parte vuota e si costruisce con l'attività vera. Alla nascita il
  cervello rifinito, ā e la corrente di fondo diventano il **fenotipo**
  dell'individuo (`born_avg`, `born_bias`): ogni vita, in ogni arena, parte da
  lì. I sopravvissuti non rinascono. Con il genoma diretto non c'è grembo
  (non c'è un cervello sviluppato da rifinire): si nasce con ā = a*.

## 12. La generazione

- La popolazione è di **1.000 individui** (`population`): abbastanza perché la
  selezione pesi più del caso. Resta sempre di quella
  dimensione.
- Tutti gli individui di una generazione vivono nella stessa arena, nello stesso
  momento.
- **La generazione finisce** quando ogni individuo è morto oppure ha completato
  il compito, oppure dopo 240 secondi (`life_max`): nel mondo del cibo,
  mangiando, si può vivere a lungo.
- **Ogni generazione ha un'arena nuova**: zone di veleno, bersaglio e punti di
  nascita vengono risorteggiati.

## 13. Chi si riproduce

- **Si riproduce solo chi ha completato il compito**, cioè chi è **maturo**
  (nel mondo del cibo: chi ha mangiato almeno N cibi). Chi non ci arriva non
  lascia nulla.
- **Nel mondo del cibo, come in natura, più risorse vuol dire più figli**
  (`fecundity_by_food`): madri e partner si scelgono con probabilità
  proporzionale ai cibi mangiati in tutta la vita.
- **La vita continua dopo la maturità** (`mature_keep_living`): chi è maturo
  continua a nutrirsi e a rischiare; passano alla generazione dopo i maturi
  ancora vivi alla fine. (L'esame invece si ferma alla maturità.)
- Nel mondo del bersaglio tutti i genitori contano uguale e chi completa si
  ferma.

## 14. Come si forma la generazione successiva

1. **I genitori sopravvivono**: ogni genitore passa identico alla generazione
   successiva, dove dovrà rifare il compito in un'arena nuova.
2. **Nessun limite di età**: un individuo resta nella popolazione finché
   continua a completare il compito, e sparisce la prima volta che fallisce.
   (`max_age` permette di imporre un massimo di generazioni; 0, il default,
   vuol dire nessun limite.)
3. **I posti restanti vanno ai figli**: 1.000 meno i sopravvissuti.

Per ogni figlio:

- si sorteggia una madre, tutti i genitori con la stessa probabilità;
- con la probabilità scritta nel **gene `sex` della madre** il figlio ha anche
  un **partner**, un altro genitore a caso; altrimenti è una copia della madre.
  Il gene parte da 0,5 (`crossover_rate`) e muta come i tassi di mutazione:
  quanto conviene il sesso lo decide l'evoluzione (per ora senza costi);
- con un partner, come nella meiosi si appaiano i **geni omologhi**, quelli con
  la stessa identità, ovunque stiano nei due genomi. Per ogni coppia di
  omologhi il blocco viene dalla madre o dal padre con la stessa probabilità.
  I collegamenti seguono l'identità delle sorgenti: un ingresso dal naso resta
  collegato al naso anche se nel figlio il naso sta in un altro posto;
- un gene che ha **un solo genitore** passa al figlio con probabilità 1/2, come
  una variante che solo uno dei due porta (nel suo stesso posto, se è libero);
- **scambio ineguale**: in una famiglia di n copie, con probabilità tasso di
  duplicazione × (n − 1) due copie si appaiano per sbaglio, e se si somigliano
  (con probabilità pari alla somiglianza: 1 se identiche) il figlio riceve una
  copia in più o una in meno. È così che in natura le famiglie di geni
  crescono e si accorciano;
- poi il figlio viene **mutato**.

Di conseguenza il numero di figli per genitore dipende da quanti completano: con
12 genitori sono circa 80 a testa, con 500 genitori circa 1.

## 15. Le mutazioni

Si applicano a ogni figlio, ciascuna in modo indipendente. **Quanto muta un
figlio non è fissato da noi**: ogni individuo porta i propri tassi di
mutazione come geni. In natura i tassi dipendono dalle proteine che copiano e
riparano il DNA, scritte in geni come tutti gli altri, e quei geni mutano come
tutti gli altri. Qui vale lo stesso: i geni dei tassi vengono ritoccati con la
stessa probabilità e la stessa ampiezza dei pesi (in scala logaritmica: un
ritocco cambia un tasso di un fattore, per esempio del ±30%). Non esiste una
"velocità di cambiamento dei tassi" scelta da noi. Una nascita usa i tassi
ereditati; quelli eventualmente ritoccati valgono per i figli del figlio.
Nessun tasso scende sotto 1e-5 (`mutation_floor`): la copiatura perfetta non
esiste.

**Le ampiezze hanno un tetto naturale.** In natura gli effetti delle
mutazioni sono per lo più piccoli (la varianza che aggiungono in una
generazione è circa un millesimo di quella dovuta all'ambiente) e i mutatori
cambiano i tassi di un fattore. Un ritocco sposta quindi un gene al più di
circa 1 (quanto differiscono due individui alla partenza), una posizione di
0,3 lati del cervello, un gene di recettore di 0,2 volte la sua scala; la
portata dei collegamenti λ arriva al più a 3. Senza tetto l'ampiezza, che
muta con se stessa, si autoalimentava: senza selezione in 60 generazioni
arrivava a 100 e i geni a ±170 (corretto il 2026-10-21). Una sinapsi, anche
nel genoma diretto, non supera mai il suo massimo fisico (`synapse_max`), né
nel genoma né alla nascita; una velocità della plasticità (e^gene) non sale
oltre e^12 volte quella di partenza (ma può spegnersi del tutto).

| Mutazione | Valore di partenza del tasso | Costante iniziale |
|---|---|---|
| ritocco di un peso o di un gene proprio | 10% per ogni gene, di poco (±0,3) | `mut_weight_prob`, `mut_weight_std` |
| deriva dei geni di un recettore (sensibilità, sintonia, apertura, direzioni) | 10% per gene, ciascuno nella sua scala | `mut_rec_drift` |
| un collegamento assente compare | 1% tra neuroni vicinissimi, cala con la distanza con la λ dell'individuo | `mut_conn_add` |
| un collegamento presente sparisce | 1% | `mut_conn_del` |
| un neurone o recettore si accende o si spegne | 1% per posto | `mut_neuron_toggle` |
| un neurone o recettore acceso si **duplica** | 0,5% | `mut_dup` |
| un neurone (o il nervo di un recettore) si sposta nel cervello | 10%, di poco | `mut_pos_prob`, `mut_pos_std` |

**La duplicazione** è il modo in cui crescono cervelli e corpi sensoriali,
come in natura:

- il neurone o il recettore viene copiato nel primo posto libero, con gli
  stessi geni e gli stessi collegamenti in ingresso e in uscita; la copia
  nasce vicina all'originale nel cervello;
- i collegamenti in uscita di originale e copia vengono **dimezzati**: insieme
  fanno esattamente quello che faceva l'originale da solo. La duplicazione non
  cambia il comportamento, ma la copia costa, ed è libera di divergere;
- per un recettore le due copie hanno rumore indipendente: la loro somma è già
  un poco più precisa. Poi, se conviene, la sintonia di una copia deriva e
  nasce un recettore di tipo nuovo (come i coni dei colori).
- **un recettore sensibile si duplica come un gene dell'olfatto**
  (`receptor_dup_split`), dove ogni cellula esprime un solo gene recettore:
  non si copia l'organo intero, ma le cellule che ora esprimono il gene nuovo,
  l'equivalente di una cellula normale, passano alla copia con le loro
  sinapsi. La copia prende quella quota di sensibilità e di pesi in uscita,
  l'originale la perde: cellule, costo e segnale restano quelli di prima.
  Misurato su una popolazione con un naso di sensibilità circa 60: con la
  copia completano il 44,8% contro il 46,0% senza; con la copia intera di
  prima il 36,7%, e il costo del corpo raddoppiava. Così una copia resta a
  lungo quasi neutrale, come in natura, e può derivare verso un senso nuovo.
  I recettori che valgono meno di due cellule costano pochissimo e si copiano
  ancora interi, con le uscite dimezzate.

- Un collegamento che compare usa il peso già scritto in quel punto del genoma.
- Un collegamento che esiste resta anche se i suoi due neuroni si allontanano
  (l'assone c'è già), ma costa di più.
- Nelle reti a impulsi il tipo di un neurone cambia quando il ritocco del suo
  gene lo porta dall'altra parte dello zero.

## 16. I casi limite

| Quanti hanno completato | Cosa succede |
|---|---|
| due o più | regole normali |
| **uno solo** | regole normali: sopravvive e tutti i figli sono suoi |
| **nessuno**, ma in passato sì | si riparte dai genitori dell'ultima generazione che ne ha avuti: tutti figli, con mutazioni normali; nessuno sopravvive identico |

Non c'è nessun "soccorso" con mutazioni rinforzate quando i genitori sono
pochi: in natura non esiste.
| **nessuno**, e mai nessuno prima | popolazione primordiale nuova |

## 17. Ripetibilità

Tutto ciò che è casuale (arene, reti iniziali, mutazioni, incroci, scelta dei
genitori) viene da un generatore di numeri pseudocasuali che parte da un seme.
A parità di seme e di impostazioni, due corse danno esattamente le stesse arene,
le stesse reti e gli stessi risultati. Semi diversi danno corse diverse.

Questo non toglie casualità all'evoluzione: dentro una corsa i numeri si
comportano come se fossero casuali, e la selezione lavora su di essi come
farebbe sul caso vero. Il seme decide soltanto *quale* storia, tra tutte quelle
possibili, viene raccontata. Il vantaggio è poterla rifare identica: per
controllare un risultato, per capire un comportamento strano, per confrontare
due versioni del codice sulle stesse condizioni. Per questo il seme usato è
sempre mostrato e salvato, anche quando è estratto a caso.

Un risultato si giudica su più corse con semi diversi, non su una sola: una
corsa è un'estrazione, e la sua fortuna conta.

**I due motori.** Il motore GPU usa numeri a 32 bit e il generatore di numeri
casuali di JAX: è ripetibile con lo stesso seme, ma non dà gli stessi numeri del
motore NumPy, quindi una corsa fatta con un motore non si rifà identica con
l'altro. I salvataggi invece valgono per entrambi.

**La velocità non cambia il risultato.** Il tempo della simulazione non è
quello dell'orologio: ogni passo vale sempre 1/60 di secondo (`dt`), qualunque
sia la velocità scelta. x1, x16, turbo, pagina principale, massima velocità,
`--evolve` nel terminale o corsa in sottofondo cambiano solo quanti passi si
calcolano per ogni fotogramma, non i passi stessi; il disegno legge lo stato
senza toccarlo. Con lo stesso seme si ottengono le stesse generazioni (c'è un
test che lo verifica). Cambiano il risultato solo le azioni dell'utente: la
guida manuale (`M`) e le costanti regolate dal vivo (`C`), che valgono dal
momento in cui le si cambia.

Per andare più veloci senza cambiare nulla si può solo calcolare di più per
secondo: meno disegno (modalità massima velocità) e codice più efficiente. Per
esempio sensi e reti vengono calcolati solo per gli agenti ancora in gara, dato
che i comandi di morti e arrivati non hanno effetto, e la vista tratta tutte le
zone di veleno in un'operazione sola: con le stesse operazioni sugli stessi
numeri il risultato resta identico bit per bit (lo verificano i test). Allungare il passo `dt`, invece,
cambierebbe la simulazione: fisica più grossolana, vista e reti aggiornate meno
spesso, un agente veloce potrebbe attraversare il bersaglio senza toccarlo, e le
reti evolute non sarebbero più le stesse.

## 18. Il numero di ogni individuo

Ogni individuo riceve un numero alla nascita (`#123`) e lo conserva finché
resta nella popolazione: un sopravvissuto lo porta con sé di generazione in
generazione, ogni figlio ne riceve uno nuovo. L'**età** è quante prove ha vinto
di fila; il **più anziano** è quello con l'età più alta, ed è mostrato con il
suo numero e la generazione in cui è nato.

## 19. La selezione finale (facoltativa)

Con un numero di generazioni stabilito si può chiedere, alla fine, una
selezione senza figli (riga "Selezione finale" nel menu, `--select K` nel
terminale):

1. finite le generazioni la popolazione viene salvata come al solito;
2. restano in gara solo i **sopravvissuti**, cioè chi ha vinto l'ultima
   generazione; non nasce nessuno;
3. rifanno la prova K volte, ogni volta in un'arena nuova; **chi fallisce è
   eliminato**;
4. se in una prova falliscono tutti, la selezione si ferma e restano quelli che
   avevano superato la prova precedente (sono a pari merito);
5. i rimasti vengono salvati a parte, in `<nome>-selezionati.npz`, e la corsa si
   mette in pausa con il riepilogo di fine esperimento (vedi "Fine
   dell'esperimento" in [INTERFACCIA.md](INTERFACCIA.md)), da cui li si può guardare in arena.

Durante la selezione la pagina principale e l'arena dicono a che prova si è e
quanti sono in gara; al posto dell'ultimo grafico c'è l'esito di ogni prova.
Finita la selezione, `Spazio` riprende l'evoluzione: i selezionati corrono la
generazione in corso da soli, poi la popolazione torna piena con i loro figli.
Lo stesso succede riprendendo l'evoluzione dal file dei selezionati. Il file si
può anche aprire dal menu per esplorarne le reti o provarle una per una.

## 20. L'esame fisso

Il successo nel compito del momento non basta per dire se gli agenti stanno
diventando più capaci: se l'ambiente cambia, la percentuale cambia anche
quando gli agenti restano uguali. Per questo c'è un esame **sempre uguale e mai
usato per la selezione**, come un test di laboratorio. Lo fanno i **20
individui più anziani** della popolazione (a parità di età, i nati prima),
cioè quelli che hanno superato più prove di fila. È composto da 7 prove:

| Prova | Cosa cambia rispetto all'ambiente d'esame | Cosa misura |
|---|---|---|
| base | niente | il compito di sempre, in arene mai viste |
| lontano | il bersaglio nasce ad almeno 90 u da tutti | esplorazione |
| veleno+ | 24 zone di veleno invece di 12 | prudenza |
| veleno0 | nessun veleno | se il veleno era solo un ostacolo o anche un indizio |
| arena | arena quadrata 160 x 160 invece di 200 x 140 | generalizzazione |
| rumore | disturbo casuale su ogni senso (deviazione standard 0,1) | robustezza |
| mobile | il bersaglio si muove a 5 u/s e rimbalza sulle pareti | inseguire |

- **8 arene per prova, 5 copie di ogni rete per arena**: 40 tentativi per rete
  e per prova. Il punteggio di una prova è la media sulle 20 reti; il
  punteggio dell'esame è la media delle 7 prove; "migliore" è la rete con la
  media più alta.
- **Non segue l'ambiente della corsa.** Le costanti dell'ambiente d'esame
  (fisica, energia, veleno, vista...) sono scritte in `evo/exam.py` e non
  cambiano se cambi quelle della corsa. Dalla corsa vengono solo il tipo e la
  forma della rete. Anche le arene hanno semi fissi, diversi da quelli
  dell'evoluzione e del banco di prova.
- **Non cambia l'evoluzione**: usa copie delle reti e generatori casuali
  propri. Farlo o non farlo dà la stessa corsa.
- **Gira in processi separati**, su tutti i core tranne due: circa 12 secondi
  con 12 core, senza rallentare l'interfaccia. Ogni arena è un lavoro a sé,
  quindi il risultato non dipende da quanti processi lo fanno.
- **Quando**: in automatico ogni `exam_every` generazioni (riga "Esame ogni"
  del menu, `--exam-every K` nel terminale, default 50; se il precedente è
  ancora in corso quello nuovo si salta), con `E` sulla pagina principale, o
  con `--exam FILE` su un salvataggio. I risultati finiscono nel salvataggio.
  Un esame non finito quando si esce dalla corsa va perso.
- **Mondo del cibo**: al posto delle 7 prove ci sono 8 livelli fissi lungo la
  scala (esame versione 4, [SPEC_MONDO.md](SPEC_MONDO.md)).
- **Con lesione** (`L` sulla pagina principale, `--exam FILE --lesion X` nel
  terminale): lo stesso esame con una parte del cervello tolta. Il risultato
  è registrato a parte e non conta come esame della corsa.
- Se un giorno le prove cambiano, cambia anche il numero di versione
  dell'esame, salvato con ogni risultato: punteggi di versioni diverse non si
  confrontano.

### Un punto di riferimento

In una prova da 120 generazioni con le impostazioni di default (reti continue) i
successi sono passati dal 2% a circa il 30–40%, ancora in crescita, in meno di
quattro minuti. La percentuale oscilla molto da una generazione all'altra perché
alcune arene sono più difficili di altre. È una prova sola: i numeri sono
indicativi.

## 21. I salvataggi

I salvataggi (`evo/saves.py`) si scrivono in modo sicuro: prima un file
temporaneo, poi lo si mette al posto del vecchio in un colpo solo. Se il
programma si interrompe a metà, il salvataggio precedente resta intatto.

I salvataggi contengono la popolazione, la storia e il campione di ogni
generazione (circa 20 KB l'uno: una corsa di migliaia di generazioni occupa
qualche decina di MB, in memoria e su disco). Contengono anche la
**configurazione completa** e il **registro dell'ambiente**: quando la corsa
è nata o è stata ripresa, con quale versione del codice (il commit git,
"+modifiche" se c'erano modifiche non committate), e ogni costante cambiata,
con la generazione in cui è cambiata (anche i ritocchi fatti con `C`).
**Riprendendo una corsa si gioca sempre con le regole più recenti.** Dal
salvataggio vengono la popolazione con i suoi genomi e lo **stato della
corsa**: tipo di rete, popolazione, posti per neuroni e recettori, mondo
scelto (compito e veleno), modo e posizione della scala, frequenza
dell'esame (`RUN_STATE` in `evo/evolution.py`). Tutte le altre costanti sono
regole (fisica, energia, sensi, costi, cibo, odori, mutazioni, soglie della
scala, durata della vita...) e vengono dal codice attuale. Ogni regola diversa
da quella con cui la corsa era stata salvata finisce nel registro alla
generazione della ripresa, con il valore vecchio e il nuovo; anche i ritocchi
fatti con `C` valgono quindi fino alla ripresa. Il registro compare in cima
alla pagina principale, nel menu accanto al salvataggio e nel terminale con
`--load`.

Attenzione: la posizione della scala si conserva, ma se le regole sono
cambiate molto la stessa posizione può voler dire un mondo più difficile (per
esempio più cibi da mangiare). La scala allora scende da sola, passando per
generazioni con pochi genitori.
Il formato attuale è il 7 (recettori universali con adattamento e sensi
calibrati, posti che crescono, mutazioni come geni, plasticità, cervello che
cresce con le sue cellule). Quelli dei formati da 3 a 6 si aprono convertiti
(senza adattamento, senza plasticità, sensibilità ricalcolate come
moltiplicatori a parità di quanti, distanze ricalcolate), con le
costanti con cui erano nati, e il registro lo annota. Quelli precedenti avevano i
sensi fissi e non si possono più aprire.

## 22. L'arena continua

Una modalità a parte (`task = "arena"`, riga "Mondo" del menu): tutti gli
esseri vivono insieme in un mondo unico di 2000 × 1400 senza bordi, senza
generazioni né compito. Si nasce quando un adulto (almeno 30 s) ha almeno
250 di energia e non ha partorito negli ultimi 30 s: il figlio riceve 120, il
genitore ne paga 160, e se un adulto compatibile (metà delle identità dei
geni in comune) è entro 20 u, con la probabilità del gene `sex` il figlio è
un incrocio dei due. Si muore di fame o di vecchiaia (legge di Gompertz, vita
mediana 600 s). Il cibo ricresce (45 al secondo, al massimo 1500) a 8
macchie fertili che si spostano piano e cambiano con le stagioni. Gli esseri
si vedono (colore 0,25), si annusano (molecola 0,45) e si urtano; chi tocca
per primo un cibo lo mangia. Sotto i 100 vivi arrivano colonizzatori, copie
mutate di chi ha mangiato di più. L'esame fisso (i livelli del mondo del
cibo) si fa ogni 10 minuti di mondo su 20 adulti presi a caso. Dettagli e
numeri in [SPEC_ARENA.md](SPEC_ARENA.md).

**Il mondo che si complica** (solo con il motore JAX; dettagli in
SPEC_ARENA.md, sezione 15). Nell'arena ogni essere ha una **taglia** (gene:
più grande = morso più forte e più riserva, ma più lento e più caro) e
quattro **azioni** comandate da neuroni motori in più: mordere, accendere la
propria luce (colore: gene), emettere il proprio odore (molecola: gene),
nutrire un proprio figlio. Sei manopole, che il mondo alza una alla volta
quando la popolazione si regge da sola e abbassa quando torna ad aver bisogno
di colonizzatori: macchie che si esauriscono, frutti cattivi, inversione del
colore buono, predazione (si morde chi è almeno 1,2 volte più piccolo),
cibo da aprire (duro e grande), notte. Dopo ogni esame, la stessa prova senza
plasticità misura quanto imparano durante la vita.

**Cibo e clima** (solo con il motore JAX). Il cibo nasce solo nelle macchie.
La prima manopola della scala del mondo lo rende raro (fino al 40% della
ricrescita e al 35% dei cibi presenti). Il clima: fascia calda al centro del
mondo, fresca verso i bordi; al caldo il metabolismo è fino a 1,7 volte e le
macchie producono fino a 1,75 volte, al freddo il metabolismo scende a 0,6
volte e le macchie a 0,25 volte. I recettori della temperatura la sentono.


**Acqua** (solo con il motore JAX). Una seconda riserva, che cala sempre (di
più al caldo e correndo): a zero si muore di sete. Si beve stando dentro una
pozza; le pozze sono ferme, lontane dal cibo ma non troppo, si restringono
bevendo e spariscono vuote; quando ne restano meno di 4 ne nasce una nuova.
Fare un figlio costa acqua. Le pozze si vedono (blu), si annusano da lontano
(umidità) e la sete si sente da dentro. L'ottava manopola, la siccità, le fa
evaporare.

**Sete graduale e neuroni iniziali** (arena continua). La sete è tarata sulla
scala dell'energia (senza bere ~140 s da fermi) e una manopola della scala
del mondo la porta fino al doppio; le pozze nascono appena fuori dalle macchie e una manopola le
allontana; il figlio nasce idratato. Gli esseri primordiali hanno 2 neuroni
interni già collegati, e un neurone che compare in un figlio nasce collegato.

**Piacere secondo il bisogno** (arena continua, motore JAX). Mangiare e bere
danno un segnale positivo al sistema di valore tanto più forte quanto più si
ha fame o sete: da affamati un cibo vale come prima, da sazi quasi niente.
