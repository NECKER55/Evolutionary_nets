# Cervello ragionante (progetto)

Stato: progetto. Del codice esistono il mondo 3D con le pile, i due agenti di
controllo e, del cervello, la sezione C (`evonets/brains/ragionante/`); il resto no. Un **nuovo tipo di cervello**, accanto a quelli di oggi: il cervello a sezioni (docs/SPEC_CERVELLO_SEZIONI.md) resta il riferimento da
battere. Il nome è provvisorio. Sostituisce la proposta di
docs/SPEC_PIANIFICAZIONE.md, di cui tiene il criterio della rivalutazione e
l'arena delle casse. I numeri di neuroni sono ordini di grandezza da tarare.

Il mondo è **3D**, con MuJoCo per la fisica e le immagini, e il sensore è una telecamera
finta. Le sezioni si addestrano **una alla volta**: ognuna deve superare un esame prima
di entrare nel cervello.

## L'idea

Il cervello percepisce il mondo, sceglie a che cosa prestare attenzione in base a
ciò che sta pensando, ricorda ciò che succede, capisce che cosa provoca che cosa,
conserva le situazioni che contano e ragiona su due livelli: che cosa voglio
ottenere e che cosa faccio adesso.

Il gradiente non insegna il mondo. **Insegna a imparare il mondo.**

## Schema delle sezioni

Ogni sezione ha una lettera:

| lettera | sezione | domanda a cui risponde |
|---|---|---|
| **P** | Percezione | che cosa c'è davanti a me? |
| **A** | Attenzione | su che cosa mi concentro? |
| **L** | Memoria di lavoro, a breve termine | che cosa ho in mente adesso? |
| **M** | Memoria spaziale, a breve termine | dove sono, e dove sono le cose? |
| **W** | Modello del mondo | che cosa succede se faccio questo? |
| **R** | Ricordi, memoria a lungo termine | che cosa so di questa situazione? |
| **D** | Decisione e ragionamento | che cosa conviene fare? |
| **C** | Comandi, il motore | come muovo questo corpo? |
| **B** | Bisogni | che cosa mi serve adesso? |

```
                                   MONDO E CORPO
      telecamera        sensi del corpo      piacere, dolore      energia, fame
          │                    │                    │                   │
          ▼                    │                    │                   ▼
  ┌───────────────┐            │                    │          ┌────────────────┐
  │ P. PERCEZIONE │            │                    │          │ B. BISOGNI     │
  └──┬─────────┬──┘            │                    │          └───────┬────────┘
     │ oggetti │ scena         │                    │                  │ (va a W, D)
     ▼         │ sfocata       │                    │                  │
  ┌──────────┐ │               │                    │                  │
  │ A. ATTEN-│◄┼───────────────┼────────────────────┼──────────────────┼─ domanda da D
  │    ZIONE │ │               │                    │                  │
  └──┬───────┘ │               │                    │                  │
     │ vettore │               │                    │                  │
     │ attento │               │                    │                  │
     ├─────────┼─────────┐     │                    │                  │
     │         ├─────────┼──┐  │                    │                  │
     │         │         │  │  │                    │                  │
     ▼         ▼         ▼  ▼  ▼                    │                  │
  ┌───────────────┐   ┌──────────────────┐          │                  │
  │ L. MEMORIA    │◄──│ M. MEMORIA       │◄─────────┼──────────────────┼─ domanda da D
  │    DI LAVORO  │   │    SPAZIALE      │◄─────────┼──────────────────┼─ ricordo da R
  │ a eventi      │◄──┼──────────────────┼──────────┘                  │
  │               │   │ a ogni passo     │                             │
  └──┬─────────┬──┘   └──┬───────────┬───┘                             │
     │         │ residuale, verso D  │ direzione, verso C              │
     │         │         │ luogo     │                                 │
     │         │         ├───────────┼───────────────┐                 │
     ▼         │         ▼           │               │                 │
  ┌──────────────────────────┐       │               │                 │
  │ W. MODELLO DEL MONDO     │◄──────┼───────────────┼─────────────────┤
  │ che cosa succede se...   │◄──────┼───────────────┼─────────────────┼─ azione immaginata, da D
  │                          │◄──────┼───────────────┼─────────────────┼─ azione eseguita, da C
  └──┬──────────────────┬────┘       │               │                 │
     │ errore           │ simulazioni│               │                 │
     ▼                  │            │               │                 │
  ┌──────────────────┐  │            │               ▼                 │
  │ R. RICORDI       │◄─┼────────────┼───────────── luogo, da M        │
  │ fatti e luoghi,  │◄─┼────────────┼───────────── memoria da L       │
  │ a lungo termine  │◄─┼────────────┼───────────── richiesta da D     │
  └──┬───────────────┘  │            │                                 │
     │ ricordi          │            │                                 │
     ▼                  ▼            │                                 │
  ┌──────────────────────────┐       │                                 │
  │ D. DECISIONE E           │◄──────┼─────────────────────────────────┘
  │    RAGIONAMENTO          │◄──────┼─ memoria da L, luogo da M, azione eseguita da C
  │ valutazione; alto: scopo │       │
  │ e piano; basso: prossima │       │
  │ azione                   │       │
  └──┬───────────────────────┘       │
     │ azione voluta                 │
     ▼                               ▼
  ┌──────────────────────────────────────┐
  │ C. COMANDI (MOTORE)                  │◄─ movimento sentito, dai sensi del corpo
  │ traduce l'azione per il corpo        │─► azione eseguita, verso L, W, D
  └──┬───────────────────────────────────┘
     ▼
   MONDO E CORPO
```

Uscite che non si possono disegnare senza incrociare tutto: D manda la **domanda** ad A e M, la
**richiesta** a R, l'**azione immaginata** a W. R manda il **ricordo del luogo** a M.

Il **movimento sentito** viene dai sensi del corpo, non dai comandi: è ciò che il corpo ha fatto
davvero, quindi non cambia se cambiano i motori.

**Le lingue tra le sezioni.** Il vettore attento è nella lingua di P, perché A sceglie e non traduce.
Ciò che R restituisce a M è ciò che M stessa ha scritto. Una domanda di D è un tipo più l'embedding
di ciò che si cerca. Chi arriva dopo impara a parlare a chi è già fatto: D imparerà il formato di M.

**Le memorie a breve termine sono due, separate e collegate:** L tiene a mente *che cosa*, M tiene
*dove*. Come nel cervello, il concetto e lo spazio non stanno nello stesso circuito, e hanno ritmi
diversi: M cambia a ogni passo, L quando c'è qualcosa da scrivere. **R ne ha una sola controparte:**
tiene sia i fatti sia i luoghi.

## Due anelli di apprendimento

| | anello esterno | anello interno |
|---|---|---|
| quando | prima della vita, su moltissime vite | durante una vita |
| che cosa cambia | pesi innati, dinamiche delle sezioni, regole di plasticità | attività delle memorie, sinapsi plastiche |
| strumento | gradiente, più l'evoluzione per le costanti | nessun gradiente: solo attività e regole locali |
| che cosa si impara | **come imparare** | **questo mondo**: quale pila è buona, dove stanno, che cosa è pericoloso |

Una vita non sa quale colore è buono. Sa cercarlo, perché l'anello esterno ha
costruito un cervello capace di farlo in mondi sempre diversi. Il piacere e il dolore
in vita sono **ingressi**, non errori da propagare all'indietro. È il
meta-apprendimento per rinforzo: secondo Wang e colleghi (2018) la corteccia
prefrontale funziona così.

Rispetto alla regola di docs/SPEC_ARENA_PILE.md ("gradiente solo come diagnosi, mai per
addestrare gli esseri"), resta vero che nessun essere impara il mondo col gradiente mentre
vive. L'eccezione è il motore (sezione C), che così impara soltanto il proprio corpo.
Cambia che il suo cervello di nascita è prodotto anche dal gradiente.

## Addestrare una sezione alla volta

Se tutte le sezioni imparano insieme, niente garantisce che ognuna faccia ciò che dice il
suo nome. Quindi ogni sezione si addestra da sola su un compito suo, viene esaminata e poi
si blocca. La sezione dopo impara con quelle prima già bloccate.

| sezione | esame, sempre in mondi mai visti | compagni |
|---|---|---|
| C | si tara da sola in corpi sconosciuti, raggiunge l'oggetto indicato e lo morde (**superato**) | comandante finto |
| L e W | dopo pochi eventi prevede il prossimo evento e il suo esito | in coppia: che cosa ricordare dipende da che cosa serve prevedere |
| M | dopo aver girato indica dov'erano gli oggetti non più inquadrati | percezione e movimento sentito |
| R | richiama l'esito giusto dopo molti eventi diversi, anche partendo dall'esito; regge ai cambi di regime | sequenze generate da noi |
| A | con una domanda data da noi sceglie l'oggetto giusto; un pericolo vicino passa comunque | domande date da noi |
| D basso | raggiunge un sotto-scopo dato | A, L, W, R, C bloccate |
| D alto | **rivalutazione alla prima prova** | tutto bloccato, anche D basso |
| P | trova gli oggetti e dà la stessa interfaccia del detector finto (**superato**, con le due reti: vedi "P. Percezione") | etichette del simulatore |

Regole:

- **Imparare, non memorizzare.** L'esame di L, W e R è "impara in fretta un mondo mai visto",
  non "conosci questo mondo". Ogni sezione si allena su molti mondi casuali, con i due anelli.
- **Chi agisce mentre W impara.** A W non servono azioni giuste, servono azioni varie con il
  loro esito vero. Agiscono gli agenti di controllo, con errori aggiunti apposta: un agente
  che non morde mai una pila cattiva non insegnerebbe il dolore, uno che vaga a caso non
  aprirebbe mai una cassa. Quando esiste un cervello, si alterna la raccolta con lui.
- **Che cosa spinge a prevedere.** Il mondo dà la risposta un istante dopo, senza maestro: si
  misura l'errore tra previsione e realtà lungo tutta la vita, e il gradiente dell'anello
  esterno lo riduce. La catena si allena rimettendo in W le sue stesse previsioni.
- **Ritocco finale.** Una sezione bloccata può aver scartato un dettaglio che serviva più
  avanti. Alla fine si addestra tutto insieme, a passo piccolo, e si **ripetono tutti gli
  esami**: una sezione che non li supera più ha perso il suo significato.
- **Che cosa non garantisce.** L'esame dice che W sa prevedere, non che D la usi. Lo dicono
  solo la rivalutazione e le lesioni.

## Il mondo in 3D

**MuJoCo** (3.17) simula i corpi e produce le immagini. Tutto il resto è codice nostro:
`Mondo3D` eredita da `PileWorld` e ne cambia solo il movimento, quindi energia, morso, zone e
inversioni sono **le stesse regole**, e funzionano già anche `senses()` e `objects()`.

| di chi | che cosa |
|---|---|
| MuJoCo | corpi, collisioni, contatti, telecamera, profondità, oggetto per ogni pixel |
| nostro | energia e fame, pile, morso, casse e interruttori, regole, mondi casuali, cervello, interfaccia grafica |

Il morso resta una regola nostra: MuJoCo dice solo che due oggetti si toccano.

**Interfaccia del mondo.** Il cervello e la grafica vedono solo: stato, azione, immagine,
oggetti. Non chiamano mai MuJoCo direttamente, così il motore si può sostituire.

**Primo mondo.** L'arena delle pile in 3D (fatta): 5 m di lato, pile cilindriche che un urto
sposta, che nascono lontane dagli esseri e compaiono e spariscono in 0,6 s. Corpo su due ruote con telecamera a 120° in orizzontale, immagine 264×264 più profondità in cm; si può scegliere anche 640×480 o 1280×720, con la stessa apertura orizzontale (13,6 ms per immagine contro 3,2). La telecamera
sta dentro il corpo, 8 cm davanti al centro: sul muso, a contatto, finirebbe dentro ciò che tocca. Il
telaio si inclina al massimo di 6° perché ribaltandosi resterebbe bloccato. Poi l'arena
delle casse, con casse e interruttori come forme colorate. Le zampe, se serviranno, arrivano dopo.

**Mappe** (`maps.py`, fatta): "aperta" è l'arena di sempre; "stanze" sono 9 stanze in 12 m e
"labirinto" 25 stanze in 13 m, unite da porte, con la pianta decisa dal seme. Ogni stanza ha muri di un
colore, e due stanze possono averlo uguale. Le pile stanno in una delle quattro stanze d'angolo.
Niente arancio dentro: da lontano, oltre una porta, sembrava una pila.
Il perfetto, che conosce la strada, fa 10,9 pile al minuto con le stanze. Il vedente, che non ha
memoria dello spazio, ne fa da 0 a 0,25 e muore di fame: è la misura che M deve migliorare. La rete
visiva è riaddestrata anche sulle stanze: 0,08 pile fantasma per immagine, 0,03 nel labirinto.
Nell'arena aperta fa 8,1 pile al minuto contro 8,9 dei pesi di prima, su 3 semi da 6 a 11: una
differenza dentro il rumore, ma non è provato che non costi nulla.

**Agenti di controllo** (`evonets/brains/mondo3d/`), non cervelli: un cervello si misura tra i due.

| agente | che cosa sa | pile buone/min | cattive/min | scelta |
|---|---|---|---|---|
| **perfetto** (soffitto) | legge la verità dal simulatore; non usa la telecamera | 11,7 | 0 | 1,00 |
| **base** (pavimento) | solo la profondità: vaga e morde ciò che ha davanti | 0,06 | 0,06 | 0,50 |

Misure con 1 essere per recinto: perfetto su 4 recinti per 20 minuti, base su 4 recinti per 12
minuti (solo 6 pile: la scelta è indicativa). Il cervello ragionante con la sola sezione C, 3 recinti per 5
minuti compreso il minuto di taratura, fa in media 9,25 pile buone al minuto sui 6 corpi di
prova, con 3 semi (minimo 8,1, massimo 10,6). Il vedente, con la percezione vera dalla
telecamera, fa circa 8 pile al minuto su 3 recinti per 4 minuti, senza morti. Il perfetto ha la scelta perfetta, ma la strada
è avida: non è il massimo teorico di pile al minuto. Il guidato dell'arena 2D, con le
stesse regole, fa 9,0 pile al minuto, di cui il 96% buone.

**Immagine e cervello.** Ogni immagine sono 209.088 numeri a ogni passo. La telecamera è nel
mondo da subito, ma nel cervello entra con il detector vero. Fino ad allora il **detector finto** legge gli oggetti dal simulatore,
tiene solo quelli inquadrati e li passa con la stessa interfaccia.

**Interfaccia grafica.** Resta la nostra in pygame (`evonets/viz/mondo3d/`): MuJoCo disegna
senza aprire finestre, in un thread suo perché quello principale ha il contesto di SDL, e
l'immagine va in una superficie pygame. Il visualizzatore di MuJoCo
non basta, perché mostra il mondo ma non la rete. Sullo schermo:

- il mondo, da una telecamera esterna;
- ciò che vede l'essere, con oggetti trovati e attenzione;
- attività delle sezioni, memorie e catena dell'immaginazione.

Col cervello ragionante la schermata mostra anche la sezione C: neuroni, sinapsi, cablaggio
nascosto e sensi, con i pulsanti "Tara i sensi" e "Cambia corpo". Col vedente, sull'immagine
compaiono i riquadri degli oggetti trovati, con la loro distanza.

Sullo schermo la profondità ha una scala logaritmica a colori, perché più dell'80% dei pixel
che non sono cielo è pavimento a meno di 50 cm: la telecamera è bassa (11 cm) e larga. La
distanza è misurata lungo l'asse della telecamera.

**Lo scambio.** Quali pile sono buone, i conti alla rovescia, i dischi sul pavimento e
l'annuncio sono **solo per noi**. Gli esseri non vengono avvisati: lo scoprono mordendo, ed è
ciò che si misura. I segni non entrano mai nella telecamera (c'è un test).

Del terzo punto c'è solo lo stato del corpo: le attività arrivano con il cervello. Le immagini
per noi possono avere dimensioni maggiori. Quella data alla rete deve restare uguale
all'addestramento.

**Misure.** Il 2026-10-31, su questo portatile (i5-1335U, GPU integrata Intel, niente
NVIDIA), con 20 sfere, pavimento e una telecamera 64×64 (la risoluzione cambia poco il
disegno), senza cervello:

| caso | immagini al secondo |
|---|---|
| con ombre, 1 processo | 170 |
| senza ombre, 1 processo | 850 |
| senza ombre, 2-8 processi | 8.300-9.600 in totale |

Fisica: 42.000 passi al secondo su un core, con passi da 5 ms. Le ombre vanno quindi spente in
addestramento (`shadowsize="0"`, `offsamples="0"`), e servono almeno 2 processi. La lentezza
con un solo processo non è spiegata. A 20 Hz e 9.000 immagini al secondo bastano per 450
esseri in tempo reale.

Nel mondo vero, oggi in un solo processo: il perfetto (senza immagini) fa 70 s di mondo al
secondo con 4 esseri e 5,5 con 100 esseri; il base, che legge la profondità a 264×264, 18 con 1
essere.

## Il mondo a eventi

Le sezioni che lavorano a eventi (L, W, R, D) hanno bisogno di molte vite, non di fisica: nel 3D
costerebbero circa 13 giorni di calcolo. Il mondo a eventi non simula niente: ogni vita è una
lista di righe "oggetto, azione, esito", generata da una regola nascosta. Gira in JAX.

**Regole tra le vite.** Ogni vita ha la sua regola, e il cervello deve scoprirla vivendo. Se
fossero tutte uguali la memorizzerebbe. I livelli, uno alla volta:

1. l'esito dipende da una caratteristica, con scambi;
2. in ogni vita conta una caratteristica diversa, o una combinazione;
3. l'esito arriva qualche evento dopo;
4. sequenze: "premo X, si apre Q".

Una famiglia di regole resta fuori dall'addestramento, per l'esame. Le sezioni si provano poi,
bloccate, su eventi veri registrati nel 3D: senza, funzionerebbero solo sugli eventi inventati.

**Imparare una lista o imparare a imparare.** Con poche famiglie (colore, grandezza, "e") la rete
impara quelle: su "vicinanza", mai decisiva in addestramento, fa il 50%. Con la famiglia larga, 92
regole a soglia sulle caratteristiche, riconosce anche le regole tenute fuori. Non è una regola
universale: generalizza dentro la famiglia di mondi che ha visto. Tre difetti trovati, tutti da
ricontrollare per ogni livello: l'esito dentro gli ingressi (il bisogno), caratteristiche su scale
diverse (si impara solo il colore), e scambi troppo regolari (si impara quando avvengono).

| morsi giusti dopo 5, vite mai viste | limite | ricordo dell'ultimo esito | GRU, 3 famiglie | plastica, 3 famiglie | GRU, famiglia larga |
|---|---|---|---|---|---|
| regole viste | 0,96 | 0,80 | 0,97 | 0,96 | 0,93 |
| "o", mai vista | 0,96 | 0,78 | 0,82 | 0,76 | **0,86** |
| "vicinanza", mai vista | 0,97 | 0,80 | 0,50 | 0,50 | **0,90** |
| morsi veri dal 3D, con 2 scambi | | | 0,98 | 0,98 | 0,98 |

Il limite tiene una probabilità per ogni regola ma non sa che due scambi distano almeno 10 eventi:
per questo la rete può superarlo di poco. La plastica ha visto 768.000 vite, la GRU 2 milioni, la
famiglia larga 5 milioni. Senza la memoria si torna intorno al 50%. Gli eventi veri sono solo
regole di colore: non provano la generalizzazione nel 3D.

## Le sezioni

### P. Percezione

- **Fa:** separa gli oggetti e li descrive.
- **Entra:** immagine della telecamera. **Esce:** un embedding per oggetto, fino a un massimo fisso.
- **Anello esterno:** impara a vedere. **Anello interno:** non cambia, o molto poco.
- **Interfaccia:** `Percept` (`vision.py`), uguale per tutti: che cos'è, angolo, distanza,
  larghezza, embedding. Nessun detector sa quali oggetti siano buoni.
- **Detector finto:** legge il simulatore. Costa niente, ma vede attraverso i muri: serve
  ad allenare le altre sezioni, non a misurare.
- **Rete nostra** (`Detector`): piccola rete convoluzionale, 18 KB, addestrata sulle etichette
  del simulatore. Dice che cosa c'è in ogni pixel; le macchie sono gli oggetti, la profondità
  dice dove sono. Conosce solo i tipi su cui è stata addestrata.
- **Reti pronte, oggi** (`ReadyDetector`, opzionali, gruppo `percezione`): una regola sulla
  profondità indica che cosa sporge dal pavimento, **SlimSAM** ne fa la maschera, **DINOv2 small**
  più il colore fanno il vettore, `ObjectMemory` lo conserva anche su disco e dice se l'oggetto è già noto. Il
  `Tracker` lo segue tra un'immagine e l'altra. Niente etichette: per i nomi bastano 5 esempi
  per tipo. Un oggetto mai visto si riconosce la volta dopo senza riaddestrare nulla.
- **Da solo DINOv2 vede quasi solo la forma:** un cubo blu e uno arancione avevano somiglianza
  0,91. Per questo il colore si aggiunge al vettore.

| esame in mondi mai visti | rete nostra | reti pronte |
|---|---|---|
| pile trovate, guidando | 88% | 96% |
| rilevamenti giusti, guidando | 97% | 97% |
| pile trovate, scene difficili (muri, pile a un passo) | 90% | 92% |
| rilevamenti giusti, scene difficili | 85% | 89% |
| errore di angolo / distanza | 0,9° / 2,4% | 0,6° / 1,2% |
| tempo per vista, su CPU | 19 ms | 3,2 s |
| pile/min in guida (1 essere, 4 min) | 8,9 | 7,4 |

In guida le reti pronte guardano ogni 0,5 s di mondo, ma a ogni sguardo il mondo le aspetta:
su questa macchina non reggono il tempo reale.

**Identità** (`exam_identity.py`), con oggetti mai visti, memoria ricaricata da disco e nuovi
punti di vista: 18 su 18 riconosciuti, nessuno scambiato, 18 su 18 oggetti nuovi detti nuovi,
con soglia 0,8. Un oggetto che cambia colore sotto gli occhi resta lo stesso grazie al
tracker; fuori dalla vista diventa un oggetto nuovo. Due oggetti che si toccano si separano se
sono diversi. **Due oggetti uguali, anche uno dietro l'altro, restano uno solo.**

**Il limite di oggi:** il vettore resta dentro P. Serve solo a dare un nome tra tre (pila rossa,
pila gialla, essere: il più simile di 15 esempi, senza soglia, quindi anche ciò che non somiglia a
niente riceve un nome) e l'identità. Fuori da P escono solo nome, angolo e distanza: per il resto
del cervello reti pronte e detector finto sono quasi uguali.

#### Il flusso scelto (da costruire)

Niente nomi: P dice com'è fatto ogni oggetto, e che due oggetti simili stanno insieme lo
impareranno le sezioni dopo, dalle conseguenze.

1. **Telecamera** 1280×720 a colori, più la profondità di ogni pixel.
2. **FastSAM-s** (YOLOv8 da segmentazione, circa 11 milioni di pesi, licenza AGPL) trova in un
   colpo solo tutte le cose dell'immagine, senza nomi: una maschera per cosa. Nessuna regola
   sull'altezza, quindi anche oggetti alti, bassi o vicini al muro.
3. **Filtro** delle maschere che sono sfondo: pezzi di muro, pavimento e cielo, circa 12 per
   immagine. Da decidere e misurare: un filtro generale su area e profondità, oppure lasciarle
   passare e far imparare all'attenzione a ignorarle.
4. **Profondità sulla maschera:** la mediana dei suoi pixel è la distanza z. Da z e dalla posizione
   nell'immagine vengono angolo, distanza dal centro dell'essere e larghezza vera in cm.
5. **Ritaglio a scala fissa:** un quadrato di 40 cm veri attorno all'oggetto (a z cm sono
   40 · 640 / tan 60° / z pixel), grigio fuori dalla maschera. L'oggetto occupa nel quadrato la
   parte che corrisponde alla sua misura vera, a ogni distanza.
6. **DINOv2 small** porta il quadrato a 224×224, lo divide in 16×16 pezzi da 14 pixel e restituisce
   il riassunto: 384 numeri. Più 3 numeri di colore, pesati 1,5: 387.
7. **Riduzione** a 64 numeri con la PCA della banca: un calcolo fisso, non una rete.
8. **Esce, per ogni oggetto** (al massimo 12, dal più vicino): vettore 64, larghezza in cm, angolo
   (seno e coseno), vicinanza, "c'è". Va all'attenzione A, che sceglie; L e M ricevono lo stesso
   vettore, non una traduzione.

**Segmentatori a confronto** (180 scene del 3D, 771 oggetti veri: pile, pile alte 30 cm, forme e
colori mai visti, esseri; trovato = una maschera lo copre almeno a metà; CPU, 12 core):

| | trovati 264 | trovati 1280 | a 2–3,5 m, 1280 | alti 30 cm, 1280 | maschere in più per immagine | s per immagine, 1280 |
|---|---|---|---|---|---|---|
| **FastSAM-s** | 93% | **96%** | **94%** | **100%** | 12 | **0,4** |
| FastSAM-x | 92% | 96% | 95% | 99% | 12 | 3,0 |
| CutLER, soglia 0,35 | 51% | 62% | 44% | 62% | 4,5 | 4,2 |
| CutLER, soglia 0,05 | 75% | 86% | 77% | 88% | 16 | ~6 |
| SlimSAM con la regola della profondità | 86% | 83% | 85% | 33% | 1 | ~2,7 |

CutLER, addestrato senza etichette su foto di ImageNet, in un oggetto mancato su tre non lo vede
affatto: oggetti piccoli, sintetici e a tinta unita sono lontani da ciò che conosce. È anche il
più lento, e la licenza è solo non commerciale. La regola della profondità sbaglia proprio dove
si temeva: gli oggetti alti li prende per muri. FastSAM-x costa 7 volte FastSAM-s senza trovare di più.

**Ritagli a confronto** (31 tipi di oggetto, 100 viste ciascuno, le stesse a 264 e a 1280; il
vicino più simile ha la stessa caratteristica?):

| | colore | forma | grandezza | tipo esatto |
|---|---|---|---|---|
| 264, ritaglio stretto | 100% | 99,1% | 76,3% | 76,7% |
| 264, ritaglio stretto + larghezza | 100% | 99,5% | 95,4% | 95,1% |
| 264, scala fissa | 100% | 99,1% | 91,7% | 91,4% |
| 264, scala fissa + larghezza | 100% | 99,1% | 97,5% | 97,0% |
| 1280, ritaglio stretto | 100% | 100% | 82,1% | 82,6% |
| 1280, ritaglio stretto + larghezza | 100% | 100% | 97,3% | 97,4% |
| 1280, scala fissa | 100% | 100% | 93,9% | 94,1% |
| **1280, scala fissa + larghezza** | 100% | 99,9% | **98,7%** | **98,7%** |

Il ritaglio stretto viene ingrandito sempre alla stessa misura, quindi il vettore perde la
grandezza; con la scala fissa la grandezza resta nell'immagine, anche a 3 m. La larghezza misurata
con la profondità resta più precisa, e insieme danno il meglio. Nella prova "+ larghezza" il peso
del numero accanto al vettore è scelto a mano (0,5): più avanti lo impara l'attenzione.

**Oggetti visti in parte** (ritaglio stretto, 264, una parte tolta di lato): con metà oggetto il
colore resta giusto nel 100% dei casi e la forma nel 92%; con un quarto la forma scende al 71%.
Sotto metà oggetto lo si può confondere: lo risolvono memoria e movimento, cioè L e M.

### A. Attenzione

- **Fa:** sceglie su che cosa concentrarsi.
- **Entra:** gli embedding (chiavi) e la **domanda da D**, cioè l'uscita del
  ragionamento al passo prima. È un'attenzione a domanda: D dice che cosa cerca e gli
  oggetti rispondono secondo la somiglianza.
- **Esce:** un vettore attento. **Non ha memoria propria.**
- **Salienza innata:** resta una piccola parte dal basso per ciò che è nuovo o
  molto vicino. Senza, un essere concentrato non si accorgerebbe di un pericolo, e alla
  nascita D non ha ancora niente da chiedere.

### L. Memoria di lavoro (a breve termine)

È una delle due memorie a breve termine: l'altra, spaziale, è M.

- **Fa:** tiene gli ultimi eventi. I vecchi si sovrascrivono.
- **Entra:** vettore attento, senso della scena sfocato, luogo da M, azione eseguita, piacere e dolore.
- **Esce:** il contenuto recente, verso W, R e D (linea residuale).
- **Un evento:** la terna **oggetto, azione, esito**, scritta quando cambia qualcosa,
  non a ogni passo. Altrimenti si riempirebbe di passi quasi uguali.
- **Anello esterno:** impara che cosa scrivere e quando, e con quale regola cambiare.
  **Anello interno:** cambiano le attività e, nella seconda variante, le sinapsi.
- **Due varianti, sullo stesso esame:** una GRU, con la memoria nell'attività, e una rete con
  sinapsi plastiche veloci che decadono, con la memoria anche nei pesi. Non a impulsi: per tenere
  l'informazione nei pesi serve la plasticità, non gli impulsi, che costano di più.
- **Due filtri:** A decide che cosa arriva, in dettaglio; L decide se tenerlo. Il senso della scena
  arriva senza attenzione, ma sfocato.
- **Ricostruzione, in addestramento:** oltre a W, la memoria deve saper ricostruire ciò che ha visto
  qualche evento fa. Con la sola previsione terrebbe solo ciò che porta piacere o dolore. Il
  ricostruttore poi si butta. La sua uscita resta grezza: nessun numero ha un nome.
- **Da fare:** oggi riceve 7 numeri scritti a mano e un solo oggetto scelto dal mondo, cioè etichette
  pulite e attenzione finta; deve ricevere embedding veri, da una banca registrata nel 3D perché
  costano troppo per milioni di vite. Va provata anche a ogni passo, imparando da sola quando scrivere.
- **Che cosa tiene:** ciò che serve adesso, non il sapere. Una regola nuova nasce qui mentre la si
  scopre e resta qui mentre la si usa, ma da conservare spetta a R: se la regola torna giorni dopo, si
  ripesca da lì. Oggi L fa anche questo solo perché R non esiste, e la vita dura 60 eventi.
- **Fatta, per ora la GRU:** `memory.py`. Le sinapsi plastiche non hanno reso di più (tabella sotto),
  costano circa tre volte a passo, e dopo eventi inutili non ricordano meglio.

### W. Modello del mondo

**Oggi è solo un previsore di esito:** dato un oggetto e un'azione dice piacere, dolore o niente, col
dubbio. Non prevede il prossimo evento, quindi non si può ancora concatenare.

La sezione causale e quella della previsione sono **la stessa**: "A causa B" si impara
come "se succede A, faccio X, prevedo B".

- **Fa:** data la memoria e un'azione, prevede il prossimo evento e il suo esito. Con L è un
  **world model** come Dreamer (L lo stato, W la previsione, D chi agisce nell'immaginazione),
  ma a eventi, sugli oggetti e non sulle immagini, e deve cambiare idea in pochi eventi.
- **Incertezza:** dà anche quanto è sicura. "Non lo so" è diverso da "non succede niente", e
  guida l'esplorazione.
- **Entra:** memoria a breve, azione (eseguita o immaginata), bisogni, esito vero.
- **Esce:** previsione, ed **errore di previsione**.
- **Due usi:** con l'azione eseguita prevede e confronta. Con un'azione immaginata
  simula, senza che il corpo si muova. Così D chiede "e se facessi questo?".
- **Lavora a eventi**, non a passi: "premo X, si apre Q". Le azioni che riceve sono
  "vai lì e mordi", non comandi di sterzo. Il dettaglio del percorso resta a C.
- **Esiti nel tempo:** il dolore e il piacere possono variare e arrivare dopo l'azione.
  Per questo il modello legge più eventi insieme, non solo l'ultimo: una causa può
  essere a qualche evento di distanza.
- **Causa e coincidenza:** osservare da solo non basta. Serve provare, quindi il
  modello collabora con l'esplorazione: dove il suo errore è alto, conviene agire.
- **Anello esterno:** impara come si impara. **Anello interno:** ciò che vale in questo mondo
  si fissa nella memoria di L e, se la variante plastica rende, nelle sinapsi.

### R. Memoria a lungo termine

- **Fa:** conserva le situazioni che contano e quanto erano buone o pericolose, e i luoghi.
- **Un archivio solo, non due:** ogni ricordo ha i campi luogo, posizione, che cosa, esito, contesto e
  quando, e si cerca da uno qualsiasi: dal luogo a ciò che c'era, dalla fame al luogo. Un luogo
  visto molte volte si rinforza, uno visto una volta resta un episodio.
- **Entra:** uscita di W, memoria a breve (residuale) e **richiesta da D** ("che cosa portava a questo esito?").
- **Esce:** i ricordi simili alla situazione di adesso, verso D.
- **Scrive solo ciò che ha sorpreso**, guidata dall'errore di W.
- **Richiama per somiglianza:** una situazione fa riaffiorare quelle che le somigliano.
- **Contesti:** ciò che è pericoloso può cambiare. I ricordi restano legati al loro
  contesto, come i cassetti di K, per non cancellarsi a vicenda.

### D. Ragionamento

Il pensiero resta nella rete da un passo all'altro: non è un'informazione che passa ed
esce, come in un transformer. Tre parti:

| parte | compito | ritmo |
|---|---|---|
| **Valutazione** | "che cosa conviene adesso?": confronta bisogni, ricordi e situazione, e sceglie lo scopo | a eventi |
| **Livello alto** | stato ricorrente: tiene lo scopo e il passo del piano | lento |
| **Livello basso** | sceglie la prossima azione per arrivarci: l'intenzione verso C | veloce |

- **Nessun cancello separato.** È la rete a decidere quanto aggiornare il proprio pensiero. Deve
  imparare due cose: ignorare ciò che non conta quando gli ingressi cambiano ma lo scopo no, e
  cambiare passo quando lo scopo è raggiunto ma gli ingressi sono uguali. Non è garantito:
  i mondi di addestramento devono contenere distrazioni e compiti lunghi, e lo si misura.
- **I bisogni guidano la ricerca.** Con la fame la valutazione chiede a R che cosa dà energia e
  a A di cercarlo; da sazio smette di chiedere. Il valore di un ricordo dipende quindi dal
  bisogno di adesso: è ciò che permette la rivalutazione senza riprovare. Da R arriva anche che
  cosa fa male. Smette di cercare, non di vedere: un pericolo vicino passa comunque (A).
- **Esito: atteso contro accaduto.** Nessuno etichetta "fatto" o "fallito". Fatto: ciò che si
  voleva coincide con ciò che si sente (il bisogno è calato) o si vede (l'oggetto è preso).
  Fallito: ci si aspettava piacere e arriva dolore, lo scopo non si vede più, o lo si insegue
  senza avvicinarsi. Vengono da bisogni, percezione ed errore di W.
- **Più avanti:** interrompere un piano per un pericolo e riprenderlo dopo. Serve con i compiti
  a più passi.
- **Entra:** ricordi da R, memoria a breve, simulazioni da W, errore di W, bisogni, azione
  eseguita.
- **Esce:** domande verso A e R, azione voluta verso C, azioni immaginate verso W.
- **Non prevede.** Propone e valuta: la previsione è tutta in W (vedi "Come D usa W").
- **Esplorare:** quando l'incertezza è alta conviene scoprire. Il livello alto lo
  riceve dall'errore di W e dai bisogni.

### C. Motore (fatta)

- **Fa:** traduce un'intenzione in comandi per un corpo che non conosce.
- **Entra:** tre intenzioni, sempre uguali qualunque sia il corpo: velocità, rotazione, bocca.
  Più i sensi del corpo, gli stessi tre: che cosa ha fatto davvero.
- **Chi comanda non importa:** il motore riceve intenzioni dal comandante finto che legge la
  verità (`Perfect`) o da quello che ha solo gli occhi (`Seeing`), senza saperlo.
- **Esce:** K uscite, tante quante ne ha il corpo. Tra ingressi e uscite ci sono 8 neuroni
  nascosti (tanh).
- **Infanzia:** alla nascita la rete non sa niente e si tara per 60 s. Impara col gradiente,
  mentre vive: è l'unica eccezione alla regola "nessun gradiente mentre si vive", perché
  questa sezione impara solo il corpo, non il mondo. Ingresso il movimento sentito, bersaglio
  il comando dato. Nella prima metà i comandi sono a caso, nella seconda prova direzioni a
  caso, perché col solo caso certi corpi non raggiungono i loro estremi. Negli urti non
  impara. Il resto del cervello non interviene.
- **Separazione:** il cervello sopra produce solo intenzioni e non sa niente del corpo.
  Cambiare corpo o tarare tocca soltanto questa sezione.
- **Ritaratura:** si può fare in ogni momento, col pulsante "Tara i sensi". Parte anche da
  sola per 30 s, senza che nessuno lo avverta, se in 5 s la velocità sentita è meno del 30%
  di quella chiesta, se la rotazione va al contrario, o se per 1 s vuole mordere e la bocca
  non si apre. Gli
  esempi del corpo di prima si buttano, la rete resta.
- **Ancora scritto a mano:** il "quando mordere" e "vai lì" stanno nel comandante finto
  (per ora la logica del perfetto), che tiene il posto delle sezioni mancanti.
- **Limite:** la rete traduce un'intenzione nel comando di adesso, senza memoria. Zampe o
  braccia, che richiedono sequenze di movimenti, avranno bisogno di più.
- **Scorciatoie:** riflessi e guida verso l'oggetto passeranno da qui, senza attendere il
  ragionamento. Non ci sono ancora.

### M. Memoria spaziale (a breve termine)

Una rete che impara, separata da L: **non un grafo né una mappa scritta a mano.** Nei pesi fissi sta
il modo di costruire una mappa, nello stato la mappa di questa casa. Non esiste ancora.

- **Fa:** stima dove sono, ricorda dove sono gli oggetti appena visti, indica la direzione per
  arrivarci. Risponde a "dove?", a ogni passo; W risponde a "che cosa succede se?", a eventi.
- **Entra:** movimento sentito, senso della scena (profondità e colore su 16 colonne), oggetto
  attento con angolo e distanza, domanda da D, ricordo da R.
- **Esce:** posizione e dubbio, risposta alla domanda, direzione verso C, il luogo verso L e R.
- **Stato:** l'attività tiene dove sono adesso. Le sinapsi veloci, se rendono più della sola GRU,
  tengono che cosa c'è in ogni posto: lo decide l'esame.
- **Con R:** il ricordo del luogo si richiama quando lo si riconosce, e si scrive solo ciò che è
  cambiato. Non si scarica e ricarica una mappa intera.
- **Anello esterno:** ogni vita una casa nuova: esplora, poi "dov'era quella pila?", "torna al punto
  di partenza", "vai in quella stanza". Il gradiente passa tra le vite. **Anello interno:** cambiano
  attività e sinapsi veloci. La posizione vera serve solo a correggere in addestramento.
- **Mondo per addestrarla:** stanze senza fisica, con la stessa impronta della vista: con le
  telecamere non si avrebbero abbastanza case. Si controlla poi nel 3D, sulle mappe vere.
- **Esame:** case mai viste e labirinto tenuto fuori. Errore di posizione a 60 s e a 5 minuti, con e
  senza vista; si ritrova dopo uno spostamento nascosto; indica oggetti fuori vista; pile al minuto.
  La geometria fatta a mano non entra nel cervello: è il riferimento da battere.

### B. Bisogni

- **Fa:** dice che cosa manca al corpo.
- **Entra:** energia e stati del corpo. **Esce:** verso W e D.
- Un bisogno cambia il valore delle cose e il perché di ogni azione. Senza, l'essere non
  ha motivi. Se un bisogno cambia, cambia anche la domanda di D verso A: la fame
  agisce sull'attenzione passando dal ragionamento.

## L'azione: dove va

Esiste una copia dell'azione, come nel cervello vero (la copia efferente). La copia
giusta è l'azione **eseguita**, perché un riflesso può aver cambiato quella voluta. Non sono mai
i comandi ai motori, che cambiano col corpo: sono ciò che il corpo sente di aver fatto
(velocità, rotazione, bocca), nello stesso spazio dell'intenzione, e a livello di evento
"mi avvicino", "mordo", "ignoro". Cambiando corpo si ritara solo C.

| sezione | riceve l'azione? | perché |
|---|---|---|
| L. Memoria a breve | **sì** | serve a ricordare "che cosa ho fatto" |
| W. Modello del mondo | **sì** | senza azione non può prevedere né capire le cause |
| D. Ragionamento | **sì** | sa da che punto riparte e se il piano sta andando avanti |
| R. Memoria a lungo | indirettamente | la riceve con gli eventi da L e W |
| A. Attenzione | indirettamente | la riceve con la domanda da D |
| P. Percezione | no, per ora | servirà per togliere dalla vista il proprio movimento |

## Dove avviene la previsione

Nella sezione W. Ogni passo reale segue quest'ordine:

```
1. D sceglie un'azione, e può prima provarne altre immaginandole con W.
2. W prevede evento ed esito dell'azione scelta.
3. C esegue. Il mondo risponde.
4. W confronta previsione e realtà: l'errore.
5. L'errore decide che cosa scrivere in R e sposta l'attenzione di D.
```

Più avanti servirà una previsione più veloce del corpo, accanto a C, come il cervelletto.

## Come D usa W

Il modello è **uno solo**. I due livelli gli chiedono cose diverse:

| chi | che cosa fa | direzione |
|---|---|---|
| D alto | sceglie l'esito voluto; chiede a R da dove veniva quell'esito; ne ricava sotto-scopi | all'indietro, dallo scopo |
| R | richiama per esito: "il cibo stava in Q", "Q si apriva dopo X" | ricordo |
| D basso | propone un'azione candidata | avanti |
| W | risponde con evento ed esito previsti | avanti |

Pensare è una catena. L'uscita di W rientra in W al posto della memoria:

```
scopo: mangiare. Da R: il cibo stava in Q.   -> candidato: premo X
W(adesso, premo X)      -> si apre Q, esito 0
W(Q aperta, mordo Q)    -> pile, esito buono: coincide con lo scopo
Decisione: premo X, che da solo vale 0.
```

Una **causa** non si prevede: si cerca così, risalendo con R e verificando in avanti con W.
Non serve un modello solo per le cause.
Se R non ha niente, il candidato viene dall'esplorazione.

Regole dell'immaginazione:

- Non scrive in L e in R, non muove le sinapsi plastiche, non produce errore.
- Lo stato interno di W usato per immaginare è separato da quello reale.
- Il mondo non aspetta: il tempo per pensare è un limite, pochi tentativi per evento.
- **Immaginare è un'azione di D basso**, come in Jensen e colleghi (2024): è la rete a
  decidere quando pensare e quando agire. La catena è fissa, cosa provare si impara.

## La prova

Il criterio resta quello di docs/SPEC_PIANIFICAZIONE.md: la **rivalutazione alla prima
prova** nell'arena delle casse. Per gli anelli si aggiunge un criterio nuovo: il
cervello va provato in **mondi mai visti** durante l'addestramento. Un cervello che riesce
solo su quelli visti ha memorizzato, non ha imparato a imparare.

Lesioni: ogni sezione deve poter essere tolta, misurando che cosa si perde.

Gli esami delle sezioni sono in "Addestrare una sezione alla volta". Il cervello a sezioni
resta il riferimento: nell'arena 3D lo si collega con il detector finto.

## Fasi di lavoro

Si costruiscono prima le sezioni che D dovrà usare, e D per ultima: così i suoi sottomoduli
si progettano sulle uscite vere, non su versioni perfette che non daranno mai gli stessi errori.
Le sezioni L, W e R hanno esami propri e restano utili con qualunque ragionamento.

| fase | lavoro | misura per passare alla fase dopo |
|---|---|---|
| 0 | mondo 3D: **fatti** MuJoCo, arena delle pile, agenti di controllo, interfaccia grafica di base; **mancano** arena delle casse, mondi casuali, immagini su più processi | stessa vita con lo stesso seme identica (**verificato** su questa macchina); velocità nota, con almeno 100 esseri (**5,5 volte il tempo reale, senza immagini**) |
| 1 | C, con l'infanzia al posto dell'imitazione (**fatta**) | tutti i corpi di prova imparati da zero tra i muri, e ritaratura da sola |
| 2 | P: percezione dalla telecamera (**fatta**, con due reti da confrontare) | trova gli oggetti con l'interfaccia del detector finto |
| 3 | mondo a eventi, L e W (**livelli 1 e 2 fatti**, mancano ritardi, sequenze, ricostruzione, embedding veri e il pannello) | superano l'esame |
| 4 | banca di embedding veri, poi M: memoria spaziale | dopo aver girato indica dov'erano gli oggetti; il vedente trova più pile al minuto |
| 5 | R | supera l'esame; ritorno rapido a un regime già vissuto |
| 6 | **ragionatore finto**, scritto a mano: propone, chiede a W, cerca in R, sceglie | rivalutazione superata in mondi mai visti. **Se fallisce si correggono L, W e R prima di costruire D** |
| 7 | A | supera l'esame; il ragionatore la usa per trovare ciò che cerca |
| 8 | D basso, con la catena dell'immaginazione | sequenze di sotto-scopi che reggono ai distrattori |
| 9 | D alto | rivalutazione alla prima prova; togliendo la catena, peggiora |
| 10 | ritocco finale di tutto insieme, percezione compresa | tutti gli esami ripetuti e superati |
| 11 | evoluzione sopra il gradiente | tara le costanti e si misura che cosa cambia |

La fase 6 dice se le sezioni danno a D ciò che le serve, a un costo piccolo. Non dice se una D
addestrata imparerà a usarle: quel rischio resta fino alle fasi 8 e 9, ed è accettato.

## Criticità

- **Come passa il gradiente in una vita.** Una vita di 400 s sono 8000 passi: il gradiente
  su tutta la vita è costoso e instabile. Addestrando le sezioni una alla volta il problema
  resta quasi solo per D, che lavora a eventi con il resto bloccato. Si misura nelle fasi 8 e 9.
- **Potenza di calcolo.** Molte vite per ogni aggiornamento, e ogni immagine costa. Su questo
  portatile bastano poche centinaia di esseri. Il detector finto evita gran parte del
  costo, e la versione batch di MuJoCo per GPU, da verificare, richiederebbe altra macchina.
- **Riproducibilità.** Con lo stesso seme due vite sono identiche bit per bit su questa
  macchina (il test non usa immagini). Tra macchine, versioni e GPU non è stato provato: la
  versione è fissata in `uv.lock`, e le immagini possono cambiare tra GPU diverse.
- **Scatola nera.** Una rete addestrata può funzionare senza che si capisca come. Esami e
  lesioni la riducono ma non la eliminano: gli esami dicono che cosa fa una sezione, non come.
- **La catena di immaginazione moltiplica il costo.** Ogni tentativo aggiunge passi interni
  attraverso cui passa il gradiente. Gli errori di W si sommano a ogni anello: la catena
  sarà utile solo per pochi eventi.
- **D arriva per ultima.** Se non impara a usare le altre sezioni lo si scopre tardi. Il
  ragionatore finto della fase 6 riduce il rischio: controlla che le sezioni bastino, non che D impari.
- **Mondi troppo simili.** Il cervello impara i mondi di addestramento, non a imparare. La
  loro varietà è una scelta di progetto importante quanto l'architettura.
- **Il principio "si fissa la struttura".** Va riletto: si fissano le sezioni e chi parla con
  chi; le regole dentro le sezioni non sono più scelte da noi ma addestrate.
- **La sezione D resta la parte meno definita.** Una rete ricorrente mostra una forma di
  ragionamento solo se il compito la richiede, e non è garantito nemmeno allora.

## Che cosa si riusa

| di oggi | diventa |
|---|---|
| idea di `objects_of` | detector finto, letto dal simulatore 3D |
| logica del perfetto | comandante finto sopra la sezione C, finché mancano le altre |
| idea di O (fame sul valore) | bisogni e valore in W e D |
| idea di K (cassetti) | contesti di R |
| JAX | anello esterno |
| `evonets/viz/` | interfaccia grafica |
| banco di prova e lesioni | stessa metodologia |

Nel codice: `evonets/arenas/mondo3d/`, `evonets/brains/mondo3d/` (controlli),
`evonets/brains/ragionante/` (il cervello, che non importa niente da quello a sezioni),
`evonets/viz/mondo3d/`. Il cervello a sezioni e l'arena delle pile non cambiano.

## Riferimenti da tenere presenti

- Wang e colleghi (2018): corteccia prefrontale e meta-apprendimento per rinforzo.
- Duan e colleghi (2016), RL²: imparare a imparare con una rete ricorrente.
- Miconi e colleghi (2018): plasticità addestrata con il gradiente.
- Sutton (1991); Hafner e colleghi (Dreamer): pianificare con un modello imparato.
- Jensen, Hennequin e Mattar (2024): immaginare come azione in una rete ricorrente.
- Ha e Schmidhuber (2018), World Models: vista e modello addestrati a parte.
- Tolman (1948); Daw e colleghi (2011): la prova della rivalutazione.

## Decisioni aperte

1. **Memoria a breve:** si confrontano GRU e sinapsi plastiche. Una fila di posti, solo se falliscono.
2. **In vita cambiano solo le attività, o anche sinapsi plastiche?** Si decide con quel confronto.
3. **Che cosa cambia tra una vita e l'altra:** deciso: i quattro livelli del mondo a eventi.
4. **Tempo:** a eventi per 4, 5 e 6, a passi per 1, M e 7. Per 3 a eventi, finché la prova a ogni
   passo non dice se sa decidere da sola quando scrivere.
5. **Evoluzione:** resta sopra il gradiente fin dall'inizio, o entra solo alla fine?
6. **Valore:** deciso in parte. La scelta sta nella Valutazione di 6, ma che cosa ricevere da 4 e
   R è ancora da definire.
7. **Immaginazione:** quanti tentativi per evento, e quanto lunga la catena?

## Prossimi passi

Da fare in quest'ordine. Prima di ogni addestramento si spiega la struttura esatta.

### 1. Banca di embedding (fatta)

`bank.py`, `embeddings.npz`: 6.200 righe, 31 tipi di oggetto, 200 viste per tipo, 14 minuti su CPU.
La PCA tiene il 99,2% della varianza. Il vicino più simile ha lo stesso colore nel 100% dei casi, la
stessa forma nel 99,4%, ma la stessa grandezza solo nel 68%. Due cause: il ritaglio si ridimensiona,
e le sfere piccole e grandi erano identiche (errore, corretto in `bank.py`).

**Da rifare** con il flusso scelto di P: ritaglio a scala fissa, telecamera 1280×720, sfere corrette,
e la larghezza in cm salvata in ogni riga.

1. Si registrano nel 3D gli oggetti, da molti punti di vista, distanze e luci: pile rosse e gialle,
   piccole e grandi, esseri, più forme e colori che le pile non hanno, come il blu.
2. Per ogni oggetto la maschera vera del simulatore sceglie i pixel, DINOv2 più il colore ne fa il
   vettore da 387 numeri.
3. Una PCA, un calcolo e non una rete, li riduce a 64. Riduzione e vettori si salvano in un file.
4. Ogni riga ricorda l'oggetto, l'angolo e la distanza da cui è stato visto.

Esame: due viste dello stesso oggetto più simili di due oggetti diversi, anche dopo la riduzione.

### 2. Mondo di case semplificato

- JAX, senza fisica: 2 o 3 stanze per lato, porte e colori a caso. Il labirinto resta fuori.
- Una vita dura 2 minuti, con un passo ogni 0,2 s: 600 passi.
- Cammina un agente scritto a mano, che passa dalle porte e parte da una stanza a caso.
- Controlli: stessa vista del 3D, nessuna informazione nascosta negli ingressi.

### 3. Sezione M

Ingressi a ogni passo, 266 numeri:

| ingresso | numeri | da dove, in addestramento |
|---|---|---|
| movimento sentito | 2 | calcolato, con rumore |
| scena sfocata | 64 | profondità e colore su 16 colonne, con rumore |
| vettore attento | 69 | dalla banca: 64 più angolo (2), vicinanza, grandezza e "c'è" |
| domanda | 67 | tipo (3) più embedding dalla banca (64); generata dal mondo |
| ricordo da R | 64 | zeri finché R non esiste |

L'attenzione finta sceglie quasi sempre l'oggetto più vicino, a volte un altro, a volte nessuno.

Interno: nucleo ricorrente, codice di posizione, sinapsi veloci che legano posizione e contenuto.
Uscite: posizione stimata e dubbio, risposta a "dov'è?", direzione, scena prevista.

Due nuclei, con pesi quasi uguali:

| | GRU | rete a perdita |
|---|---|---|
| neuroni | 256 | circa 440, ognuno con la sua velocità appresa |
| velocità di apprendimento | 0,002 | 0,001 |
| per non perdere l'informazione | cancelli | collegamenti iniziali scelti, rumore sull'attività, penalità se l'attività esplode |

Addestramento uguale per entrambi, 3 semi per nucleo: ogni passo 64 case nuove, gradiente su
spezzoni di 150 passi, stima di 1-5 milioni di vite. Errori sommati: prossima vista, "dov'era
quell'oggetto?", "torna alla partenza", "da che parte vado?", dubbio. La posizione vera corregge le
domande, non entra mai.

Esame in case mai viste, col calcolo geometrico come riferimento: errore a 60 s e a 5 minuti, con e
senza vista; spostamento nascosto; oggetti fuori vista; direzione, anche nel labirinto; sinapsi veloci
tagliate. Se la rete a perdita è entro il 10% dalla GRU si tiene lei.

### 4. Prova nel 3D

Camminate registrate sulle mappe vere, con le immagini vere. Se regge si collega al vedente, e si
misurano le pile al minuto con le stanze: oggi da 0 a 0,25.

