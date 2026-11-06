# Cervello ragionante (progetto)

Stato: progetto. Del codice esistono il mondo 3D con le pile, i due agenti di
controllo e, del cervello, la sezione 7 (`evonets/brains/ragionante/`); il resto no. Un **nuovo tipo di cervello**, accanto a quelli di oggi: il cervello a sezioni (docs/SPEC_CERVELLO_SEZIONI.md) resta il riferimento da
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

```
                              MONDO E CORPO
                  ┌──────────────┬───────────────┐
                  │ telecamera   │ piacere,      │ energia, fame,
                  │              │ dolore        │ stanchezza
                  ▼              │               ▼
      ┌──────────────────────┐   │    ┌────────────────────┐
      │ 1. PERCEZIONE        │   │    │ B. BISOGNI         │
      │ segmentazione e      │   │    │ quanto mi serve    │
      │ detection            │   │    │ che cosa, adesso   │
      └──────────┬───────────┘   │    └──────────┬─────────┘
                 │ embedding degli oggetti       │ (va a 4, 6)
                 ▼               │               │
      ┌──────────────────────┐   │               │
      │ 2. ATTENZIONE        │◄──┼───────────────┼──── domanda da 6
      │ salienza innata +    │   │               │     (passo prima)
      │ domanda di 6         │   │               │
      └──────────┬───────────┘   │               │
                 │ vettore attento               │
                 ▼               ▼               │
      ┌──────────────────────────────┐           │
      │ 3. MEMORIA A BREVE TERMINE   │◄─────┐    │
      │ eventi: oggetto, azione,     │      │    │
      │ esito. I vecchi si           │      │    │
      │ sovrascrivono                │      │    │
      └───┬──────────────────┬───────┘      │    │
          │                  │ residuale    │    │
          ▼                  │              │    │
      ┌──────────────────┐   │              │    │
      │ 4. MODELLO DEL   │◄──┼──── azione ──┤    │
      │ MONDO            │◄──┼── azione ────┼────┼── immaginata, da 6
      │ cosa causa cosa, │   │   eseguita   │    │
      │ che cosa seguirà │◄──┼──────────────┼────┘
      └───┬──────────┬───┘   │              │
          │ errore   │       │              │
          ▼          │       ▼              │
      ┌──────────────┼───────────────────┐  │
      │ 5. MEMORIA A LUNGO TERMINE       │  │
      │ situazioni di piacere e pericolo │  │
      │ richiamate per somiglianza o     │  │
      │ su richiesta di 6                │  │
      └──────────┬───┼───────────────────┘  │
                 │   │ simulazioni          │
                 ▼   ▼                      │
      ┌──────────────────────────────────┐  │
      │ 6. RAGIONAMENTO                  │  │
      │ valutazione: che cosa conviene   │  │
      │ alto: scopo, piano (lento)       │  │
      │ basso: prossima azione (veloce)  │◄─┼── azione eseguita
      │ memoria del pensiero (ricorrente)│  │
      └──────────┬───────────────────────┘  │
                 │ azione voluta            │
                 ▼                          │
      ┌──────────────────────────────────┐  │
      │ 7. MOTORE                        │──┘ azione eseguita
      │ traduce l'azione in comandi;     │
      │ riflessi e guida verso l'oggetto │
      └──────────┬───────────────────────┘
                 ▼
            MONDO E CORPO
```

Anche la memoria a breve termine entra direttamente in 6: è la linea residuale. Inoltre 6
interroga 5 con l'esito che vuole.

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
vive. L'eccezione è il motore (sezione 7), che così impara soltanto il proprio corpo.
Cambia che il suo cervello di nascita è prodotto anche dal gradiente.

## Addestrare una sezione alla volta

Se tutte le sezioni imparano insieme, niente garantisce che ognuna faccia ciò che dice il
suo nome. Quindi ogni sezione si addestra da sola su un compito suo, viene esaminata e poi
si blocca. La sezione dopo impara con quelle prima già bloccate.

| sezione | esame, sempre in mondi mai visti | compagni |
|---|---|---|
| 7 | si tara da sola in corpi sconosciuti, raggiunge l'oggetto indicato e lo morde (**superato**) | comandante finto |
| 3 e 4 | dopo pochi eventi prevede il prossimo evento e il suo esito | in coppia: che cosa ricordare dipende da che cosa serve prevedere |
| M | dopo aver girato indica dov'erano gli oggetti non più inquadrati | percezione e movimento sentito |
| 5 | richiama l'esito giusto dopo molti eventi diversi, anche partendo dall'esito; regge ai cambi di regime | sequenze generate da noi |
| 2 | con una domanda data da noi sceglie l'oggetto giusto; un pericolo vicino passa comunque | domande date da noi |
| 6 basso | raggiunge un sotto-scopo dato | 2, 3, 4, 5, 7 bloccate |
| 6 alto | **rivalutazione alla prima prova** | tutto bloccato, anche 6 basso |
| 1 | trova gli oggetti e dà la stessa interfaccia del detector finto (**superato**, con le due reti: vedi "1. Percezione") | etichette del simulatore |

Regole:

- **Imparare, non memorizzare.** L'esame di 3, 4 e 5 è "impara in fretta un mondo mai visto",
  non "conosci questo mondo". Ogni sezione si allena su molti mondi casuali, con i due anelli.
- **Chi agisce mentre 4 impara.** A 4 non servono azioni giuste, servono azioni varie con il
  loro esito vero. Agiscono gli agenti di controllo, con errori aggiunti apposta: un agente
  che non morde mai una pila cattiva non insegnerebbe il dolore, uno che vaga a caso non
  aprirebbe mai una cassa. Quando esiste un cervello, si alterna la raccolta con lui.
- **Che cosa spinge a prevedere.** Il mondo dà la risposta un istante dopo, senza maestro: si
  misura l'errore tra previsione e realtà lungo tutta la vita, e il gradiente dell'anello
  esterno lo riduce. La catena si allena rimettendo in 4 le sue stesse previsioni.
- **Ritocco finale.** Una sezione bloccata può aver scartato un dettaglio che serviva più
  avanti. Alla fine si addestra tutto insieme, a passo piccolo, e si **ripetono tutti gli
  esami**: una sezione che non li supera più ha perso il suo significato.
- **Che cosa non garantisce.** L'esame dice che 4 sa prevedere, non che 6 la usi. Lo dicono
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
sposta, che nascono lontane dagli esseri e compaiono e spariscono in 0,6 s. Corpo su due ruote con telecamera a 120°, immagine 264×264 più profondità in cm. La telecamera
sta dentro il corpo, 8 cm davanti al centro: sul muso, a contatto, finirebbe dentro ciò che tocca. Il
telaio si inclina al massimo di 6° perché ribaltandosi resterebbe bloccato. Poi l'arena
delle casse, con casse e interruttori come forme colorate. Le zampe, se serviranno, arrivano dopo.

**Agenti di controllo** (`evonets/brains/mondo3d/`), non cervelli: un cervello si misura tra i due.

| agente | che cosa sa | pile buone/min | cattive/min | scelta |
|---|---|---|---|---|
| **perfetto** (soffitto) | legge la verità dal simulatore; non usa la telecamera | 11,7 | 0 | 1,00 |
| **base** (pavimento) | solo la profondità: vaga e morde ciò che ha davanti | 0,06 | 0,06 | 0,50 |

Misure con 1 essere per recinto: perfetto su 4 recinti per 20 minuti, base su 4 recinti per 12
minuti (solo 6 pile: la scelta è indicativa). Il cervello ragionante con la sola sezione 7, 3 recinti per 5
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

Col cervello ragionante la schermata mostra anche la sezione 7: neuroni, sinapsi, cablaggio
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

Le sezioni che lavorano a eventi (3, 4, 5, 6) hanno bisogno di molte vite, non di fisica: nel 3D
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

### 1. Percezione

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
- **Reti pronte** (`ReadyDetector`, opzionali, gruppo `percezione`): la profondità indica che
  cosa sporge dal pavimento, **SlimSAM** ne fa la maschera, **DINOv2 small** più il colore
  fanno il vettore, `ObjectMemory` lo conserva anche su disco e dice se l'oggetto è già noto. Il
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

### 2. Attenzione

- **Fa:** sceglie su che cosa concentrarsi.
- **Entra:** gli embedding (chiavi) e la **domanda da 6**, cioè l'uscita del
  ragionamento al passo prima. È un'attenzione a domanda: 6 dice che cosa cerca e gli
  oggetti rispondono secondo la somiglianza.
- **Esce:** un vettore attento. **Non ha memoria propria.**
- **Salienza innata:** resta una piccola parte dal basso per ciò che è nuovo o
  molto vicino. Senza, un essere concentrato non si accorgerebbe di un pericolo, e alla
  nascita 6 non ha ancora niente da chiedere.

### 3. Memoria a breve termine

- **Fa:** tiene gli ultimi eventi. I vecchi si sovrascrivono.
- **Entra:** vettore attento, azione eseguita, piacere e dolore.
- **Esce:** il contenuto recente, verso 4, 5 e 6 (linea residuale).
- **Un evento:** la terna **oggetto, azione, esito**, scritta quando cambia qualcosa,
  non a ogni passo. Altrimenti si riempirebbe di passi quasi uguali.
- **Anello esterno:** impara che cosa scrivere e quando, e con quale regola cambiare.
  **Anello interno:** cambiano le attività e, nella seconda variante, le sinapsi.
- **Due varianti, sullo stesso esame:** una GRU, con la memoria nell'attività, e una rete con
  sinapsi plastiche veloci che decadono, con la memoria anche nei pesi. Non a impulsi: per tenere
  l'informazione nei pesi serve la plasticità, non gli impulsi, che costano di più.
- **Fatta, per ora la GRU:** `memory.py`. Le sinapsi plastiche non hanno reso di più (tabella sotto),
  costano circa tre volte a passo, e dopo eventi inutili non ricordano meglio.

### 4. Modello del mondo

La sezione causale e quella della previsione sono **la stessa**: "A causa B" si impara
come "se succede A, faccio X, prevedo B".

- **Fa:** data la memoria e un'azione, prevede il prossimo evento e il suo esito. Con 3 è un
  **world model** come Dreamer (3 lo stato, 4 la previsione, 6 chi agisce nell'immaginazione),
  ma a eventi, sugli oggetti e non sulle immagini, e deve cambiare idea in pochi eventi.
- **Incertezza:** dà anche quanto è sicura. "Non lo so" è diverso da "non succede niente", e
  guida l'esplorazione.
- **Entra:** memoria a breve, azione (eseguita o immaginata), bisogni, esito vero.
- **Esce:** previsione, ed **errore di previsione**.
- **Due usi:** con l'azione eseguita prevede e confronta. Con un'azione immaginata
  simula, senza che il corpo si muova. Così 6 chiede "e se facessi questo?".
- **Lavora a eventi**, non a passi: "premo X, si apre Q". Le azioni che riceve sono
  "vai lì e mordi", non comandi di sterzo. Il dettaglio del percorso resta a 7.
- **Esiti nel tempo:** il dolore e il piacere possono variare e arrivare dopo l'azione.
  Per questo il modello legge più eventi insieme, non solo l'ultimo: una causa può
  essere a qualche evento di distanza.
- **Causa e coincidenza:** osservare da solo non basta. Serve provare, quindi il
  modello collabora con l'esplorazione: dove il suo errore è alto, conviene agire.
- **Anello esterno:** impara come si impara. **Anello interno:** ciò che vale in questo mondo
  si fissa nella memoria di 3 e, se la variante plastica rende, nelle sinapsi.

### 5. Memoria a lungo termine

- **Fa:** conserva le situazioni che contano e quanto erano buone o pericolose.
- **Entra:** uscita di 4, memoria a breve (residuale) e **richiesta da 6** ("che cosa portava a questo esito?").
- **Esce:** i ricordi simili alla situazione di adesso, verso 6.
- **Scrive solo ciò che ha sorpreso**, guidata dall'errore di 4.
- **Richiama per somiglianza:** una situazione fa riaffiorare quelle che le somigliano.
- **Contesti:** ciò che è pericoloso può cambiare. I ricordi restano legati al loro
  contesto, come i cassetti di K, per non cancellarsi a vicenda.

### 6. Ragionamento

Il pensiero resta nella rete da un passo all'altro: non è un'informazione che passa ed
esce, come in un transformer. Tre parti:

| parte | compito | ritmo |
|---|---|---|
| **Valutazione** | "che cosa conviene adesso?": confronta bisogni, ricordi e situazione, e sceglie lo scopo | a eventi |
| **Livello alto** | stato ricorrente: tiene lo scopo e il passo del piano | lento |
| **Livello basso** | sceglie la prossima azione per arrivarci: l'intenzione verso 7 | veloce |

- **Nessun cancello separato.** È la rete a decidere quanto aggiornare il proprio pensiero. Deve
  imparare due cose: ignorare ciò che non conta quando gli ingressi cambiano ma lo scopo no, e
  cambiare passo quando lo scopo è raggiunto ma gli ingressi sono uguali. Non è garantito:
  i mondi di addestramento devono contenere distrazioni e compiti lunghi, e lo si misura.
- **I bisogni guidano la ricerca.** Con la fame la valutazione chiede a 5 che cosa dà energia e
  a 2 di cercarlo; da sazio smette di chiedere. Il valore di un ricordo dipende quindi dal
  bisogno di adesso: è ciò che permette la rivalutazione senza riprovare. Da 5 arriva anche che
  cosa fa male. Smette di cercare, non di vedere: un pericolo vicino passa comunque (2).
- **Esito: atteso contro accaduto.** Nessuno etichetta "fatto" o "fallito". Fatto: ciò che si
  voleva coincide con ciò che si sente (il bisogno è calato) o si vede (l'oggetto è preso).
  Fallito: ci si aspettava piacere e arriva dolore, lo scopo non si vede più, o lo si insegue
  senza avvicinarsi. Vengono da bisogni, percezione ed errore di 4.
- **Più avanti:** interrompere un piano per un pericolo e riprenderlo dopo. Serve con i compiti
  a più passi.
- **Entra:** ricordi da 5, memoria a breve, simulazioni da 4, errore di 4, bisogni, azione
  eseguita.
- **Esce:** domande verso 2 e 5, azione voluta verso 7, azioni immaginate verso 4.
- **Non prevede.** Propone e valuta: la previsione è tutta in 4 (vedi "Come 6 usa 4").
- **Esplorare:** quando l'incertezza è alta conviene scoprire. Il livello alto lo
  riceve dall'errore di 4 e dai bisogni.

### 7. Motore (fatta)

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

### M. Mappa

- **Fa:** tiene un'idea dello spazio: dove sono io, dove sono le cose, anche fuori dalla vista.
- **Non è 4:** 4 risponde a "che cosa succede se faccio questo?", a eventi; la mappa a "dove
  sono le cose?", a ogni passo.
- **Entra:** il movimento sentito (dove sono io) e angolo e distanza da 1 (dove sono gli oggetti).
- **Esce:** gli oggetti con la loro posizione, verso 4 e 6: "vai lì" vale anche per ciò che non
  è inquadrato.
- **Prima versione:** geometria, senza rete. L'errore si accumula col movimento: su un robot vero
  andrà corretta con ciò che si vede.

### B. Bisogni

- **Fa:** dice che cosa manca al corpo.
- **Entra:** energia e stati del corpo. **Esce:** verso 4 e 6.
- Un bisogno cambia il valore delle cose e il perché di ogni azione. Senza, l'essere non
  ha motivi. Se un bisogno cambia, cambia anche la domanda di 6 verso 2: la fame
  agisce sull'attenzione passando dal ragionamento.

## L'azione: dove va

Esiste una copia dell'azione, come nel cervello vero (la copia efferente). La copia
giusta è l'azione **eseguita**, perché un riflesso può aver cambiato quella voluta. Non sono mai
i comandi ai motori, che cambiano col corpo: sono ciò che il corpo sente di aver fatto
(velocità, rotazione, bocca), nello stesso spazio dell'intenzione, e a livello di evento
"mi avvicino", "mordo", "ignoro". Cambiando corpo si ritara solo 7.

| sezione | riceve l'azione? | perché |
|---|---|---|
| 3. Memoria a breve | **sì** | serve a ricordare "che cosa ho fatto" |
| 4. Modello del mondo | **sì** | senza azione non può prevedere né capire le cause |
| 6. Ragionamento | **sì** | sa da che punto riparte e se il piano sta andando avanti |
| 5. Memoria a lungo | indirettamente | la riceve con gli eventi da 3 e 4 |
| 2. Attenzione | indirettamente | la riceve con la domanda da 6 |
| 1. Percezione | no, per ora | servirà per togliere dalla vista il proprio movimento |

## Dove avviene la previsione

Nella sezione 4. Ogni passo reale segue quest'ordine:

```
1. 6 sceglie un'azione, e può prima provarne altre immaginandole con 4.
2. 4 prevede evento ed esito dell'azione scelta.
3. 7 esegue. Il mondo risponde.
4. 4 confronta previsione e realtà: l'errore.
5. L'errore decide che cosa scrivere in 5 e sposta l'attenzione di 6.
```

Più avanti servirà una previsione più veloce del corpo, accanto a 7, come il cervelletto.

## Come 6 usa 4

Il modello è **uno solo**. I due livelli gli chiedono cose diverse:

| chi | che cosa fa | direzione |
|---|---|---|
| 6 alto | sceglie l'esito voluto; chiede a 5 da dove veniva quell'esito; ne ricava sotto-scopi | all'indietro, dallo scopo |
| 5 | richiama per esito: "il cibo stava in Q", "Q si apriva dopo X" | ricordo |
| 6 basso | propone un'azione candidata | avanti |
| 4 | risponde con evento ed esito previsti | avanti |

Pensare è una catena. L'uscita di 4 rientra in 4 al posto della memoria:

```
scopo: mangiare. Da 5: il cibo stava in Q.   -> candidato: premo X
4(adesso, premo X)      -> si apre Q, esito 0
4(Q aperta, mordo Q)    -> pile, esito buono: coincide con lo scopo
Decisione: premo X, che da solo vale 0.
```

Una **causa** non si prevede: si cerca così, risalendo con 5 e verificando in avanti con 4.
Non serve un modello solo per le cause.
Se 5 non ha niente, il candidato viene dall'esplorazione.

Regole dell'immaginazione:

- Non scrive in 3 e in 5, non muove le sinapsi plastiche, non produce errore.
- Lo stato interno di 4 usato per immaginare è separato da quello reale.
- Il mondo non aspetta: il tempo per pensare è un limite, pochi tentativi per evento.
- **Immaginare è un'azione di 6 basso**, come in Jensen e colleghi (2024): è la rete a
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

Si costruiscono prima le sezioni che 6 dovrà usare, e 6 per ultima: così i suoi sottomoduli
si progettano sulle uscite vere, non su versioni perfette che non daranno mai gli stessi errori.
Le sezioni 3, 4 e 5 hanno esami propri e restano utili con qualunque ragionamento.

| fase | lavoro | misura per passare alla fase dopo |
|---|---|---|
| 0 | mondo 3D: **fatti** MuJoCo, arena delle pile, agenti di controllo, interfaccia grafica di base; **mancano** arena delle casse, mondi casuali, immagini su più processi | stessa vita con lo stesso seme identica (**verificato** su questa macchina); velocità nota, con almeno 100 esseri (**5,5 volte il tempo reale, senza immagini**) |
| 1 | 7, con l'infanzia al posto dell'imitazione (**fatta**) | tutti i corpi di prova imparati da zero tra i muri, e ritaratura da sola |
| 2 | 1: percezione dalla telecamera (**fatta**, con due reti da confrontare) | trova gli oggetti con l'interfaccia del detector finto |
| 3 | mondo a eventi, 3 e 4 (**livelli 1 e 2 fatti**, mancano ritardi e sequenze, e il pannello) | superano l'esame |
| 4 | M: mappa | dopo aver girato indica dov'erano gli oggetti; il vedente trova più pile al minuto |
| 5 | 5 | supera l'esame; ritorno rapido a un regime già vissuto |
| 6 | **ragionatore finto**, scritto a mano: propone, chiede a 4, cerca in 5, sceglie | rivalutazione superata in mondi mai visti. **Se fallisce si correggono 3, 4 e 5 prima di costruire 6** |
| 7 | 2 | supera l'esame; il ragionatore la usa per trovare ciò che cerca |
| 8 | 6 basso, con la catena dell'immaginazione | sequenze di sotto-scopi che reggono ai distrattori |
| 9 | 6 alto | rivalutazione alla prima prova; togliendo la catena, peggiora |
| 10 | ritocco finale di tutto insieme, percezione compresa | tutti gli esami ripetuti e superati |
| 11 | evoluzione sopra il gradiente | tara le costanti e si misura che cosa cambia |

La fase 6 dice se le sezioni danno a 6 ciò che le serve, a un costo piccolo. Non dice se una 6
addestrata imparerà a usarle: quel rischio resta fino alle fasi 8 e 9, ed è accettato.

## Criticità

- **Come passa il gradiente in una vita.** Una vita di 400 s sono 8000 passi: il gradiente
  su tutta la vita è costoso e instabile. Addestrando le sezioni una alla volta il problema
  resta quasi solo per 6, che lavora a eventi con il resto bloccato. Si misura nelle fasi 8 e 9.
- **Potenza di calcolo.** Molte vite per ogni aggiornamento, e ogni immagine costa. Su questo
  portatile bastano poche centinaia di esseri. Il detector finto evita gran parte del
  costo, e la versione batch di MuJoCo per GPU, da verificare, richiederebbe altra macchina.
- **Riproducibilità.** Con lo stesso seme due vite sono identiche bit per bit su questa
  macchina (il test non usa immagini). Tra macchine, versioni e GPU non è stato provato: la
  versione è fissata in `uv.lock`, e le immagini possono cambiare tra GPU diverse.
- **Scatola nera.** Una rete addestrata può funzionare senza che si capisca come. Esami e
  lesioni la riducono ma non la eliminano: gli esami dicono che cosa fa una sezione, non come.
- **La catena di immaginazione moltiplica il costo.** Ogni tentativo aggiunge passi interni
  attraverso cui passa il gradiente. Gli errori di 4 si sommano a ogni anello: la catena
  sarà utile solo per pochi eventi.
- **6 arriva per ultima.** Se non impara a usare le altre sezioni lo si scopre tardi. Il
  ragionatore finto della fase 6 riduce il rischio: controlla che le sezioni bastino, non che 6 impari.
- **Mondi troppo simili.** Il cervello impara i mondi di addestramento, non a imparare. La
  loro varietà è una scelta di progetto importante quanto l'architettura.
- **Il principio "si fissa la struttura".** Va riletto: si fissano le sezioni e chi parla con
  chi; le regole dentro le sezioni non sono più scelte da noi ma addestrate.
- **La sezione 6 resta la parte meno definita.** Una rete ricorrente mostra una forma di
  ragionamento solo se il compito la richiede, e non è garantito nemmeno allora.

## Che cosa si riusa

| di oggi | diventa |
|---|---|
| idea di `objects_of` | detector finto, letto dal simulatore 3D |
| logica del perfetto | comandante finto sopra la sezione 7, finché mancano le altre |
| idea di O (fame sul valore) | bisogni e valore in 4 e 6 |
| idea di K (cassetti) | contesti di 5 |
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
4. **Tempo:** deciso: a eventi per 3, 4, 5 e 6, a passi per 1, M e 7.
5. **Evoluzione:** resta sopra il gradiente fin dall'inizio, o entra solo alla fine?
6. **Valore:** deciso in parte. La scelta sta nella Valutazione di 6, ma che cosa ricevere da 4 e
   5 è ancora da definire.
7. **Immaginazione:** quanti tentativi per evento, e quanto lunga la catena?
