# Cervello ragionante (progetto)

Stato: progetto. Del codice esistono il mondo 3D con le pile e i due agenti di
controllo; il cervello no. Un **nuovo tipo di cervello**, accanto a quelli di oggi: il cervello a sezioni (docs/SPEC_CERVELLO_SEZIONI.md) resta il riferimento da
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
addestrare gli esseri"), resta vero che nessun essere usa il gradiente mentre vive.
Cambia che il suo cervello di nascita è prodotto anche dal gradiente.

## Addestrare una sezione alla volta

Se tutte le sezioni imparano insieme, niente garantisce che ognuna faccia ciò che dice il
suo nome. Quindi ogni sezione si addestra da sola su un compito suo, viene esaminata e poi
si blocca. La sezione dopo impara con quelle prima già bloccate.

| sezione | esame, sempre in mondi mai visti | compagni |
|---|---|---|
| 7 | raggiunge l'oggetto indicato e lo morde | nessuno |
| 3 e 4 | dopo pochi eventi prevede il prossimo evento e il suo esito | in coppia: che cosa ricordare dipende da che cosa serve prevedere |
| 5 | richiama l'esito giusto dopo molti eventi diversi, anche partendo dall'esito; regge ai cambi di regime | sequenze generate da noi |
| 2 | con una domanda data da noi sceglie l'oggetto giusto; un pericolo vicino passa comunque | domande date da noi |
| 6 basso | raggiunge un sotto-scopo dato | 2, 3, 4, 5, 7 bloccate |
| 6 alto | **rivalutazione alla prima prova** | tutto bloccato, anche 6 basso |
| 1 | trova gli oggetti e dà la stessa interfaccia del detector finto | etichette del simulatore |

Regole:

- **Imparare, non memorizzare.** L'esame di 3, 4 e 5 è "impara in fretta un mondo mai visto",
  non "conosci questo mondo". Ogni sezione si allena su molti mondi casuali, con i due anelli.
- **Dati che seguono il cervello.** Un essere che si muove a caso apre poche casse e mangia
  ancora meno. Si alternano raccolta col cervello attuale e riaddestramento.
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
sposta. Corpo su due ruote con telecamera a 120°, immagine 264×264 più profondità in cm. Il
telaio si inclina al massimo di 6° perché ribaltandosi resterebbe bloccato. Poi l'arena
delle casse, con casse e interruttori come forme colorate. Le zampe, se serviranno, arrivano dopo.

**Agenti di controllo** (`evonets/brains/mondo3d/`), non cervelli: un cervello si misura tra i due.

| agente | che cosa sa | pile buone/min | cattive/min | scelta |
|---|---|---|---|---|
| **perfetto** (soffitto) | legge la verità dal simulatore; non usa la telecamera | 11,7 | 0 | 1,00 |
| **base** (pavimento) | solo la profondità: vaga e morde ciò che ha davanti | 0,06 | 0,06 | 0,50 |

Misure con 1 essere per recinto: perfetto su 4 recinti per 20 minuti, base su 4 recinti per 12
minuti (solo 6 pile: la scelta è indicativa). Il perfetto ha la scelta perfetta, ma la strada
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

## Le sezioni

### 1. Percezione

- **Fa:** separa gli oggetti e li descrive.
- **Entra:** immagine della telecamera. **Esce:** un embedding per oggetto, fino a un massimo fisso.
- **Anello esterno:** impara a vedere. **Anello interno:** non cambia, o molto poco.
- **Prima versione:** un **detector finto**, cioè gli oggetti letti dal simulatore,
  trasformati in embedding, con un'interfaccia uguale a quella del detector vero.
  Il detector vero legge la telecamera e arriva quando il resto funziona.

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
- **Anello esterno:** impara che cosa scrivere e quando. **Anello interno:** cambia solo
  l'attività.
- **Fatta di neuroni:** un gruppo ricorrente con cancelli, come D, o una fila di
  posti. Quale dei due è una decisione aperta.

### 4. Modello del mondo

La sezione causale e quella della previsione sono **la stessa**: "A causa B" si impara
come "se succede A, faccio X, prevedo B".

- **Fa:** data la memoria e un'azione, prevede il prossimo evento e il suo esito.
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
- **Anello esterno:** impara come si impara. **Anello interno:** sinapsi plastiche che
  fissano ciò che vale in questo mondo.

### 5. Memoria a lungo termine

- **Fa:** conserva le situazioni che contano e quanto erano buone o pericolose.
- **Entra:** uscita di 4, memoria a breve (residuale) e **richiesta da 6** ("che cosa portava a questo esito?").
- **Esce:** i ricordi simili alla situazione di adesso, verso 6.
- **Scrive solo ciò che ha sorpreso**, guidata dall'errore di 4.
- **Richiama per somiglianza:** una situazione fa riaffiorare quelle che le somigliano.
- **Contesti:** ciò che è pericoloso può cambiare. I ricordi restano legati al loro
  contesto, come i cassetti di K, per non cancellarsi a vicenda.

### 6. Ragionamento

- **Livello alto:** sceglie lo scopo, con un ritmo lento. Si aggiorna quando un evento
  conta, non a ogni passo.
- **Livello basso:** sceglie la prossima azione, con un ritmo veloce, per
  avvicinarsi allo scopo.
- **Memoria del pensiero:** uno stato ricorrente che rientra in sé stesso: a che punto
  sono, che cosa ho appena pensato.
- **Entra:** ricordi da 5, memoria a breve, simulazioni da 4, bisogni, azione eseguita.
- **Esce:** domanda verso 2, azione voluta verso 7, azioni immaginate verso 4.
- **Non prevede.** Propone e valuta: la previsione è tutta in 4 (vedi "Come 6 usa 4").
- **Esplorare:** quando l'incertezza è alta conviene scoprire. Il livello alto lo
  riceve dall'errore di 4 e dai bisogni.

### 7. Motore

- **Fa:** traduce l'azione voluta in comandi.
- **Scorciatoie:** riflessi e guida verso l'oggetto passano direttamente da qui,
  senza attendere il ragionamento.
- **Esce:** comandi al corpo e **azione eseguita**.

### B. Bisogni

- **Fa:** dice che cosa manca al corpo.
- **Entra:** energia e stati del corpo. **Esce:** verso 4 e 6.
- Un bisogno cambia il valore delle cose e il perché di ogni azione. Senza, l'essere non
  ha motivi. Se un bisogno cambia, cambia anche la domanda di 6 verso 2: la fame
  agisce sull'attenzione passando dal ragionamento.

## L'azione: dove va

Esiste una copia dell'azione, come nel cervello vero (la copia efferente). La copia
giusta è l'azione **eseguita**, in uscita da 7: un riflesso può aver cambiato quella voluta.

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

Si costruisce dalla fine, come prima, perché il rischio maggiore è in 4 e 6, non nella vista.
Ma il rischio di 6 si misura **subito**, prima di costruire le sezioni che dovrebbe usare.

| fase | lavoro | misura per passare alla fase dopo |
|---|---|---|
| 0 | mondo 3D: **fatti** MuJoCo, arena delle pile, agenti di controllo, interfaccia grafica di base; **mancano** arena delle casse, mondi casuali, immagini su più processi | stessa vita con lo stesso seme identica (**verificato** su questa macchina); velocità nota, con almeno 100 esseri (**5,5 volte il tempo reale, senza immagini**) |
| 1 | 7 | supera l'esame in scene nuove |
| 2 | **prova di 6:** una 6 semplice con 3, 4 e 5 **perfette**, lette dal simulatore; anello esterno | rivalutazione superata in mondi mai visti. Il gradiente regge, con un costo accettabile. **Se fallisce ci si ferma e si ripensa 6** |
| 3 | 3 e 4 | superano l'esame; al posto di quelle perfette la rivalutazione regge |
| 4 | 5 | supera l'esame; ritorno rapido a un regime già vissuto |
| 5 | 2 | supera l'esame; 6 la usa per trovare ciò che cerca |
| 6 | 6 basso, con la catena dell'immaginazione | sequenze di sotto-scopi che reggono ai distrattori |
| 7 | 6 alto | rivalutazione alla prima prova; togliendo la catena, peggiora |
| 8 | ritocco finale di tutto insieme | tutti gli esami ripetuti e superati |
| 9 | 1: detector vero sulla telecamera, poi ritocco | i risultati restano vicini a quelli col detector finto |
| 10 | evoluzione sopra il gradiente | tara le costanti e si misura che cosa cambia |

La fase 2 è la più importante: se 6 non ragiona nemmeno con sezioni perfette, inutile
costruire quelle vere. Se la supera, ogni fase dopo sostituisce un pezzo perfetto con uno vero
e misura quanto si perde.

## Criticità

- **Come passa il gradiente in una vita.** Una vita di 400 s sono 8000 passi: il gradiente
  su tutta la vita è costoso e instabile. Addestrando le sezioni una alla volta il problema
  resta quasi solo per 6, che lavora a eventi con il resto bloccato. Si misura in fase 2.
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
- **6 arriva per ultima.** Se non impara a usare le altre sezioni lo si scopre tardi. Per
  questo la fase 2 la prova in anticipo.
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
| motore F e riflessi | idee per la sezione 7, da rifare per il corpo 3D |
| idea di O (fame sul valore) | bisogni e valore in 4 e 6 |
| idea di K (cassetti) | contesti di 5 |
| JAX | anello esterno |
| `evonets/viz/` | interfaccia grafica |
| banco di prova e lesioni | stessa metodologia |

Nel codice: `evonets/arenas/mondo3d/`, `evonets/brains/mondo3d/` (controlli),
`evonets/viz/mondo3d/`; il cervello andrà in `evonets/brains/ragionante/`. Il cervello a sezioni e l'arena delle pile non cambiano.

## Riferimenti da tenere presenti

- Wang e colleghi (2018): corteccia prefrontale e meta-apprendimento per rinforzo.
- Duan e colleghi (2016), RL²: imparare a imparare con una rete ricorrente.
- Miconi e colleghi (2018): plasticità addestrata con il gradiente.
- Sutton (1991); Hafner e colleghi (Dreamer): pianificare con un modello imparato.
- Jensen, Hennequin e Mattar (2024): immaginare come azione in una rete ricorrente.
- Ha e Schmidhuber (2018), World Models: vista e modello addestrati a parte.
- Tolman (1948); Daw e colleghi (2011): la prova della rivalutazione.

## Decisioni aperte

1. **Memoria a breve:** gruppo ricorrente con cancelli, o una fila di posti?
2. **In vita cambiano solo le attività, o anche sinapsi plastiche?** Solo attività è più
   semplice. Le sinapsi durano di più e servono alla memoria a lungo termine.
3. **Che cosa cambia tra una vita e l'altra:** quali regole del mondo sono da scoprire?
4. **Tempo:** passi fissi o eventi?
5. **Evoluzione:** resta sopra il gradiente fin dall'inizio, o entra solo alla fine?
6. **Valore:** una sezione a parte, o è parte di 4 e 6?
7. **Immaginazione:** quanti tentativi per evento, e quanto lunga la catena?
