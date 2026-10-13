# Note di progetto

Idee e direzioni non ancora in programma. Non descrivono il codice attuale.

## Percorso futuro: reti che imparano durante la vita

Obiettivo dichiarato: arrivare, se possibile, a una rete che impara *online*,
cioè mentre vive, e non solo di generazione in generazione. Oggi i pesi sono
fissi dalla nascita alla morte.

### Il punto chiave: l'evoluzione deve avere un motivo per inventare l'apprendimento

Se ogni vita presenta lo stesso problema (trova il cerchio azzurro, evita il
verde), la soluzione più economica è scriverla nei pesi alla nascita. Un agente
che deve imparare perde tempo rispetto a uno che nasce già capace, quindi la
selezione lo scarta. Si può dare alla rete tutta la plasticità che si vuole:
resterà inutilizzata.

Perché imparare convenga servono tre condizioni, e riguardano il mondo, non la
rete:

- **Qualcosa che i geni non possono prevedere.** Esempio: a ogni vita si
  sorteggia quale "colore" è cibo e quale è veleno. Chi nasce con una regola
  fissa sbaglia metà delle volte; vince solo chi assaggia e si adatta.
- **Più occasioni nella stessa vita.** Oggi chi tocca il bersaglio ha finito:
  non c'è una seconda volta in cui usare ciò che ha capito. Servono bersagli
  ripetuti, per esempio cibo che ricarica energia e ricompare altrove.
- **Un segnale di "bene" e "male".** Il dolore c'è già; manca il corrispettivo
  positivo, che arriverebbe col cibo che ricarica.

Senza queste, il resto è lavoro sprecato. Con queste, anche una rete semplice
può sviluppare apprendimento.

### Quanto dettaglio biologico serve nei neuroni a impulsi

Il dettaglio serve dove tocca l'apprendimento, non dove rende il neurone più
"vero". In ordine di utilità:

| Dettaglio | A cosa serve | Giudizio |
|---|---|---|
| Collegamenti che cambiano forza in base a chi scatta prima di chi | è il meccanismo di apprendimento vero e proprio | indispensabile |
| Segnale diffuso di ricompensa e punizione (l'analogo della dopamina) che dice ai collegamenti quando cambiare | senza, la rete cambia ma non sa in che direzione | indispensabile |
| Geni che descrivono la *regola* di apprendimento, non i pesi | è ciò che l'evoluzione deve poter regolare | indispensabile |
| Autoregolazione dell'attività (un neurone troppo attivo si calma, uno muto si sensibilizza) | impedisce alla rete che impara di esplodere o spegnersi | quasi necessario |
| Costo energetico per ogni impulso | spinge verso reti parsimoniose; in natura il cervello consuma circa un quinto dell'energia del corpo, da noi pensare è gratis | utile ed economico |
| Pausa obbligata dopo l'impulso, affaticamento | dinamiche più ricche | facoltativo |
| Ritardi sui collegamenti | senso del tempo più fine | facoltativo |

Cosa **non** vale la pena: canali ionici, forma dettagliata dell'impulso,
neuroni con più compartimenti, varietà di sostanze chimiche. Moltiplicano il
costo di calcolo (le reti a impulsi costano già circa il doppio di quelle
continue) e non aggiungono capacità di apprendere. Il neurone attuale è al
livello di dettaglio giusto: tutto ciò che conta si può aggiungere sopra.

### Il ruolo delle reti continue (`rate`)

Tenerle come termine di paragone anche per l'apprendimento. Le stesse regole
("i collegamenti tra neuroni attivi insieme si rinforzano, modulati dalla
ricompensa") funzionano anche sui neuroni continui, sono più semplici e meglio
studiate. Se un giorno `spike-ei` impara e `rate` no, o viceversa, è un
risultato vero.

### Ordine consigliato

1. Cambiare il mondo perché imparare convenga: cibo che ricarica e ricompare, e
   un elemento sorteggiato a ogni vita.
2. Aggiungere il segnale di ricompensa come senso.
3. Aggiungere la plasticità con regola evolvibile, prima su `rate` e poi su
   `spike-ei`.
4. Solo dopo, i dettagli di contorno della tabella.

È un cambio di direzione rispetto ai compiti 2 e 3 del piano iniziale (spingere
l'oggetto, consegna): le due strade sono compatibili ma vanno messe in ordine.

### Cose da ricordare quando si riprende

- **Si combina col genoma a regole di crescita** descritto più sotto: il genoma
  descriverebbe come la rete cresce *e* come impara.
- **Confronti tra tipi di rete: servono più prove.** Finora ogni tipo è stato
  provato una volta sola, con un solo seme. Prima di trarre conclusioni conviene
  ripetere ogni configurazione con almeno 5 semi e guardare media e variabilità.
- **Osservazione da verificare:** nelle reti a impulsi la quota di neuroni
  inibitori è salita da sola dal 20% iniziale a circa il 36–38% in 150
  generazioni (mappa piccola, una prova). Non è chiaro se sia un vantaggio reale
  o una deriva casuale.
- **Interruttore non ancora provato in evoluzione:** `sensors_excitatory_only`
  (sensi solo eccitatori). Sulle reti casuali non cambia i successi iniziali.

## Percorso futuro: un genoma che descrive regole di crescita

### L'idea

Oggi il genoma è **diretto**: contiene la rete stessa, numero per numero (quali
neuroni esistono, quali collegamenti, ogni peso, ogni soglia). Un gene = un
pezzo di rete.

In natura non è così. Il DNA umano non elenca le sinapsi del cervello (non ne
avrebbe lo spazio): contiene **regole di sviluppo** che, eseguite durante la
crescita, costruiscono il cervello. Poche regole, riusate molte volte, producono
una struttura grande e regolare.

Un genoma **indiretto** farebbe lo stesso: l'evoluzione non modificherebbe i
collegamenti, ma le regole che li generano. Tra genoma e rete ci sarebbe una
fase nuova, lo **sviluppo**.

### Perché potrebbe valere la pena

- **Forme nuove.** È la strada più promettente verso reti "con una forma diversa
  da quelle attuali": regolarità, simmetrie e moduli ripetuti emergono dalle
  regole invece di dover essere scoperti peso per peso.
- **Simmetria gratis.** Il corpo è simmetrico (raggi a sinistra e a destra). Una
  regola come "collega ogni raggio al motore dal lato opposto" vale per tutti i
  raggi insieme; col genoma diretto va scoperta 9 volte.
- **Scala.** La dimensione del genoma non cresce con quella della rete: si
  possono avere migliaia di neuroni con poche decine di geni.
- **Corpo che cambia.** Se un giorno il numero di raggi evolvesse, una regola
  continuerebbe a funzionare; una tabella di pesi no.
- **Mutazioni più sensate.** Una mutazione cambia una regola, quindi molti
  collegamenti in modo coerente, invece di un collegamento solo.

### Tre modi di farlo, dal più semplice al più "naturale"

1. **Regola geometrica (in stile HyperNEAT).** Ogni neurone ha una posizione
   (i raggi sono già disposti a ventaglio, i motori sono sinistra/destra). Il
   genoma è una piccola funzione che, date le posizioni di due neuroni, dice se
   sono collegati e con che peso. Si evolve la funzione. È il passo più piccolo
   e sfrutta subito la geometria del corpo.
2. **Crescita per regole di riscrittura (L-system, codifica cellulare).** Si
   parte da un neurone solo; il genoma è un elenco di istruzioni tipo "dividiti",
   "collega il figlio al genitore", "ripeti". La rete è il risultato
   dell'esecuzione.
3. **Sviluppo con segnali chimici (reti di regolazione genica).** I neuroni
   nascono in uno spazio 2D, emettono e sentono "sostanze", e i geni si accendono
   o spengono in base alle concentrazioni: da lì crescono, si dividono, allungano
   collegamenti verso chi emette un certo segnale. È il più vicino alla biologia
   e il più difficile da far funzionare.

Consiglio di partenza: il primo. Dà già il vantaggio principale (regolarità e
simmetria) con poco codice.

### Cosa serve nel codice, e cosa c'è già

Il punto chiave è che **il resto del sistema non deve accorgersene**. Simulatore,
sensi, calcolo della rete e selezione restano identici. Cambia solo una cosa:

    genoma diretto:    genoma  ==  tabelle della rete
    genoma indiretto:  genoma  --sviluppo-->  tabelle della rete

`Brains` lavora già su quattro tabelle (neuroni attivi, collegamenti, pesi,
soglie). Basta quindi una funzione di sviluppo che le produca a partire dal nuovo
genoma, e nuovi operatori di incrocio e mutazione che agiscano sulle regole.
Per questo conviene tenere separati, come ora, il genoma e i suoi operatori
(`evonets/evo/`) dal calcolo della rete (`evonets/net/`).

Servirà inoltre dare una **posizione** a ogni neurone d'ingresso e d'uscita
(angolo del raggio, tipo di canale, lato del motore): oggi è implicita
nell'ordine degli ingressi.

### Come procedere quando sarà il momento

1. Avere prima un riferimento solido col genoma diretto: quante generazioni
   servono per ciascun compito, che reti emergono.
2. Assegnare le posizioni ai neuroni e implementare la regola geometrica.
3. Confrontare a parità di tutto il resto: stessi compiti, stessa popolazione,
   stesse regole di selezione. Misurare generazioni necessarie, dimensione delle
   reti, e soprattutto se le soluzioni reggono quando il compito o il corpo
   cambiano.
4. Solo se il confronto è incoraggiante, passare alla crescita vera e propria.

### Rischi

- **Più lento da far partire.** Una regola casuale produce spesso reti inutili
  (tutte collegate o per niente): la percentuale di successi casuali iniziali
  può crollare, e con la regola "si riproduce solo chi completa il compito"
  questo significa estinzioni.
- **Difficile ritoccare i dettagli.** Le regole sono brave nelle regolarità e
  scarse nelle eccezioni. Una soluzione comune è un sistema misto: regole per la
  struttura, più qualche gene diretto per le correzioni.
- **Più difficile da interpretare.** Capire perché una regola produce una certa
  rete è meno immediato che guardare i pesi.

### Altre idee correlate, più lontane

- **Due copie di ogni gene con dominanza** (come nei genomi diploidi): protegge
  varianti nascoste e rallenta la perdita di varietà. Complessità notevole per
  un beneficio incerto qui.
- **Plasticità**: vedi la sezione "reti che imparano durante la vita" sopra.
- **Corpo evolvibile**: numero, apertura e portata dei raggi nel genoma, con un
  costo energetico per ogni sensore.
