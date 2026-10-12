# Note di progetto

Idee e direzioni non ancora in programma. Non descrivono il codice attuale.

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
- **Plasticità**: pesi che cambiano durante la vita secondo regole a loro volta
  evolute. Si combina bene col genoma indiretto (il genoma descrive come la rete
  cresce *e* come impara).
- **Neuroni di tipo fisso eccitatorio o inibitorio**: un gene per neurone decide
  il tipo; da confrontare coi pesi con segno attuali.
- **Corpo evolvibile**: numero, apertura e portata dei raggi nel genoma, con un
  costo energetico per ogni sensore.
