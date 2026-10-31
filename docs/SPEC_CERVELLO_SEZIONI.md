# Cervello a sezioni (progetto)

Stato: progetto, niente codice. Diventerà un **nuovo tipo di cervello**,
selezionabile dalle impostazioni iniziali; i tipi di oggi e le arene
esistenti non cambiano.

## Principio: si fissa la struttura, non il contenuto

Definiamo **come è fatto** il cervello: quante sezioni, quanti neuroni, chi
parla con chi, a che velocità cambia ciascuna parte, quali regole generiche
usa. **Non** definiamo cosa rappresenta ogni neurone. "Rosso a sinistra",
"avvicinati", "rosso → male" sono ipotesi da verificare, non specifiche.

| livello | fissato da noi | emerge dall'esperienza (e dall'evoluzione) |
|---|---|---|
| **architettura** | sezioni, numero di neuroni, collegamenti tra sezioni | |
| **velocità** | per sezione, il range di τ (reazione) ed η (apprendimento) | |
| **regole generiche** | competizione laterale, omeostasi, plasticità a tre fattori, segnale di valore V | |
| **contenuto** | | cosa rappresenta ogni neurone: colori, lati, oggetti, programmi, ricordi |

## Perché

Il test di capacità (docs/SPEC_CAPACITA.md) ha mostrato che il cervello di
oggi non impara dentro la vita, anche con la regola giusta e veloce:

- il cambiamento utile (0,3% per frutto) affoga nel rumore dell'hebbiano
  (10% in una vita);
- anche forzato ×100, cambia le sinapsi di entrambi i colori e il
  comportamento non cambia: quando arriva il dolore il cervello non sa più
  *cosa* ha mangiato (il frutto è fuori vista, gli occhi si adattano);
- il critico sui sensi grezzi (la sorpresa) peggiora: l'occhio che si adatta
  durante l'avvicinamento sembra una delusione.

Il problema, quindi, non è solo quante cose il cervello sa fare, ma che
mancano **parti con ruoli diversi** e collegate bene: una memoria di ciò
che si è appena visto, una descrizione stabile, un posto dove la scelta
viene fatta e imparata. Le offriamo come struttura; come riempirle lo
decide l'esperienza.

## Le scale della memoria

| scala | dove sta | come | esempio (ipotesi) |
|---|---|---|---|
| secondi | sezione D | **attività** che resta accesa, nessuna sinapsi cambia | ciò che si stava facendo |
| minuti-ore | sezione C | sinapsi molto plastiche: imparano in un colpo e sbiadiscono | ciò che è appena successo |
| vita | sezioni B, L | sinapsi poco plastiche, cambiano con molte esperienze o per consolidamento | come sono fatte le cose (B), cosa vale sempre (L) |
| generazioni | geni | evoluzione | l'architettura stessa, i pesi di nascita |

Memoria veloce e lenta insieme sono i *sistemi di apprendimento
complementari* (McClelland, McNaughton, O'Reilly 1995): la memoria veloce
impara subito, la lenta lentamente, e i ricordi passano dalla prima alla
seconda (consolidamento) senza cancellare ciò che la seconda sapeva.

## Lo schema

```
  sensori ─► [A interfaccia sensoriale, fissa] ─► [B nucleo stabile] ──┬──► [E selezione] ─► [F motore] ─► corpo
  (qualsiasi)     continua, nessuna                 lento              │       ▲
                                                                       │       │
                                    [D memoria di lavoro] ◄────────────┤       │
                                    [C memoria veloce]   ◄────────────┘       │
                                    [L memoria lunga]    ◄── consolidamento da C
                                         └──► E (inclinazioni stabili)        │
                                                                               │
                                    [V valore] ◄── corpo (gusto, dolore) ─ e ─┘
                                        │
                                        └── δ (segnale di sorpresa) ──► C, E, L
```

Tutto **continuo**, nella prima versione. Gli impulsi restano un'opzione
per sensori che già parlano a impulsi (telecamere a eventi).

## Le sezioni

Per ogni sezione: cosa è fissato, cosa emerge, come si misura che ha
funzionato. Numeri di neuroni indicativi.

### A. Interfaccia sensoriale (fissa, continua, nessun apprendimento)

**Fissato.**
- Riceve i valori dei sensori, anche molti e di tipo diverso (telecamere,
  vibrazioni, rilevatori di colore).
- Li porta su una scala comune tra 0 e 1, separa per ogni canale un valore
  **lento** ("com'è") e uno **veloce** ("cosa è cambiato"), e riduce a
  qualche decina di canali.
- Non impara: è l'equivalente della retina e dei recettori.

**Emerge.** Niente: è una funzione fissa. Dal punto di vista di B, cambiare
sensore significa cambiare solo A.

**Come si verifica.** Stessi valori di ingresso da sensori diversi danno
gli stessi canali in uscita; il resto del cervello non cambia.

### B. Nucleo stabile (continuo, apprendimento lentissimo)

**Fissato.**
- Riceve tutti i canali di A.
- Ha N neuroni continui (proposta: 12–20), con inibizione laterale tra
  neuroni vicini. L'inibizione è una regola generica, non dice quali neuroni
  si inibiscono per colore o lato: lo decide la topologia del foglio, che
  lascio come parametro.
- Costante di tempo di reazione τ_B in un range medio.
- Apprendimento lentissimo (η circa 1/100 di quello di E), più omeostasi
  (ogni neurone cerca un suo livello di attività).
- Riceve un po' di δ: impara di più ciò che si accompagna a sorprese
  (attenzione minima; senza, rischia di descrivere bene cose inutili).
- Riceve un segnale dall'alto da D ed E ("cosa sto cercando"), che rafforza
  ciò che è pertinente (anello all'indietro).
- Manda il suo stato corrente a C, D, E e V. Non prende decisioni.

**Emerge.**
- Quali combinazioni di canali rappresenta ogni neurone. Con gli stimoli
  giusti dovrebbero dividersi tra colori, posizioni, oggetti, perché è
  quello che si ripete nell'esperienza: come i neuroni della corteccia
  visiva che imparano bordi e colori senza che nessuno glielo dica.

**Come si verifica.** Dopo la vita (o dopo l'evoluzione), per ogni neurone
di B: a quali canali risponde? Si misura la **selettività**, cioè quanto
un neurone risponde a uno solo dei canali. Se le selettività sono alte e
diverse tra neuroni, la specializzazione è emersa. Se sono uniformi, il
problema è nella struttura, nei segnali o nelle velocità, non nel contenuto.

### D. Memoria di lavoro (continua, nessuna sinapsi che cambia)

**Fissato.**
- Gruppo di neuroni con autoeccitazione e competizione interna.
- Costante di tempo di memoria di qualche secondo (2–5 s).
- Riceve B e il segnale di "evento" di A.
- **Tenere** è fisso: attrattore o τ lungo, nessuna sinapsi cambia per
  tenere.
- **Il cancello** (quando scrivere, quando sostituire) impara con δ, come nel
  modello di O'Reilly: si apre di più per ciò che ha portato sorprese.

**Emerge.** Cosa tenere, cioè quando il cancello si apre.

**Come si verifica.** Dopo un evento, uno stato resta attivo per il tempo
previsto, e viene sostituito quando arriva un evento nuovo. Si misura
quanto a lungo il contenuto resta utile per E.

### C. Memoria veloce (continua, apprendimento rapido, sbiadisce)

**Fissato.**
- Neuroni con apprendimento veloce (η alto nel range) e decadimento delle
  sinapsi in minuti.
- Rappresentazione sparsa: poche unità attive per situazione, come
  richiede la competizione.
- Riceve B e D, manda segnali a E e V, e consolida in B.

**Emerge.** Quali situazioni ricordare e come associarle a ciò che segue.

**Come si verifica.** Dopo un solo evento associato a una conseguenza, la
situazione richiama la conseguenza al prossimo incontro; senza conferme, il
ricordo sbiadisce entro qualche minuto.

### E. Selezione dell'azione (continua, apprendimento medio)

**Fissato.**
- N unità in competizione, con inibizione reciproca forte.
- Ognuna collegata a F, cioè a un comando motorio. Il collegamento
  unità → comando è fisso, ma non ha un significato predefinito.
- Riceve i segnali da B, C, D e dall'istinto (pesi di nascita).
- Apprendimento medio con la regola a tre fattori e la traccia di
  eleggibilità, modulato da δ. **Niente hebbiano di base** (nel test di
  capacità era il rumore che copriva il segnale).
- **Esplorazione regolabile**: quanto la competizione è rumorosa
  (noradrenalina in Doya 2002), gene e modulata dall'incertezza. Senza,
  l'essere non prova mai l'altra scelta.

**Emerge.** Quale situazione porta a quale unità, e quindi quale comando.

**Come si verifica.** Dopo la vita, si misura quali situazioni fanno
vincere quali unità. Se un'unità vince sempre nelle stesse situazioni
e produce un comando utile, è emersa una scelta.

### F. Motore (fisso, nessun apprendimento dentro la vita)

**Fissato.**
- Traduce i comandi in sterzo e spinta, bilaterali come oggi.
- Non impara dentro la vita: così imparare non rovina il saper muoversi.
- Cambia solo con l'evoluzione.

**Emerge.** Niente di fondamentale: è una traduzione fissa.

**Come si verifica.** Un comando fisso produce sempre lo stesso movimento
nelle stesse condizioni del corpo.

### L. Memoria lunga (continua, apprendimento lentissimo)

**Fissato.**
- Un gruppo grande di neuroni (proposta: 20–40), con plasticità molto più
  bassa di E e di C, e rappresentazione sparsa perché i ricordi non si
  sovrappongano.
- Riceve la descrizione della situazione da B (il contesto, non i dettagli
  dell'istante) e il consolidamento da C, che arriva solo per ricordi
  confermati più volte.
- Manda inclinazioni stabili a E, che si sommano ai voti di B, C e D, e
  opzionalmente a V.
- Pesi di partenza dai geni: la conoscenza di base è innata, quella di una
  vita si aggiunge sopra e si perde alla morte.
- Vincolo: **contiene solo ciò che vale tra le vite.** Ciò che cambia da una
  vita all'altra (per esempio quale colore è buono) resta in C.

**Emerge.** Quali abitudini e regolarità generali si consolidano: come si
insegue, che un morso fa male, che il cibo calma la fame, le sequenze motorie
usate spesso.

**Come si verifica.**
- *Stabilità*: dopo molte esperienze, ciò che L ha imparato resta; una
  singola esperienza nuova non lo cancella.
- *Generalità*: L non deve riprodurre il colore buono della vita precedente.
  Se lo fa, la sua plasticità è troppo alta o riceve consolidamenti di
  ricordi che valgono solo in una vita.
- *Utilità*: nelle situazioni che si ripetono, la scelta migliora senza
  diventare un'abitudine cieca.

**Confine con E.** E impara in fretta "situazione → programma" in questa
vita; L impara lentamente "situazione → programma in generale". Se fanno lo
stesso lavoro, uno dei due è di troppo.

### V. Valore, il critico (continuo, apprendimento proprio)

**Fissato.**
- Riceve B, C, D, E e i segnali del corpo (gusto, dolore, sollievo).
- Calcola la previsione V e la sorpresa δ = piacere − dolore − V/τ + dV/dt
  (TD continuo).
- Impara le sue connessioni con δ, sulla traccia dei suoi ingressi.
- **Orizzonte** τ (quanto lontano guarda nel futuro; serotonina in Doya
  2002) come gene.
- Il maestro è il corpo: il piacere e il dolore veri, più i valori innati
  legati ai sensi (sotto).

**Emerge.** Cosa prevedere e da quali stati. Il contenuto della previsione
non è scritto da nessuna parte.

**Come si verifica.** Con un indizio seguito da un cibo, ripetuto, δ
dovrebbe passare dal cibo all'indizio (il test esiste già per il critico
sui sensi: `test_surprise_moves_from_the_reward_to_the_cue`). Il test va
ripetuto con il critico sugli stati di B, C, D.

**È generale?** Il meccanismo è lo stesso per ogni compito, perché
"cosa è bene" lo decide il corpo, e cosa prevedere lo impara da solo.
Limite: prevede solo da ciò che B, C, D gli danno. Se B non distingue,
nemmeno lui.

### Valori innati legati ai sensi (dai Cyber Rodent)

- Un piccolo blocco tra il corpo e V: oltre a gusto, dolore e sollievo,
  alcune percezioni sono un po' piacevoli o spiacevoli di nascita (vedere
  cibo, sentire acqua).
- Pesi come geni, con un tetto (non devono valere più del premio vero, o
  diventano un premio a vuoto).
- Servono a dare segnali più frequenti: si impara prima di arrivare al
  premio.

### Modulatore dell'incertezza (da Doya 2002)

- Si calcola dalla media recente di |δ|: molte sorprese = il mondo è
  cambiato.
- Alza la velocità di apprendimento di C ed E e l'esplorazione di E; quando
  le sorprese calano torna giù.
- È la velocità di apprendimento che cambia con la situazione.

### Più avanti: una sezione con maestro (cervelletto)

Accanto a F, per movimenti precisi e previsione di cosa farà il corpo (con
un segnale d'errore). Nella prima versione F resta fisso.

## I traduttori

Non servono più: tutto è continuo nella prima versione. Gli impulsi, se
mai, arrivano solo dai sensori a eventi, dentro A.

## Il consolidamento: da C a B

Due modi possibili, anche insieme (decisione aperta):

1. **Ripetizione:** quando C richiama un ricordo che si rivela vero (δ vicino
   a zero o dello stesso segno), B riceve una spinta piccola.
2. **Riposo:** nei momenti fermi e sazi C riproduce i ricordi recenti e B li
   assorbe lentamente.

In entrambi i casi B cambia solo di poco: un ricordo non riscrive la sua
descrizione. Il consolidamento verso **L** è separato: va a L solo ciò che
si ripete tra le esperienze e non dipende da una vita specifica. Per L il
riposo è **indispensabile**: i ricordi nuovi vanno riprodotti alternati ai
vecchi, altrimenti li cancellano (McClelland 1995).

## Prima versione nel codice (2026-10-28)

`evonets/brains/pile/sections.py` (tipo `sezioni`), provata nell'arena delle pile
(docs/SPEC_ARENA_PILE.md, `evonets/arenas/pile/run.py`). Tutto continuo, in
NumPy; la versione JAX verrà dopo.

| sezione | come è fatta | cosa impara |
|---|---|---|
| A | sensi grezzi + "quanto è appena salito" per occhi, gusto, dolore (τ 0,3 s) | niente |
| B | 16 prototipi in competizione (vincono i 3 più vicini), τ 0,2 s, "coscienza" (DeSieno) perché nessuno resti muto | i vincitori si avvicinano all'ingresso e a ciò che segue (previsione del prossimo ingresso), η 0,05/s, ×(1 + 2·\|δ\|) |
| D | copia di B che resta (τ 4 s); cancello chiuso a riposo (0,25% a passo), aperto dagli eventi nuovi | solo il cancello, con δ |
| C | 64 neuroni, ognuno legge 4 ingressi a caso di B e D (fissi), ne vincono 4 | l'uscita verso E, con δ, in fretta; dimentica in 180 s |
| L | la stessa uscita di C | si avvicina a C solo a riposo (carichi e fermi a ricaricarsi), η 0,01/s |
| E | 5 programmi, softmax con temperatura; il programma in corso parte avvantaggiato; si sceglie ogni 0,5 s | attore: η · δ · traccia di (scelta − probabilità) × ingresso |
| V | previsione lineare da B, D, C | TD(λ), orizzonte 4 s |
| U | media di \|δ\| su 20 s rispetto a 200 s | niente: alza η di C ed E e la temperatura |
| F | 5 programmi fissi (verso il rosso, il giallo, il blu; esplora; evita) e il riflesso dei muri | niente |

**Le loss implicite.** Non c'è una loss globale: ogni regola ne minimizza
una locale. B l'errore di ricostruzione e di previsione del prossimo
ingresso (come il k-means e la codifica predittiva); V δ²; E e il cancello
di D la ricompensa attesa con segno meno (la regola δ × traccia segue in
media il gradiente: Williams 1992); C l'esito dell'episodio appena vissuto;
L la distanza da C. δ addestra solo le parti che decidono (E, cancello di
D, uscite di C) e il critico; B lo riceve solo come peso dei momenti.

**La traccia di E usa la scelta rispetto all'attesa** (azione scelta meno
probabilità), non pre·post: è ciò che rende la regola un gradiente
corretto, e per questo l'esplorazione non è un optional.

**Neuroni continui con costante di tempo.** A differenza della rete `rate`
di oggi (lo stato si ricalcola da zero a ogni passo), qui ogni stato segue
il suo ingresso con un τ proprio: così esistono sezioni lente e memorie che
durano.

**Connessioni fisse.** A, F, la tenuta di D e l'ingresso di C non imparano.
L'ingresso di C è il **corpo a fungo** degli insetti (Caron e colleghi 2013:
proiezioni fisse e casuali verso le cellule di Kenyon, impara solo l'uscita
con la dopamina). Alternativa da provare per D: un piccolo **reservoir**
(rete ricorrente fissa e casuale, Jaeger, Maass) che tiene l'eco di ciò che
è successo.

**Aggiunti (2026-10-28):** l'attenzione con salienza innata e imparata e
la fovea in A; F generico sull'oggetto attento (avvicinati, allontanati,
lascialo, esplora); tracce strette (E e cancello di D si marcano solo da
neuroni sopra 0,5); soglia adattiva su δ per tutto ciò che impara tranne
il critico. Dettagli e risultati: docs/SPEC_ARENA_PILE.md.

**Costanti come geni (2026-10-28):** ogni costante del cervello è un gene
(`genes.Genes`): nella fase A uguale per tutti, nella fase B la tara
l'evoluzione. Nel genoma ci sono anche i pesi di partenza e le connessioni
di C; il figlio eredita il genoma intero di uno dei genitori.

**Non ancora fatto:** l'anello all'indietro da D ed E verso B, i valori
innati nella fase A (sono a zero di proposito; evolvono nella fase B), il
riposo come riproduzione vera dei ricordi (per ora L si avvicina a C a
riposo), la diagnosi con il gradiente, la versione JAX.

**Correzione trovata dai test:** con il cancello di D socchiuso a riposo
(5% a passo) il ricordo spariva in meno di un secondo invece di durare τ;
ora a riposo è chiuso (0,25%).

## Un episodio, senza contenuto predefinito

Immaginiamo un essere che, per puro caso, ha una struttura che impara. Quello
che ci aspettiamo di osservare, senza averlo scritto da nessuna parte:

| momento | cosa dovremmo misurare |
|---|---|
| 1. vede un colore | in B, un gruppo di neuroni si accende più degli altri |
| 2. mangia, prova dolore | C registra la combinazione (colore, dolore); δ è negativo |
| 3. vede di nuovo il colore | C risponde subito; E sposta la sua scelta |
| a riposo | un po' della memoria di C arriva a B |

Se la misura ci dice il contrario (per esempio, B non diventa selettiva o
E non cambia scelta), la struttura o le velocità sono sbagliate: si cambia
quello, non il contenuto.

## Piano di lavoro

1. **Montare la struttura**, con regole generiche e pesi casuali dentro le
   sezioni (nessuna etichetta). Verificare che il flusso funzioni (le
   sezioni ricevono e mandano segnali nel tempo previsto).
2. **Diagnosi con il gradiente**: la stessa struttura addestrata con il
   gradiente sul compito. Se non impara nemmeno così, si cambia la
   struttura prima di provare le regole locali.
3. **Misurare l'emersione**: selettività di B, uso delle unità di E, contenuto
   di D e C, come da sezione "Come si verifica".
4. **Test di capacità** con il cervello montato: impara dentro la vita quale
   colore è buono, con e senza le singole sezioni (togliendone una alla volta
   si vede a cosa serve).
5. **Selezione naturale**: l'evoluzione **tara, non inventa**: dimensioni,
   τ, η, forza della competizione, esplorazione, orizzonte, pesi di nascita
   (anche la conoscenza innata di L) e pesi dei valori innati. Le forme delle
   regole restano quelle della tabella.
6. **Arena continua**, più avanti: bisogni (fame, sete), memoria dei luoghi.

## Regole di apprendimento per sezione

Ogni sezione ha **la sua famiglia di regole**, scelta da noi sulla base di
ciò che si sa del cervello; l'evoluzione ne tara i parametri (velocità,
soglie, forza della competizione). Idea di fondo: Doya (1999), regioni
diverse usano tipi diversi di apprendimento.

| sezione | lavoro | tipo | regola di partenza |
|---|---|---|---|
| B | rappresentare ciò che arriva | senza maestro | Hebb con competizione (Oja, BCM, codifica sparsa: Olshausen e Field 1996), lentissima; un po' di δ dice cosa conta |
| D | tenere per pochi secondi | quasi nessuna | attrattore o τ lungo; si impara solo *quando scrivere*, con la dopamina (gating, O'Reilly) |
| C | ricordare in un colpo | Hebb rapido su codici sparsi | modelli dell'ippocampo (Marr, Hopfield), con decadimento in minuti |
| L | sapere generale | Hebb lentissimo | consolidamento con riproduzione alternata (McClelland 1995) |
| E | scegliere ciò che rende | rinforzo (attore) | regola a tre fattori con traccia di eleggibilità (c'è già) |
| V | prevedere | rinforzo (critico) | TD continuo (Doya 2000; c'è già, da spostare sugli stati di B, C, D) |
| F | movimenti | fisso; più avanti con maestro | più avanti: regola dell'errore, come il cervelletto |

Limite: ogni regola ottimizza un obiettivo locale, e nessuno garantisce che
insieme servano al compito. Rimedio biologico: δ (e l'attenzione) arriva
anche alle sezioni senza maestro e dice cosa conta.

Tre strade possibili:

1. **Gradiente su tutto il cervello** (JAX lo calcola): non biologico, ma
   dice cosa la struttura *può* fare e che forma prendono le sezioni quando
   funziona. Lo usiamo come **diagnosi**, non come addestramento.
2. **Evoluzione delle regole** (imparare a imparare): già provata, non trova
   nemmeno la regola giusta per una sinapsi (docs/SPEC_CAPACITA.md).
3. **Via di mezzo (scelta):** famiglia di regole data, parametri
   all'evoluzione.

## Confronti

### Doya (1999, 2002)

Una teoria, non un modello che gira: corteccia = apprendimento senza
maestro, gangli della base = rinforzo (dopamina come errore TD), cervelletto
= con maestro (modelli interni del corpo). Nel 2002: i modulatori come
parametri dell'apprendimento (dopamina = δ, serotonina = quanto lontano
guardare nel futuro, noradrenalina = quanto esplorare, acetilcolina =
velocità di apprendimento).

| | Doya | progetto a sezioni | cervello di oggi |
|---|---|---|---|
| senza maestro | corteccia | B, L | BCM mescolato alle altre regole sulle stesse sinapsi |
| rinforzo | gangli della base | E, V | tre fattori su tutto; critico sui sensi grezzi (spento) |
| con maestro | cervelletto | manca (F fisso) | manca |
| memoria veloce, di lavoro | non al centro | C, D | mancano |
| modulatori | 4 | solo δ | solo δ |
| anelli all'indietro | sì | pochi | nessuno |
| evoluzione, corpo, mondo | no | sì | sì |

Il cervello di oggi ha quasi tutti i suoi ingredienti ma **in un unico
mucchio**, e il test di capacità mostra proprio ciò che la sua tesi
prevede: l'hebbiano copre il rinforzo, il critico senza buona
rappresentazione sbaglia. Il progetto a sezioni applica la sua divisione,
con due memorie in più e senza cervelletto.

Dove ci batte: verificato sulla biologia; ha il cervelletto (movimenti,
previsioni del corpo); quattro modulatori; anelli (attenzione, sequenze).
Dove siamo avanti: si fa girare e si misura; memorie esplicite (C, D, L);
evoluzione; un corpo in un mondo.

### Cyber Rodent (Doya, Uchibe, ATR, anni 2000)

Robot di circa 22 cm (telecamera, infrarossi, due ruote, mascella magnetica)
che sopravvivono catturando pile per ricaricarsi e si riproducono passandosi
il programma via infrarossi. Evoluzione incarnata (Elfwing, Uchibe, Doya,
Christensen 2008, 2011): si ereditano parte dei pesi, i **pesi delle
ricompense aggiuntive** (cosa è piacevole) e i **parametri
dell'apprendimento**; le ricompense utili a sopravvivere e riprodursi si
ottengono per evoluzione. La linea continua (Kanagawa e Doya 2024: quali
ricompense evolvono in ambienti diversi, con energia, morte e riproduzione).

Differenza chiave: **non hanno chiesto all'evoluzione di inventare
l'apprendimento.** Algoritmo di rinforzo collaudato, l'evoluzione ne regola
parametri e ricompense. Noi chiedevamo la regola stessa, e non la trova.

| | Cyber Rodent | noi |
|---|---|---|
| corpo | robot veri | simulati, migliaia, per migliaia di generazioni |
| ambiente | pile sul pavimento | cibo a ondate, clima, acqua, predatori, notte, siccità |
| apprendimento | rinforzo collaudato | plasticità biologica, oggi quasi inefficace |
| cosa evolve | parametri, ricompense, parte dei pesi | tutto, anche cervello e sensi |
| cosa è piacevole | evolve | fisso (evolve solo quanto pesa) |
| imparano dentro la vita | sì | appena (+1-3 punti) |

Da prendere: **ricompense innate legate ai sensi come geni** (per esempio
vedere un frutto piace un poco: segnali più frequenti, si impara prima del
premio) e la conferma della via di mezzo.

### Perché i robot generalisti di oggi vanno altrove

- **Obiettivo diverso:** compiti decisi dagli umani; l'evoluzione premia
  sopravvivere nel proprio mondo, ed è difficile da controllare.
- **Il gradiente ha vinto sull'efficienza:** a parità di calcolo batte
  evoluzione e regole locali; le regole biologiche non lo eguagliano ancora
  su larga scala (ricerca aperta: codifica predittiva, e-prop).
- **La conoscenza si copia:** modelli visione-linguaggio-azione (RT-2, π0)
  partono da miliardi di immagini e testi e imitano dimostrazioni umane.
- **I corpi veri sono lenti e fragili;** in simulazione si può (Isaac Gym),
  ma si porta male nel mondo reale.

Le idee sono sopravvissute dentro altre tecniche: Population Based Training
(DeepMind 2017), motivazione intrinseca e curiosità, meta-apprendimento,
strategie evolutive (OpenAI 2017), robot che si adattano ai danni con
repertori evoluti (Cully e colleghi, Nature 2015), mondi aperti (POET 2019).

Per noi: l'obiettivo è quello di Doya (capire come nasce l'intelligenza),
non quello delle aziende; il calcolo oggi c'è (GPU); la lezione è dare
strutture e regole che funzionano e lasciare all'evoluzione la messa a
punto.

## Decisioni aperte

1. **Quanta struttura fissare.** Il rischio è di fissare troppo (e allora
   stiamo progettando il contenuto) o troppo poco (e allora la valle di
   fitness torna). Serve la prima misura di emersione per decidere.
2. **Topologia di B.** Inibizione tra neuroni vicini in un foglio (come oggi
   per la geometria del cervello) o inibizione globale?
3. **Consolidamento:** ripetizione e riposo (il riposo serve a L); resta da
   decidere cosa conta come "riposo".
4. **Memoria veloce:** dimentica col tempo, quando è piena, o entrambi?
5. **Confine tra L ed E:** se L e E fanno lo stesso lavoro, quale dei due
   tenere? Proposta: L solo come inclinazione, E decide.
6. **Ricompense innate legate ai sensi** come geni (Cyber Rodent): quali e
   con che limite, perché non diventino un premio a vuoto.
7. **Più modulatori** (Doya 2002): almeno uno per la velocità di
   apprendimento o l'esplorazione (noradrenalina, acetilcolina).
8. **Una sezione con maestro** per il corpo (un cervelletto): movimenti
   precisi e previsione di cosa farà il corpo.
9. **Bisogni, memoria dei luoghi, attenzione, anelli all'indietro:** da
   aggiungere più avanti.


## O, il valore degli oggetti (2026-10-29)

Una sezione in più: due neuroni ("quanto vale ciò che guardo", "quanto vale
ciò che ho preso") e, per ogni caratteristica percepibile più lo sfondo, due
sinapsi che imparano (piacere e dolore, separati) con la loro incertezza.
Impara dagli esiti (Rescorla-Wagner alla fine di ogni episodio di piacere o
dolore), anche da quelli del maestro; la fame si applica quando si legge il
valore. La leggono il critico, E e l'attenzione, che non impara più niente
da sola. Dettagli e misure: SPEC_ARENA_PILE.md.

## K, i contesti (2026-10-29)

Lo stato nascosto del mondo, dedotto dagli esiti. Ogni contesto ha il suo
cassetto di O; vince quello che ha previsto meglio gli ultimi esiti; se
nessuno spiega l'esito appena avvenuto se ne recluta uno nuovo, che parte dai
valori innati. Prima si decide il contesto, poi impara solo il suo cassetto.
Prima versione: un neurone per contesto. Il passo successivo è un contesto
sparso (un pattern su più neuroni), in cui contesti simili condividano ciò
che sanno. Misure: TRAGUARDI_PILE.md.
