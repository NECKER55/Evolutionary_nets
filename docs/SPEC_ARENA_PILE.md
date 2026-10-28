# Arena delle pile (progetto)

Stato: progetto, niente codice. Un **nuovo modo di provare gli agenti**,
ispirato al Cyber Rodent project (Doya, Uchibe, ATR; docs/SPEC_CERVELLO_SEZIONI.md,
"Confronti"): esseri con i sensi già fatti che cercano pile per ricaricarsi
e si riproducono incontrandosi. Arena **nuova**, selezionabile dalle
impostazioni iniziali; le arene e i cervelli di oggi non cambiano.

Scopo: confrontare, nello stesso mondo e con gli stessi sensi, il cervello
di oggi, un attore-critico classico (riferimento) e il cervello a sezioni
(docs/SPEC_CERVELLO_SEZIONI.md), misurando soprattutto **l'apprendimento
dentro la vita**.

## Scelte concordate (2026-10-28)

| tema | scelta |
|---|---|
| cosa c'è da imparare | due tipi di pile (una buona, una cattiva) che ogni tanto si scambiano, **e** zone dove compaiono che cambiano nel tempo |
| riproduzione | numero di corpi **fisso**: chi muore rinasce con i geni di una coppia che si è accoppiata (come i robot veri) |
| geni del figlio | incrocio dei due genitori |
| scala | recinto piccolo con 4-10 esseri; centinaia di recinti in parallelo sulla GPU |
| conflitto | si mantiene: accoppiarsi costa tempo ed energia, bisogna scegliere tra pile e compagno |
| ricompense innate legate ai sensi | a zero nella fase A; evolvono nella fase B |
| fasi | prima A (popolazione fissa, nessuna nascita), poi B (con riproduzione) |
| costo del cervello | uguale per tutti i tipi di cervello: si confronta l'intelligenza, non il suo prezzo |
| sensi | un set fisso scritto per questa arena |
| riferimento | attore-critico classico (TD continuo, Doya 2000) |
| azioni scelte dal cervello | solo programmi di movimento; prendere la pila è automatico a contatto |
| gradiente | solo come diagnosi, mai per addestrare gli esseri |

## Il mondo

- **Recinto** quadrato con muri (proposta: 500 × 500), 4-10 esseri.
- **Pile** di due colori (proposta: rosso e giallo, gli occhi esistono già).
  Una è **buona** (ricarica), l'altra **cattiva** (scarica un po' e fa male).
  Ogni `T_inversione` (300 s, dopo la taratura) i ruoli si scambiano, in
  tutto il recinto.
- **Zone**: le pile compaiono in una di K zone (K = 4, gli angoli); ogni
  `T_zona` (200 s, sfasato dall'inversione) la zona attiva cambia. Una pila
  non presa si spegne dopo 40 s e ricompare nella zona attiva (altrimenti le
  cattive, evitate, si accumulano e riempiono la zona).
- **Ricarica**: quasi fermo e con la pila **davanti al muso** (entro 20°),
  l'essere la prende; una buona si consuma in 2 s, stando fermi; una
  cattiva toglie 8 di energia e fa male. Il dolore pesa quanto il danno
  rispetto a una pila buona (8/25), così avvicinarsi alle pile conviene già
  prima di distinguere i colori.
- **Energia**: cala col tempo e col movimento; a zero si muore. Tetto di età
  (proposta 600 s) perché le vite finiscano anche senza morire di fame.
- **Altri esseri**: hanno una luce blu sul muso (riconoscibile dagli occhi).
- **Accoppiamento** (fase B): due esseri con energia sopra una soglia, muso
  contro muso (angolo < 30°), vicini, per qualche secondo (proposta 2 s);
  costa energia a entrambi. Lascia una "coppia registrata" nel recinto.
- **Rinascita** (fase B): chi muore rinasce subito in un punto a caso con i
  geni di un figlio (incrocio + mutazione) dell'ultima coppia registrata;
  se non c'è ancora nessuna coppia, da due esseri vivi a caso (da decidere:
  vedi "Dubbi").
- **Fase A**: chi muore rinasce con **lo stesso cervello di partenza** del
  suo tipo, senza ciò che aveva imparato. Nessuna evoluzione: si misura
  solo l'apprendimento.

## I sensi (fissi)

| senso | come i robot | proposta |
|---|---|---|
| occhi colorati | la telecamera | ventaglio davanti, 3 direzioni (sinistra, centro, destra) × 3 colori (rosso, giallo, blu) |
| distanza | infrarossi | 3 raggi davanti, dicono quanto è vicino un muro o un oggetto |
| contatto | | 1 |
| energia | livello della batteria | 1, interno |
| "mi guardano" | ricevitore infrarosso | 1: un altro essere vicino ha il muso verso di me |
| gusto, dolore | | 2, interni (ricarica buona, pila cattiva) |

Circa 17 canali. Nessun senso dell'acqua: qui non c'è acqua.

## Il corpo

Due motori di movimento come oggi (spinta e sterzo). Prendere la pila è
automatico a contatto se si è quasi fermi; accoppiarsi è automatico se le
condizioni sono vere (quindi il cervello deve *portarsi* in quelle
condizioni).

## I tre agenti

1. **Il cervello di oggi** (`rate` e `spike-ei`), con il set di sensi
   dell'arena (non ancora collegato: vedi "Stato").
2. **Attore-critico classico** (riferimento): legge i sensi, un critico TD
   continuo e un attore con lo stesso apprendimento collaudato dei manuali.
   Non biologico nei dettagli; dice quanto bene si può fare in questo mondo.
3. **Il cervello a sezioni** (`sezioni`), prima B, D, E, V con F fisso, poi
   C, L, il modulatore dell'incertezza, i valori innati.

Nella fase A tutti partono con un **istinto motorio minimo** uguale per
tutti: i programmi di movimento (avvicinati a ciò che vedi, allontanati,
esplora) sono innati in F; da imparare c'è **quale** colore inseguire e
**dove** andare, non come muoversi.

Il **guidato** (controllo del mondo, non un concorrente) sa il colore buono
e dov'è la zona: insegue il buono, evita il cattivo davanti, altrimenti va
verso la zona.

## Le misure

Per ogni inversione e ogni cambio di zona:

- **scelta**: pile buone sul totale, prima e dopo l'inversione;
- **tempo per adattarsi**: secondi dopo l'inversione finché la scelta torna
  sopra il 75% (o non ci torna);
- **tempo per trovare la zona nuova** dopo un cambio;
- **sopravvivenza**: durata della vita, morti di fame;
- **accoppiamenti** (fase B) e quanta energia ci si spende;
- **emersione** nel cervello a sezioni: selettività di B, uso delle unità di
  E, contenuto di D;
- **lesioni**: le stesse misure togliendo una sezione alla volta.

Il confronto principale: quanto il cervello a sezioni si avvicina
all'attore-critico, e quanto supera il cervello di oggi.

## Fasi di lavoro

1. **Il mondo in NumPy** (`evonets/sim/pile.py`), con un essere guidato da
   una regola scritta a mano ("vai verso le pile del colore buono") per
   controllare che il mondo sia giusto e tarare i numeri.
2. **Il mondo sulla GPU** (`evonets/gpu/pile.py`): centinaia di recinti in
   parallelo; prova che dà gli stessi risultati della versione NumPy.
3. **I sensi fissi** dell'arena e i **cervelli di oggi** che ci girano.
4. **L'attore-critico di riferimento**, e la prima tabella delle misure
   (fase A).
5. **Il cervello a sezioni** (B, D, E, V, F fisso): prima in NumPy con le
   prove di ogni sezione da sola, poi la **diagnosi con il gradiente**, poi
   in JAX.
6. **Fase A** completa: i tre agenti, misure e lesioni.
7. **C, L, incertezza, valori innati** nel cervello a sezioni; di nuovo
   fase A.
8. **Fase B**: accoppiamento, rinascita con incrocio, valori innati che
   evolvono.
9. **Impostazioni iniziali e visualizzatore**: l'arena e il cervello a
   sezioni selezionabili; nel visualizzatore le sezioni colorate.

Ogni fase finisce con i suoi test (`tests/test_pile.py`, poi quelli del
cervello) e una nota dei risultati in questo documento.

## Stato (2026-10-28)

Fatto: il mondo in NumPy (`evonets/sim/pile.py`), il guidato,
l'attore-critico, il cervello a sezioni (`evonets/net/sections.py`), le
prove e le misure (`evonets/evo/pile_test.py`), la fase B (accoppiamento,
rinascita da incrocio, valori innati che evolvono), i test
(`tests/test_pile.py`), la versione **GPU** (`evonets/gpu/pile.py`: stesso
algoritmo in JAX, molti passi per chiamata; numeri casuali diversi, medie
uguali: 200 recinti × 25 minuti in circa 45 s, contro circa 10 minuti per
100 recinti in NumPy; è il motore predefinito, `--motore numpy` per il
riferimento), la voce nel **menu** (Mondo = "arena delle pile": cervello,
fase, lesione, recinti, esseri per recinto) e la **schermata**
(`evonets/viz/pile_view.py`, docs/INTERFACCIA.md). Non ancora: i cervelli
di oggi nell'arena, la diagnosi con il gradiente, i salvataggi delle corse
delle pile.

**Taratura.** Con i numeri della prima proposta nemmeno il guidato
sopravviveva, e un essere che impara prendeva circa 1 pila al minuto: due
esperienze tra un'inversione e l'altra (90 s), troppo poche per chiunque.
Cambiati: inversione 300 s, zona 200 s, 8 pile, pile che scadono, presa
solo davanti al muso, si mangia fermi, consumo 0,15 + 0,25·spinta, dolore
proporzionato al danno, selezione dei programmi "appiccicosa" (senza, un
esploratore a caso cambiava programma ogni 0,25 s e non arrivava mai a una
pila).

## O, il valore degli oggetti, e S che lo legge (2026-10-29)

**Perché.** Il critico non distingueva le pile: per un allievo dal
comportamento quasi casuale "vedo una buona" e "vedo una cattiva" portano
quasi alla stessa cosa, e la parte che legge C impara dall'ultimo episodio
(solo il 52% dei critici preferiva la buona). Il valore delle cose va
imparato a parte, in un modo che non dipenda da quanto bene ci si comporta.

**O** (`sections.o_*`, docstring del modulo): per ogni caratteristica
percepibile, più lo sfondo, quanto piacere e quanto dolore porta (separati).
Impara alla fine di ogni episodio (piacere o dolore sentiti dal corpo), con
Rescorla-Wagner sulle caratteristiche che si avevano in mente prima
dell'esito (traccia di `trace_o` = 0,5 s, più D): le caratteristiche
competono, il passo è diviso per quanta traccia c'è (senza, i pesi
rimbalzavano) e moltiplicato per l'incertezza di Pearce e Hall. Il valore
letto è piacere × fame di adesso − dolore. Due neuroni di uscita ("quanto
vale ciò che guardo", "ciò che ho preso") vanno al critico e a E. Dal maestro
O impara con la stessa regola, sui suoi episodi. Lesione "O".

**S non impara più**: ogni posizione somma la salienza innata, il valore
(`sal_value`), l'importanza (piacere + dolore, `sal_import`: anche il
pericolo attira) e l'incertezza (`sal_unc`) che O dà dell'oggetto.

**Risultati** (200 recinti, sonda per essere su chi ha più di 1 minuto):

| | O preferisce la buona | buona − cattiva (mediana) |
|---|---|---|
| colore, 2 e 5 minuti | 84%, 90% | +0,12, +0,45 |
| 30 s dopo l'inversione | 13% | −0,41 |
| 5 minuti dopo | 89% | +0,45 |
| grandezza (il colore non conta) | 83%, 90%, 88% | come sopra; colore ~0,1 |

Con pile buone e cattive nella mappa, la probabilità dello sguardo va alle
buone: senza O 0,49 (quota di buone 0,48); con O 0,62-0,66 (quota 0,46).

Comportamento (15 minuti; nelle misure non entra il primo di ogni recinto):

| | pile buone | pile al minuto | morti di fame al minuto |
|---|---|---|---|
| con O | **58,2%** | 3,13 | 0,212 |
| senza O (lesione O) | 52,5% | 3,46 | 0,192 |
| con O, S che non lo legge (lesione S) | 52,5% | 3,33 | 0,199 |
| maestro fantasma, senza osservare | 58,1% | 3,33 | 0,174 |
| maestro fantasma e osservazione | 58,3% | **3,75** | **0,133** |

- Il guadagno passa dall'attenzione: senza che S legga O si torna al 52,5%.
- Osservare il maestro riduce l'errore di O sui suoi episodi (da 0,51 a
  0,20) e fa mangiare di più e morire di fame meno; la quota di buone non
  cambia.
- Resta lento il riadattamento dopo l'inversione: O si rovescia in qualche
  minuto, non in secondi.
- Con il critico che insegna (0,5) va ancora peggio che senza (53,8%): con
  il maestro il critico non è più forzato a 0,5.

## Il maestro: imparare il valore osservando (2026-10-28)

**Che cos'è.** Opzione "Maestro" nel menu (o `--maestro`): il primo essere di
ogni recinto decide come il guidato (sa quali pile sono buone e dov'è la
zona; ha la stessa fovea degli altri, che è solo ciò che guarda, senza pesi).
Si vede in oro con la corona; nelle misure non entra. Il tasto M lo toglie
(al suo posto nasce un giovane, che non si conta) o lo rimette. Con il
maestro il critico insegna (`critic_mix` 0,5).

**Fantasma o visibile.** Il maestro "fantasma" (quello di partenza) vive
nello stesso recinto (stesse pile, zona, inversioni) ma nessuno lo vede o lo
urta, e la pila che mangia sparisce solo per lui (`ghost_hide`): non toglie
niente agli allievi. Quello "visibile" è un compagno vero, e si mangia le
pile buone.

**Il meccanismo** (`sections.watch`): a ogni passo il critico di ogni
allievo fa un passo TD anche sulla vita del maestro del suo recinto, come se
gli guardasse dentro. La situazione del maestro (fovea, energia) viene
codificata con le connessioni dell'allievo (B, D, C, che qui non imparano);
ciò che il maestro ha sentito (piacere, dolore, energia spesa) viene valutato
con i geni dell'allievo. I pesi del critico sono gli stessi, la traccia è
propria; la velocità è `eta_v` × `observe` (gene). Dall'osservazione imparano
solo le sinapsi che dicono che cos'è l'oggetto (identità nella fovea e in D,
il codice C). Nella schermata: "quanto imparano ancora da lui", la media di
|δ| sulla sua vita, con la sua curva.

**Due correzioni trovate provando.**
- I due apprendimenti cambiano gli stessi pesi: la previsione del passo
  prima va ricalcolata con i pesi di adesso, altrimenti nascono sorprese
  finte che si alimentano (valori a 200 invece che a 12).
- Con tutti gli ingressi il critico imparava combinazioni tipiche della vita
  del maestro, che nella vita dell'allievo davano il segno sbagliato:
  dall'osservazione si impara il valore degli oggetti, non quello della
  situazione di un altro.

**Risultati** (100-200 recinti, 10 minuti):

| | pile buone (allievi) | errore sulla vita del maestro |
|---|---|---|
| senza maestro | 54-55% | — |
| maestro, senza osservare | 34-38% | 0,26 |
| maestro e osservazione | 35% | 0,18 |

- La sola presenza del maestro abbassa gli allievi: si mangia lui le pile
  buone (concorrenza).
- L'osservazione funziona sulla vita del maestro (l'errore scende), ma non
  cambia il comportamento.
- Con il maestro fantasma (niente concorrenza; 200 recinti, 10 minuti,
  critico 0,5): senza osservare 51,2%, osservando 50,7%, osservando tre
  volte di più 51,8%. L'errore sulla vita del maestro scende solo del 10%
  circa (dall'osservazione imparano solo le sinapsi dell'identità).
- Il motivo è a monte: anche senza maestro solo il 52% dei critici dà più
  valore alla pila buona che alla cattiva (al 65% se la parte che legge C
  impara 10 volte più piano, ma con differenze minime). Il critico, così
  com'è, non distingue in modo affidabile le pile.

## Attenzione generale: mappa di priorità e oggetti con più caratteristiche (2026-10-28)

**Il mondo.** Le pile sono rosse o gialle **e** grandi o piccole, a caso; i
compagni sono blu e si muovono. Ogni oggetto ha le sue caratteristiche
(`sim.pile.FEATURES`: rosso, giallo, blu, grande, piccolo, si muove).
L'opzione `good_by` decide che cosa rende buona una pila: il colore (come
prima) o la grandezza (allora il colore non conta). Nel menu: "Pile buone
per"; da riga di comando `--buone grandezza`.

**L'attenzione (S) è una mappa di priorità fatta di posizioni**, come il
collicolo superiore: 9 neuroni-direzione che coprono il campo visivo, ognuno
con l'oggetto più vicino nella sua direzione; un oggetto nuovo accende la
sua posizione, non nasce nessun neurone. Ogni neurone somma la salienza
innata, le caratteristiche dell'oggetto per **pesi imparati uguali in ogni
posizione** (si impara "ciò che ha questa caratteristica", ovunque sia) e
l'**incertezza** (Pearce e Hall: le caratteristiche che di recente hanno dato
sorprese attirano). I pesi imparano con δ, l'incertezza con |δ|; i valori di
partenza sono geni. La fovea porta tutte le caratteristiche dell'oggetto
attento, e C le legge come identità.

**Prove controllate** (200 recinti, preferenza imparata per la
caratteristica buona del proprio recinto, esseri oltre 1 minuto):

| | grandezza | colore |
|---|---|---|
| pile buone per grandezza, dopo 5 minuti | **+0,47** | +0,01 |
| subito dopo l'inversione | −0,53 | 0,00 |
| 5 minuti dopo l'inversione | **+0,34** | −0,01 |
| pile buone per colore, dopo 5 minuti | −0,01 | **+0,49** |

L'attenzione impara la caratteristica che conta, qualunque sia, e ignora le
altre; dopo un'inversione si riadatta (in qualche minuto). I compagni
(vistosi, in movimento, ma inutili qui) non la catturano.

## Il critico rifatto, e perché per ora non insegna (2026-10-28)

**Cosa è cambiato**
- **Valore legato al corpo**: il piacere di una pila va da 0,2 (sazi) a 1,8
  (affamati) volte il suo valore; l'energia spesa (muoversi, mordere) dà un
  dispiacere nella stessa scala (spendere l'energia di una pila dispiace
  quanto mangiarne una piace). Il mondo ora dice anche quanta energia si è
  spesa a ogni passo.
- **Tre critici con orizzonti diversi** (2, 8, 30 s di partenza, geni): in
  natura zone e neuroni a dopamina diversi scontano il futuro in modo
  diverso. Un orizzonte unico era legato ai tempi di questo mondo.
- **Ingressi stabili**: fovea, D, C (con velocità propria, alta: il valore di
  un oggetto in un colpo), energia, B con peso 0,2.
- **Quanto pesa il critico** nella sorpresa che insegna è un gene
  (`critic_mix`).

**Cosa abbiamo trovato**
- Con i comportamenti di chi impara il valore **non è prevedibile**: anche la
  migliore previsione lineare possibile dagli stessi ingressi, calcolata a
  posteriori, ha correlazione 0,004 con ciò che arriva davvero. Dopo aver
  guardato una pila buona la si mangia entro 8 s solo l'8% delle volte
  (cattiva: 5%).
- Con il comportamento del guidato sì: correlazione 0,375 dalla sola fovea
  e dall'energia, e una pila buona guardata è mangiata entro 8 s nel 46% dei
  casi. Gli ingressi del critico sono quelli giusti; manca un comportamento
  abbastanza coerente da essere previsto.
- Quindi oggi ogni peso del critico peggiora (0: 59,5% di pile buone, 69%
  prima dell'inversione; 0,25: 54%; 1: 52,5%). Il valore legato alla fame
  invece aiuta (senza critico si passa dal 55% al 59,5%). Il peso di
  partenza è 0: il critico impara, ma non insegna finché la fase B non lo
  alza.

## Le costanti diventano geni; figli da un genitore solo (2026-10-28)

Tutte le costanti del cervello che avevo scelto a mano (circa 50: tempi,
velocità, soglie, pesi della salienza innata, forza dell'arbitraggio, ogni
quanto si decide...) sono geni (`sections.Genes`), una per essere. Nella fase
A valgono i valori di partenza per tutti; nella fase B le tara l'evoluzione.
Restano fissi solo la struttura (quanti neuroni e connessioni) e i limiti di
sicurezza. Ogni essere decide al suo ritmo (`decide_every` è un gene).

**Figli senza mescolamento**: il figlio ha il genoma intero di uno dei due
genitori della coppia (a caso), più la mutazione. Prima ogni gene veniva da
un genitore a caso, e questo spezzava i blocchi (un prototipo di B metà del
padre e metà della madre).

## Arbitraggio, genoma con pesi e connessioni, salvataggi (2026-10-28)

**Arbitraggio (U).** Con molte sorprese (k = U/3) il voto di C conta fino
a 3 volte, quelli di E ed L fino a metà, la salienza imparata fino a 0,3
volte (ci si guarda attorno); U non accelera più E e non aggiunge caso
(accelera solo C). La lesione "R" rimette com'era. U ora conta solo le
sorprese vere (sopra la soglia): con la media di tutto δ non si accorgeva
delle inversioni.

| | prima dell'inversione | 0-15 s | 15-30 | 30-60 | 60-90 |
|---|---|---|---|---|---|
| sezioni (arbitraggio) | 62,5% | 39,3% | 42,8% | **45,8%** | **50,0%** |
| sezioni senza arbitraggio (R) | 63,5% | 39,5% | 41,7% | 43,1% | 47,1% |

Aiuta poco (circa +3 punti nel recupero), perché le sorprese sono frequenti
anche prima dell'inversione (si sbaglia ancora una pila su tre): l'inversione
non produce un'ondata riconoscibile, e U resta intorno a 0,7. Servirà quando
le abitudini saranno buone.

**Genoma con pesi e connessioni.** Oltre ai parametri, ai valori innati e
ai pesi di F, il genoma di ogni cervello contiene i suoi **pesi di
partenza** (prototipi di B, E, V, salienza) e le **connessioni** di C (un
gene per connessione possibile: c'è se supera 0,5). Alla nascita il
cervello riparte da lì; il figlio eredita i valori di partenza dei
genitori (incrocio e mutazione), non ciò che hanno imparato. C, L, D e le
altre memorie non si ereditano. Fase A: ognuno ha i suoi valori di partenza
(a caso dove serve) e rinasce sempre da quelli.

**Salvataggi** (`evonets/evo/pile_saves.py`): mondo, cervelli (genoma e
tutto ciò che si è imparato), serie e contatori per essere; la stessa forma
per GPU e processore. Ctrl+S nella schermata, e all'uscita se ci sono
novità; nel menu "Riprendi da" li riconosce (righe dell'arena delle pile
compilate dal file). Alla ripresa valgono le regole attuali.

**Le reti una a una.** `alone`: ogni rete da sola nel suo recinto, tutte
insieme, con ciò che ha imparato; contatori per rete (buone, cattive,
morti). Nella schermata il tasto P (Esc torna alla corsa), nel menu
"Esplora" su un salvataggio delle pile, da riga di comando
`--prova salvataggio.npz --minuti 10` (le migliori e le peggiori).

## Corpo nuovo: sensi elementari, morso, F come circuito di neuroni (2026-10-28)

**Sensi** (31 canali, `sim.pile.sense`, la stessa funzione per NumPy e
GPU): occhi per colore e direzione (sinistra, centro, destra) per pile
rosse e gialle, compagni (blu) e i **quattro muri, ognuno di un colore**
(nord verde, est viola, sud arancio, ovest bianco: punti di riferimento);
tre raggi di distanza; **tre sensori d'urto** (sinistra, davanti, destra);
energia, "mi guardano", gusto, dolore.

**Motori elementari**: girare, avanti/indietro (indietro a metà velocità),
**mordere**. Si mangia solo mordendo una pila davanti al muso, vicina,
andando piano; tenere la bocca aperta costa 0,1 di energia al secondo.
Urtare una pila non vuol dire mangiarla. E decide se mordere (una "testa" in
più, sì o no a ogni decisione); C ed L votano anche sul morso.

**F è un circuito di neuroni innati** (`sim.pile.motor_net`): 14 ingressi
(il programma di E combinato con dove sta l'oggetto attento, la deriva del
vagare, il riflesso dei muri, una costante) e 4 neuroni motori (gira a
sinistra, a destra, avanti, indietro). I pesi di partenza riproducono i
programmi di prima; non imparano nella vita; sono geni (fase B).

**Risultati** (GPU, 200 recinti × 6 esseri, 25 minuti):

| agente | pile/min | buone | morti di fame/min | prima dell'inversione | 0-15 s dopo | 15-30 | 30-60 | 60-90 |
|---|---|---|---|---|---|---|---|---|
| guidato | 2,39 | **94%** | 0,000 | 93% | 97% | 93% | 94% | 93% |
| attore-critico | 4,80 | 50% | 0,10 | 51% | 48% | 50% | 50% | 49% |
| sezioni | 2,82 | 56% | 0,20 | 65% | 37% | 43% | 44% | 47% |
| sezioni senza C | 3,66 | 53% | 0,18 | 61% | 40% | 42% | 44% | 46% |
| sezioni senza S | 2,93 | 53% | 0,18 | 55% | 47% | 48% | 48% | 51% |
| sezioni senza B / D / L / U / V | 2,16-3,74 | 54-57% | 0,18-0,21 | 62-68% | 34-40% | 42-45% | 43-46% | 47-49% |

- Il morso come decisione toglie il limite del corpo: il guidato passa
  dall'80% al 94% (prima il 95% delle sue cattive erano urtate per strada).
- L'attore-critico morde tutto (4,8 pile al minuto, a caso).
- Nel cervello a sezioni ora **C conta** (senza C: 61% invece di 65% prima
  dell'inversione); S resta la parte più importante; dopo l'inversione si
  disimpara ancora lentamente, e si muore di fame più dell'attore-critico
  (mordono meno).

## C legge che cosa si guarda, D ricorda che cosa si è preso (2026-10-28)

**Diagnosi.** Dopo un'inversione C non cambiava idea: ogni incontro con lo
stesso colore accendeva neuroni di C diversi, quindi ciò che si imparava
una volta non valeva la volta dopo. Tre cause, corrette una per una:

1. C leggeva tutto lo stato di B e D (muri, energia, compagni...). Ora
   legge **che cos'è** l'oggetto guardato (il colore nella fovea) e quello
   ricordato in D (peso 0,5), più poco contesto da B (peso 0,1); non dove
   sta (vicinanza e lato cambiano di continuo).
2. "Vincono i 4 più attivi": tra una decina di neuroni pari merito vinceva
   il caso. Ora si accendono **tutti quelli sopra soglia**: lo stesso oggetto
   accende sempre gli stessi (per essere: circa 11 per colore, 1,6 in comune
   tra rosso e giallo).
3. C accumulava lentamente e senza limite: dopo 5 minuti di giallo buono un
   dolore valeva un ventesimo. Ora è veloce e satura (±0,3 per neurone):
   conta l'ultima esperienza.

D ora tiene **l'ultimo oggetto guardato** (una copia della fovea, scritta
quando entra un oggetto nuovo), anche quando non si vede più.

Con più neuroni accesi alcune regole sono diventate instabili (numeri che
esplodevano): ora il passo di C è diviso per i neuroni accesi, il critico è
normalizzato sulla traccia, δ satura a ±3, i pesi hanno un tetto.

**Effetto sul singolo essere** (dopo il suo primo dolore dopo
l'inversione): il voto di C per avvicinarsi al colore ora cattivo scende da
+1,33 a +0,27, la probabilità di avvicinarsi dal 90% al 64%. Non si
ribalta: E, L e la salienza, lenti, spingono ancora verso il vecchio colore.

**Nelle medie** (200 recinti, 25 minuti) l'effetto è piccolo: sezioni 60%
prima dell'inversione, 39% subito dopo, 47% dopo 90 s; senza C 61%, 40%,
45%. Il guidato resta all'80% per il limite del corpo (95% delle sue pile
cattive sono prese per strada mentre punta una buona).

## Attenzione, tracce strette, soglia sulla sorpresa (2026-10-28)

**Cosa è cambiato.**

- **F non sa più i colori.** I programmi sono *avvicinati*, *allontanati*,
  *lascialo*, *esplora*, e agiscono sull'**oggetto attento**, di cui F sa
  solo dove sta.
- **Attenzione.** Ogni essere è concentrato su un oggetto (pila o
  compagno), scelto per salienza: innata (vicinanza, novità, un po' di
  inerzia, inibizione del ritorno per 3 s dopo *lascialo*) più imparata (un
  peso per colore, che δ cambia: attenzione guidata dal valore). La scelta
  ha un po' di caso. Nel guidato la salienza bara (sa il colore buono).
- **Fovea**: in A, 7 canali sul solo oggetto attento (colore, vicinanza,
  lato, "nuovo").
- **Tracce strette** (solo nel cervello a sezioni): E e il cancello di D si
  marcano solo da neuroni accesi sopra 0,5.
- **Soglia adattiva**: insegna solo la parte di |δ| sopra la sorpresa media
  degli ultimi minuti; il critico impara da tutto δ.
- Nuova lesione "S": salienza solo innata.

**Risultati** (GPU, 200 recinti × 6 esseri, 25 minuti):

| agente | pile/min | buone | morti di fame/min | prima dell'inversione | 0-15 s dopo | 15-30 | 30-60 | 60-90 |
|---|---|---|---|---|---|---|---|---|
| guidato | 2,61 | 80% | 0,000 | 78% | 93% | 80% | 78% | 78% |
| attore-critico | 2,24 | 50% | 0,14 | 51% | 49% | 50% | 50% | 49% |
| sezioni | 2,08 | 53% | 0,18 | 64% | 38% | 39% | 42% | 45% |
| sezioni senza B / D / C / L / V | 1,92-2,21 | 52-54% | 0,18 | 63-64% | 36-41% | 39-40% | 41-42% | 44-46% |
| sezioni senza U | 2,75 | 54% | 0,14 | 60% | 41% | 44% | 46% | 48% |
| sezioni senza S (salienza imparata) | 2,00 | 52% | 0,17 | 55% | 44% | 44% | 47% | 49% |

- L'attore-critico, con la fovea al posto dei programmi colorati, **non
  distingue più i colori** (50%).
- Il cervello a sezioni sì (64% prima dell'inversione), e B ora porta più
  informazione sul colore (0,44 invece di 0,26): la fovea l'ha resa utile.
- Ma la scelta viene **soprattutto dalla salienza imparata** (senza S: 55%):
  imparano a *guardare* il colore buono più che a lasciare quello cattivo.
  B, C, D, L ancora non cambiano niente se tolte.
- Dopo l'inversione si disimpara ancora lentamente, e si muore di più di
  fame: si prendono meno pile (2,1 al minuto contro 2,9 di prima).

## Risultati dopo la correzione del riflesso (fase A, GPU, 200 recinti × 6 esseri, 25 minuti)

**Difetto trovato (2026-10-28):** il riflesso dei muri girava verso il lato
"più libero"; in un angolo i due lati si scambiavano a ogni passo e l'essere
oscillava per sempre spingendo contro i muri: il 76-79% degli esseri che
imparano era fermo in un angolo (il guidato no: la sua guida verso la zona
scavalcava il riflesso). Ora il verso della fuga, una volta scelto, resta
finché davanti è libero: negli angoli 0-1%. **I risultati sotto la tabella
successiva (quelli di prima) sono falsati da questo difetto.**

| agente | pile/min | buone | morti di fame/min | prima dell'inversione | 0-15 s dopo | 15-30 | 30-60 | 60-90 |
|---|---|---|---|---|---|---|---|---|
| guidato | 2,56 | 77% | 0,001 | 75% | 90% | 79% | 76% | 75% |
| attore-critico | 3,32 | 52% | 0,09 | 61% | 37% | 39% | 43% | 46% |
| sezioni | 2,90 | 52% | 0,11 | 61% | 34% | 38% | 43% | 45% |
| sezioni senza B / D / C / L / V | 2,85-2,89 | 52-53% | 0,10-0,11 | 61-62% | 35-36% | 37-40% | 40-43% | 43-46% |
| sezioni senza U | 3,32 | 55% | 0,09 | 63% | 38% | 39% | 45% | 48% |

- Ora trovano molte pile (più del guidato), ma scelgono il colore quasi a
  caso: prima dell'inversione 61%, subito dopo 34-37% (inseguono il colore
  che era buono), dopo 90 s ancora 45%. Imparano, ma disimparano troppo
  lentamente.
- I pesi sono stabili (nessuna esplosione: |w_e| massimo circa 2, |u_c|
  circa 2, |w_v| circa 0,4), ma il critico prevede poco (δ resta quasi la
  ricompensa grezza) e B porta poca informazione sul colore (0,26): i suoi
  prototipi si specializzano su ciò che è frequente (muri, energia,
  contatto), non su ciò che conta. Per questo C e D, che leggono B, non
  servono: nessuna lesione cambia qualcosa; U peggiora.

## Primi risultati (fase A, 100 recinti × 6 esseri, 25 minuti; falsati dal difetto del riflesso)

| agente | pile/min | buone | morti di fame/min | prima dell'inversione | 0-15 s dopo | 15-30 | 30-60 | 60-90 |
|---|---|---|---|---|---|---|---|---|
| guidato (sa tutto) | 2,65 | 77% | 0,001 | 75% | 89% | 80% | 76% | 75% |
| attore-critico | 1,06 | 60% | 0,11 | 65% | 39% | 41% | 50% | 54% |
| sezioni | 0,94 | 58% | 0,12 | 66% | 38% | 41% | 44% | 50% |
| sezioni senza B | 0,92 | 58% | 0,12 | 66% | 39% | 37% | 45% | 51% |
| sezioni senza D | 0,98 | 58% | 0,12 | 66% | 35% | 37% | 43% | 49% |
| sezioni senza C | 0,88 | 56% | 0,12 | 63% | 39% | 37% | 41% | 47% |
| sezioni senza L | 0,90 | 59% | 0,12 | 63% | 37% | 43% | 47% | 52% |
| sezioni senza U | 1,20 | 64% | 0,11 | 68% | 36% | 45% | 53% | 56% |
| sezioni senza V | 1,11 | 61% | 0,11 | 67% | 39% | 42% | 48% | 56% |

- **Si impara il colore dentro la vita**: prima di un'inversione circa due
  pile su tre sono buone, subito dopo si scende sotto il 40% (si insegue il
  colore che era buono: perseverazione), poi si risale lentamente; in 90 s
  nessuno torna al 75%.
- **Il cervello a sezioni non fa meglio dell'attore-critico**, e nessuna
  lesione peggiora in modo chiaro; senza U (e senza V) va un po' meglio. In
  questo compito le sezioni in più non servono ancora: il colore si lega
  subito al boccone (la pila presa è davanti agli occhi un attimo prima),
  quindi non serve memoria; e manca il senso dello spazio per il "dove"
  (vedi sotto). B porta informazione sul colore (circa 0,5), ma togliendolo
  non cambia niente.
- **Il collo di bottiglia è trovare le pile**: chi impara ne prende una al
  minuto, il guidato 2,65, e muore di fame dopo circa 9 minuti in media.

## Dubbi ancora aperti

0. **Dove: manca un senso dello spazio.** Gli esseri non hanno posizione,
   bussola o punti di riferimento: imparare *dove* è la zona è quasi
   impossibile, e il "dove" pesa molto sulla sopravvivenza (il guidato, che
   lo sa, prende 2,6 pile al minuto; chi impara circa 1). Possibilità: una
   bussola (come il senso magnetico che esiste già nel progetto), fari di
   colore diverso negli angoli, o un programma "verso il faro".

1. **Fase B senza coppie:** chi muore prima che ci sia stato un
   accoppiamento, da chi rinasce? Proposta: da un incrocio di due esseri
   vivi scelti a caso, così la popolazione non si blocca.
2. **I numeri** (dimensione del recinto, tempi di inversione e di zona,
   energia di una pila, costo dell'accoppiamento) sono proposte da tarare
   nella fase 1, con l'essere guidato a mano.
3. **Il riposo** per il consolidamento verso L: da decidere quando arriva L.
