# Evolutionary_nets

Vita artificiale e neuroevoluzione in un'arena 2D. Una popolazione di agenti,
ognuno guidato da una propria rete neurale, vive in un ambiente con zone di
veleno in movimento. Non c'è un punteggio di fitness: **si riproduce solo chi
completa il compito**, e l'energia decide chi muore prima di riuscirci.

Il progetto è una piattaforma per sperimentare: il compito è sostituibile e
pensato per diventare via via più arduo, e la forma della rete non è fissata ma
evolve. Anche i sensi evolvono: ogni agente nasce con pochi **recettori
universali**, cellule sensoriali che possono diventare sensibili a luce,
sostanze, vibrazioni, temperatura, campo magnetico o al proprio corpo. Non c'è
un elenco di sensi né un numero massimo di neuroni: ogni cosa ha un costo in
energia, e la selezione decide cosa vale la pena tenere. Lo stesso vale per i
tassi di mutazione, che sono geni. Il progetto completo è in
[docs/SPEC_FASE_C.md](docs/SPEC_FASE_C.md).

## Avvio

Serve [uv](https://docs.astral.sh/uv/). Le dipendenze (NumPy, Numba, JAX con
CUDA, pygame-ce) si installano da sole al primo comando: la prima volta JAX
scarica circa 1 GB di librerie NVIDIA.

**Motore di simulazione.** Di default la simulazione gira sul motore compilato
con JAX (`engine = "gpu"`): sulla scheda video NVIDIA se c'è, altrimenti sulla
CPU. Con una RTX 5070 una generazione è circa 20 volte più veloce del motore
originale, e più semi insieme circa 60 volte. Il motore originale in NumPy
resta (`--engine numpy`) come riferimento: i test verificano che i due diano
gli stessi risultati. Ogni nuova forma di rete richiede una compilazione di
qualche secondo; le compilazioni si conservano in `~/.cache/evonets-jax`.

```
uv run main.py                        # interfaccia grafica: menu, corsa, esploratore delle reti
uv run main.py --evolve 300           # 300 generazioni nel terminale, senza grafica, poi salva
uv run main.py --evolve 300 --select 5  # ... poi i sopravvissuti rifanno la prova 5 volte, senza figli
uv run main.py --evolve 300 --exam-every 100  # esame fisso ogni 100 generazioni (default 50; 0 = mai)
uv run main.py --exam runs/pop.npz    # esame fisso dei più anziani di un salvataggio, aggiunto al file
uv run main.py --evolve 500 --seeds 4 # 4 semi insieme sulla GPU: runs/pop-s<seme>.npz
uv run main.py --engine numpy ...     # motore originale (più lento)
uv run main.py --bench                # misura la velocità della simulazione
uv run pytest                         # test
```

Senza `--evolve` né `--bench`, le opzioni `--net rate|spike-ei`, `--agents N`,
`--seed N`, `--load FILE` e `--save FILE` precompilano il menu. Senza `--seed`
il seme è casuale e viene stampato, così la corsa si può rifare.

## L'interfaccia

Una finestra sola con quattro schermate: menu, corsa, esploratore delle reti e
banco di prova. **Una corsa aperta continua in sottofondo** anche mentre si
guardano l'esploratore o il banco di prova; lì `B` la mette in pausa o la fa
ripartire, e una riga in basso dice a che generazione è arrivata.

### Il menu

Si sceglie la corsa e la si lancia con Invio (o il pulsante).

| Riga | Significato |
|---|---|
| Modalità | **veloce**: si parte dalla pagina principale (statistiche e reti), l'arena si apre con `F`; **massima velocità**: solo statistiche, ridisegnate due volte al secondo, tutto il resto del tempo all'evoluzione; **dal vivo**: si parte con l'arena a schermo. Il risultato è lo stesso in tutte e tre |
| Rete | continua o a impulsi con neuroni eccitatori/inibitori |
| Individui | dimensione della popolazione |
| Casualità | **seme casuale** (uno nuovo a ogni corsa) o **seme fisso** (quello della riga Seme). In entrambi i casi il seme usato compare nel menu, in cima alla corsa e nel salvataggio |
| Seme | il seme fisso; con il seme casuale mostra quello appena estratto |
| Generazioni | dopo quante fermarsi e salvare; 0 = senza limite |
| Esame ogni | ogni quante generazioni fare l'esame fisso ai 20 più anziani (vedi "L'esame fisso"); 0 = mai. Default 50 |
| Selezione finale | finite le generazioni, quante prove fanno ancora i sopravvissuti, senza figli (vedi "La selezione finale"); 0 = niente. Si attiva solo con Generazioni > 0 |
| Riprendi da | un salvataggio in `runs/` da cui continuare; rete, individui e tutte le costanti vengono da lì |
| Salva in | dove salvare; per una corsa nuova propone un nome con data e ora, così non sovrascrive nulla |

Con un salvataggio scelto, "Esplora reti e genomi" lo apre nell'esploratore
senza far ripartire l'evoluzione; da lì si possono anche provare le sue reti.

Su/Giù o Tab scelgono la riga, Sinistra/Destra (o il clic) cambiano il valore,
numeri e percorso si scrivono da tastiera. Nei numeri le cifre si aggiungono in
coda: per scriverne uno nuovo si cancella prima con Backspace.

### La corsa: la pagina principale

Mentre l'evoluzione va avanti, la pagina è divisa in tre.

**A sinistra, le statistiche.** In alto i numeri chiave: successi dell'ultima
generazione, media delle ultime 10, record (con la generazione in cui è
arrivato), il più anziano (il suo numero e quante prove ha vinto di fila),
neuroni e collegamenti medi, durata dell'ultima generazione, recettori usati e
accesi in media (usato = ha almeno un collegamento in uscita: un recettore
acceso ma scollegato si paga senza servire).
In cima alla pagina una riga dice anche in che generazione è nato. **Il più
anziano è cliccabile** (riquadro bordato o riga sottolineata in cima, oppure
`O`): si apre la sua rete. Sotto, quattro grafici per generazione:

- successi (linea sottile: ogni generazione; spessa: media delle ultime 10);
- dimensione media delle reti: neuroni interni, collegamenti e recettori usati,
  con sopra quanti recettori usati ha in media ogni individuo per ciascun
  fenomeno (specializzati: un fenomeno pesa almeno l'80% della sensibilità) e
  quanti sono ancora generalisti;
- sopravvissuti e generazioni vinte di fila dal più anziano: quanto a lungo un
  genitore continua a vincere;
- quota di neuroni inibitori (reti a impulsi) o durata della generazione (reti
  continue). Appena c'è un esame fisso, questo grafico lascia il posto ai
  punteggi d'esame (media e migliore, un punto per esame), con sopra i
  risultati dell'ultimo esame prova per prova.

In cima alla pagina c'è anche il registro dell'ambiente (invariato, oppure
l'ultimo cambio di costante) e, se c'è, l'avanzamento dell'esame in corso.

**Al centro, le reti.** Tre elenchi (`1`, `2`, `3` o `Tab`):

- **campioni**: uno per ogni generazione già conclusa con almeno un vincitore.
  Il campione **non è scelto a caso**: è il vincitore più anziano di quella
  generazione (quello che ha vinto più prove di fila) e, a parità di età, il
  nato prima (numero più basso). Accanto, la percentuale di successi di quella
  generazione. Si aggiorna da solo: se è scelta la riga più in alto, si segue
  l'ultimo campione arrivato;
- **sopravvissuti**: i vincitori dell'ultima generazione conclusa, che ora
  corrono di nuovo (dopo una selezione finale: i rimasti);
- **tutti**: tutta la popolazione in gara, sopravvissuti più figli appena nati
  (età 0, non ancora messi alla prova).

Sopravvissuti e tutti sono una fotografia: `R` la rifà. La rete scelta è
disegnata accanto (vedi "Come si legge una rete").

In cima alla pagina una riga dice sempre quanti sono i sopravvissuti e quanti i
figli nuovi della generazione in corso.

Sopra ogni elenco un'intestazione dice cosa c'è in ogni colonna:

| Colonna | Significato |
|---|---|
| gen | generazione di cui è campione |
| succ | percentuale della popolazione che ha completato il compito in quella generazione |
| num. | numero dell'individuo (vedi punto 18) |
| età | prove vinte di fila |
| neur | neuroni interni esistenti |
| colleg | collegamenti effettivi |

**A destra, il neurone toccato**: chi è (senso, interno eccitatorio o
inibitorio, motore), i suoi geni propri e **tutti** i suoi collegamenti, in
ingresso ("riceve da") e in uscita ("manda a"), ciascuno con il peso e se eccita
o inibisce, dal più forte. Se sono tanti si scorrono con la rotella.

| Tasto | Azione |
|---|---|
| Su/Giù, clic | scegli la rete nell'elenco |
| clic su un neurone | lo fissa nel pannello di destra (un altro clic lo libera); passarci sopra basta per vederlo |
| `T` | **prova da sola la rete scelta** (banco di prova) |
| `A` | **mette in arena tutti i sopravvissuti insieme** (banco di prova) |
| `Tab` / `1` / `2` / `3` | campioni / sopravvissuti / tutti |
| `Invio` | a esperimento concluso, riapre il riepilogo |
| `O`, o clic sul più anziano | mostra la rete del più anziano (nell'arena: lo segue) |
| `P` | rami morti in grigio o nascosti |
| `I` | esploratore a tutto schermo, con il genoma |
| `F` | apre l'arena (la corsa continua) |
| `E` | **esame fisso adesso** dei 20 più anziani (gira su altri core; va avanti anche in pausa) |
| `Z` | massima velocità sì/no: nasconde l'esploratore, ridisegna due volte al secondo e dà il resto del tempo all'evoluzione |
| `Spazio` | pausa |
| `S` | salva la popolazione |
| `Esc` | torna al menu, salvando se ci sono generazioni nuove (lo stesso alla chiusura della finestra) |

La pagina principale lascia all'interfaccia una parte del tempo per restare
reattiva; con `Z` (o la modalità "massima velocità" nel menu) la corsa va più
veloce, con lo stesso risultato.

### La corsa: l'arena

`F` apre l'arena in turbo senza fermare l'evoluzione; `F` di nuovo torna alla
pagina principale. A sinistra gli agenti, a destra la rete dell'agente seguito
e, sotto, il grafico dei successi. In alto a sinistra lo stato della generazione,
il più anziano in gara e l'agente seguito (che all'apertura è proprio il più
anziano).

| Nell'arena | Significato |
|---|---|
| Cerchi verdi sfumati | veleno: drena energia, poco al bordo e molto al centro |
| Cerchio azzurro | bersaglio |
| Agente giallo | quello che stai seguendo, con i suoi recettori: per ognuno due linee grigie ai bordi del suo campo e una linea al centro, più lunga e più accesa quanto più risponde; il colore dice a cosa è sensibile (vedi sotto) |
| Anello bianco | sopravvissuto di una generazione precedente |
| Bianco pieno | ha completato il compito |
| Grigio | morto |

| Nel pannello della rete | Significato |
|---|---|
| Colonna sinistra | i recettori accesi, all'altezza in cui il loro nervo entra nel cervello; il bordo dice a cosa sono sensibili: giallo luce, verde sostanze, arancio meccanica, rosso temperatura, viola magnetismo, grigio interno, bianco generalista |
| Centro | i neuroni interni di quell'individuo, dove stanno nel cervello |
| Destra | i motori |
| Pallino arancione / azzurro | neurone attivo in positivo / negativo |
| Linea arancione / azzurra | collegamento che eccita / inibisce; si illumina quando passa segnale |

| Tasto | Azione |
|---|---|
| `+` / `-` | velocità da x1 a x16 |
| `T` | turbo |
| clic | segui un altro agente |
| `M` | guida manuale dell'agente seguito (WASD o frecce) |
| `V` | recettori disegnati sì/no |
| `C` | elenco delle costanti; `Tab` sceglie, `,` `.` regolano |
| `O` | segui il più anziano: il pannello a destra mostra la sua rete |
| `F`, `I`, `Spazio`, `S`, `Esc` | come nella pagina principale |

### Come si legge una rete

Vale per la pagina principale, per l'esploratore e per il pannello della rete
nell'arena. Ogni neurone è disegnato **dove sta davvero nel cervello** (vedi il
punto 8): recettori sul bordo sinistro, motori sul destro, neuroni interni nella
loro posizione dentro il riquadro, che è il cervello. Un collegamento lungo sul
disegno è lungo anche nel cervello, e costa energia in proporzione. Un
cerchietto sopra un neurone è un collegamento con se stesso.

Colore = segno (arancio eccita, azzurro inibisce), intensità e spessore =
forza. Nelle reti a impulsi il bordo di un neurone interno dice il suo tipo. In
grigio i **rami morti**: neuroni e collegamenti che non arrivano a nessun
motore e quindi non cambiano il comportamento. Passando su un neurone (o
cliccandolo) si accendono solo i suoi collegamenti, **in entrata e in uscita**,
con una **freccia a metà** che ne indica il verso. Il colore è sempre l'effetto
di chi trasmette: arancio se eccita, azzurro se inibisce. Nelle reti a impulsi
lo decide il tipo del neurone di partenza, quindi le uscite di un neurone
inibitorio sono tutte azzurre e in entrata si vede chi lo eccita e chi lo
inibisce. Anche i collegamenti dei rami morti prendono il colore, solo più
tenue. Il pannello dei dettagli dice la posizione del neurone, a
quanti passi è dai recettori e, per ogni collegamento, peso, effetto e
lunghezza. Per un recettore mostra tutti i suoi geni: dove sta sul corpo, dove
guarda, l'apertura e, per ogni fenomeno, la sensibilità g, la quota sul totale
e la sintonia (centro ± larghezza). In cima alla rete ci sono il cablaggio e
quanta energia costano in tutto corpo e cervello al secondo.

### L'esploratore delle reti

`I` apre a tutto schermo lo stesso elenco e la stessa rete della pagina
principale, più il **genoma** come matrice: una riga per blocco (neurone
interno o motore), una colonna per sorgente (recettori | interni | motori). Colore =
peso effettivo, grigio = gene latente (collegamento scritto nel genoma ma con un
capo spento), righe scure = neuroni spenti. Passando col mouse su una cella se
ne legge il valore; il clic su una riga sceglie quel neurone.

`T` prova la rete scelta, `X` scrive il genoma completo (blocco per blocco, con i
geni latenti) in un file di testo in `runs/genomi/`, `Esc` torna indietro. Il
campione di ogni generazione finisce nel salvataggio, quindi tutto questo si
può fare anche dopo, aprendo il salvataggio dal menu.

### Fine dell'esperimento

Quando le generazioni richieste sono finite (e l'eventuale selezione finale è
conclusa) la corsa si ferma e sulla pagina principale compare un
**riepilogo**: quante generazioni, i successi dell'ultima, **quanti
sopravvissuti ci sono e chi sono** (i loro numeri), il più anziano, l'esito di
ogni prova della selezione, dove sono stati salvati. Sotto, i pulsanti per le
cose da fare dopo:

| Pulsante | Cosa fa |
|---|---|
| Guarda i sopravvissuti in arena (`A`) | li mette tutti insieme nel banco di prova, per vedere come si comportano |
| Esplora le loro reti (`2`) | chiude il riepilogo e mostra l'elenco dei sopravvissuti con le loro reti |
| Continua l'evoluzione (`Spazio`) | riprende da dove si era fermata, senza limite |
| Chiudi (`Invio`) | chiude il riepilogo; `Invio` lo riapre |

### Il banco di prova

Una rete sola, o un gruppo, nell'arena fuori dall'evoluzione: nessuno nasce e
nessuno viene eliminato, si osserva soltanto.

- **una rete**: un campione o un singolo individuo, scelto con `T` dalla pagina
  principale o dall'esploratore;
- **un gruppo**: tutti i sopravvissuti insieme, con `A` (o dal riepilogo di
  fine esperimento). Si segue un agente col clic o con `Tab`; la sua rete è dal
  vivo nel pannello di destra. In basso a destra una riga per rete: quante
  arene viste ha completato e, dopo `E`, la sua percentuale nella
  valutazione, le migliori in alto.

- Le arene sono **sempre le stesse** (arena 0, 1, 2, ..., con semi propri,
  diversi da quelli dell'evoluzione): reti diverse si confrontano sulle stesse
  prove. A fine arena si passa da sola alla successiva; in basso a destra un
  quadratino per arena vista (azzurro = completato).
- `E` lancia una **valutazione**: 20 arene fisse, con 50 copie della rete per
  arena (10 copie a testa per un gruppo) che partono da punti diversi. Dà la
  percentuale di prove completate (per ogni rete, in un gruppo), il tempo medio
  per arrivare e, per una rete sola, una barra per arena. Va avanti a pezzi,
  senza bloccare la schermata.

| Tasto | Azione |
|---|---|
| `Spazio` | pausa |
| `+` / `-` | velocità da x1 a x64 |
| `N` / `R` | arena successiva / rifai questa |
| `A` | passa da sola alla successiva sì/no |
| `E` | valutazione su 20 arene (50 copie, 10 a testa per un gruppo) |
| clic, `Tab` | segui un altro agente del gruppo |
| `M` | guida manuale (WASD o frecce) |
| `V` | recettori disegnati sì/no |
| `B` | pausa/ripresa della corsa in sottofondo |
| `Esc` | torna indietro |

## Le regole, in dettaglio

Tutti i numeri qui sotto sono i valori di default. Tra parentesi c'è il nome
della costante in `evonets/config.py` che li controlla.

### 1. Il tempo

- La simulazione avanza a passi di 1/60 di secondo (`dt`).
- A ogni passo, nell'ordine: ogni agente legge i propri sensi, la sua rete
  calcola i comandi, la fisica muove tutti, si aggiorna l'energia, si controlla
  chi è morto e chi ha completato il compito.

### 2. L'arena e gli agenti

- **Arena**: rettangolo di 200 × 140 unità (`arena_w`, `arena_h`) chiuso da pareti.
- **Agente**: un cerchio di raggio 1 (`agent_radius`) con una direzione (il muso).
- **Nascita**: posizione casuale, orientamento casuale, fermo, ad almeno 4 unità
  dal bordo di ogni zona di veleno (`spawn_margin`).
- **Gli agenti non si toccano tra loro**: si attraversano. Ognuno affronta la
  stessa arena come se fosse solo.

### 3. Il movimento

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

### 4. L'energia e la morte

- Ogni agente nasce con **150 di energia** (`energy_init`). **Non si ricarica mai.**
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
  contro il 39% di quelle evolute con 100 (dettagli in `docs/NOTE.md`).
- Cablaggio, neuroni e recettori si pagano anche da fermi, come il cervello
  vero che consuma energia sempre: un cervello grande o tanti sensi precisi
  accorciano la vita. Al posto di un numero massimo c'è un prezzo, e la
  selezione tiene un neurone, un recettore o un collegamento lungo solo se
  vale quello che costa. I costi sono ancora da calibrare con le misure (vedi
  la specifica, sezione 10).
- **Un agente morto** resta fermo dov'è, diventa grigio e non può più completare
  il compito.

### 5. Il veleno

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

### 6. Il compito

Compito attuale: **raggiungere il bersaglio**.

- Il bersaglio è un cerchio di raggio 3 (`target_radius`) in una posizione
  casuale. Può capitare anche dentro o vicino a una zona di veleno.
- È uno solo per tutti e non si consuma.
- **Completa il compito** l'agente vivo che lo tocca, cioè il cui centro arriva a
  meno di 4 unità dal centro del bersaglio.
- Chi completa è **in salvo**: si ferma, diventa bianco, non consuma più energia
  e il veleno non gli fa più nulla.
- Non conta in quanto tempo, con quanta energia residua o per quale strada:
  **completato è completato**.

### 7. I sensi: i recettori universali

L'agente non conosce la propria posizione, né dov'è il bersaglio, né il proprio
orientamento nell'arena. Sa solo ciò che i suoi recettori captano, e i
recettori sono a loro volta il risultato dell'evoluzione. Il modello completo,
con le formule, è in [docs/SPEC_FASE_C.md](docs/SPEC_FASE_C.md).

**La fisica: sei fenomeni.** Lo decide il mondo, non l'agente. Ognuno ha un
asse della "qualità" da 0 a 1:

| Fenomeno | Come arriva | Qualità | Nel mondo, di default |
|---|---|---|---|
| luce | in linea retta; il mezzo è torbido e la luce si attenua come e^(−d/12) | colore | pareti grigie (0,5), bersaglio rosso (0,85), veleno verde (0,25) che brilla come una nube e non nasconde ciò che sta dietro |
| sostanze | concentrazione nel punto del corpo | molecola | la tossina del veleno (0,15): dentro la zona è uguale al danno; fuori il veleno è inodore |
| meccanica | pressione da contatto, vibrazioni a distanza | frequenza | la pressione di una parete toccata (frequenza 0); nessuna sorgente di vibrazioni |
| temperatura | campo nel punto del corpo | la temperatura | uniforme (0,5), nessuna sorgente di calore |
| magnetismo | campo uniforme, come quello di un pianeta | — | nessun campo |
| interno | grandezze del corpo | quale grandezza | energia, velocità avanti e indietro, derapata e rotazione a destra e a sinistra |

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

### 8. La rete: struttura comune ai due tipi

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
- i **motori** stanno in posizioni fisse sul bordo destro;
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
- **I pesi possono cambiare durante la vita**, se l'individuo ha dei neuroni
  modulatori: vedi "Imparare durante la vita" qui sotto. Ciò che si impara non
  si eredita.

**Imparare durante la vita** (dettagli in [docs/SPEC_FASE_E.md](docs/SPEC_FASE_E.md)).

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

### 9. Rete continua (`--net rate`, predefinita)

- Ogni neurone ha un valore tra -1 e 1.
- A ogni passo: somma i valori delle sue sorgenti, ciascuno moltiplicato per il
  peso del collegamento, aggiunge la propria **soglia** (un numero suo, anche
  negativo) e schiaccia il risultato tra -1 e 1 (funzione tanh).
- **Segno libero**: ogni collegamento ha il suo. Peso positivo = eccita,
  negativo = inibisce. Lo stesso neurone può eccitare un vicino e inibirne un
  altro. Non esistono neuroni "eccitatori" o "inibitori".
- **2 motori**: il valore del primo è la spinta, quello del secondo la sterzata.
- Geni propri di ogni neurone: 1, la soglia.

### 10. Rete a impulsi con neuroni eccitatori o inibitori (`--net spike-ei`)

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

### 11. Il genoma

Il genoma di un individuo è tutto ciò che serve a costruire il suo corpo
sensoriale e la sua rete:

- per ciascun posto di **recettore**: se esiste e tutti i suoi geni (punto 7);
- per ciascun posto di **neurone interno**: se esiste e dove sta nel cervello;
- per ciascun collegamento possibile, se **esiste** e il suo **peso**;
- per ciascun neurone interno o motore, i suoi **geni propri**;
- il **meccanismo di mutazione**: i tassi di mutazione e la portata λ dei
  collegamenti dell'individuo (punto 15).

È organizzato a **blocchi**: uno per neurone (interno o motore: se esiste, dove
sta, i geni propri e tutti i collegamenti *in ingresso*), uno per recettore,
uno per il meccanismo di mutazione. Nell'incrocio ogni blocco arriva intero da
un genitore. I posti si abbinano per posizione: il posto k della madre con il
posto k del padre.

### 12. La generazione

- La popolazione è di **1.000 individui** (`population`): abbastanza perché la
  selezione pesi più del caso. Resta sempre di quella
  dimensione.
- Tutti gli individui di una generazione vivono nella stessa arena, nello stesso
  momento.
- **La generazione finisce** quando ogni individuo è morto oppure ha completato
  il compito. C'è anche un tetto di sicurezza di 90 secondi (`life_max`), che
  con i valori di default non scatta mai.
- **Ogni generazione ha un'arena nuova**: zone di veleno, bersaglio e punti di
  nascita vengono risorteggiati.

### 13. Chi si riproduce

- **Si riproduce solo chi ha completato il compito.** Non esiste punteggio.
- **Tutti i genitori contano uguale**: né la velocità né l'energia residua danno
  vantaggi.
- Chi è morto, o non ha completato, non lascia nulla.

### 14. Come si forma la generazione successiva

1. **I genitori sopravvivono**: ogni genitore passa identico alla generazione
   successiva, dove dovrà rifare il compito in un'arena nuova.
2. **Nessun limite di età**: un individuo resta nella popolazione finché
   continua a completare il compito, e sparisce la prima volta che fallisce.
   (`max_age` permette di imporre un massimo di generazioni; 0, il default,
   vuol dire nessun limite.)
3. **I posti restanti vanno ai figli**: 1.000 meno i sopravvissuti.

Per ogni figlio:

- si sorteggia un genitore, tutti con la stessa probabilità;
- metà dei figli (`crossover_rate`) ha anche un **secondo genitore**, sempre
  diverso dal primo; l'altra metà è copia di un solo genitore;
- con due genitori, per ogni blocco del genoma il figlio prende quello della
  madre o quello del padre, con probabilità uguale. Un neurone, un recettore e
  il meccanismo di mutazione arrivano sempre interi da un solo genitore;
- poi il figlio viene **mutato**.

Di conseguenza il numero di figli per genitore dipende da quanti completano: con
12 genitori sono circa 80 a testa, con 500 genitori circa 1.

### 15. Le mutazioni

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

- Un collegamento che compare usa il peso già scritto in quel punto del genoma.
- Un collegamento che esiste resta anche se i suoi due neuroni si allontanano
  (l'assone c'è già), ma costa di più.
- Nelle reti a impulsi il tipo di un neurone cambia quando il ritocco del suo
  gene lo porta dall'altra parte dello zero.

### 16. I casi limite

| Quanti hanno completato | Cosa succede |
|---|---|
| due o più | regole normali |
| **uno solo** | regole normali: sopravvive e tutti i figli sono suoi |
| **nessuno**, ma in passato sì | si riparte dai genitori dell'ultima generazione che ne ha avuti: tutti figli, con mutazioni normali; nessuno sopravvive identico |

Non c'è nessun "soccorso" con mutazioni rinforzate quando i genitori sono
pochi: in natura non esiste.
| **nessuno**, e mai nessuno prima | popolazione primordiale nuova |

### 17. Ripetibilità

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

### 18. Il numero di ogni individuo

Ogni individuo riceve un numero alla nascita (`#123`) e lo conserva finché
resta nella popolazione: un sopravvissuto lo porta con sé di generazione in
generazione, ogni figlio ne riceve uno nuovo. L'**età** è quante prove ha vinto
di fila; il **più anziano** è quello con l'età più alta, ed è mostrato con il
suo numero e la generazione in cui è nato.

### 19. La selezione finale (facoltativa)

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
   dell'esperimento"), da cui li si può guardare in arena.

Durante la selezione la pagina principale e l'arena dicono a che prova si è e
quanti sono in gara; al posto dell'ultimo grafico c'è l'esito di ogni prova.
Finita la selezione, `Spazio` riprende l'evoluzione: i selezionati corrono la
generazione in corso da soli, poi la popolazione torna piena con i loro figli.
Lo stesso succede riprendendo l'evoluzione dal file dei selezionati. Il file si
può anche aprire dal menu per esplorarne le reti o provarle una per una.

### 20. L'esame fisso

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
- Se un giorno le prove cambiano, cambia anche il numero di versione
  dell'esame, salvato con ogni risultato: punteggi di versioni diverse non si
  confrontano.

### Un punto di riferimento

In una prova da 120 generazioni con le impostazioni di default (reti continue) i
successi sono passati dal 2% a circa il 30–40%, ancora in crescita, in meno di
quattro minuti. La percentuale oscilla molto da una generazione all'altra perché
alcune arene sono più difficili di altre. È una prova sola: i numeri sono
indicativi.

## Struttura del codice

```
evonets/
  config.py          tutte le costanti
  sim/               simulatore: non sa nulla di reti
    world.py           arena, fisica, energia
    hazards.py         veleno
    tasks.py           compiti (e le varianti dell'esame: bersaglio lontano, in movimento)
    signals.py         fisica dei segnali: cosa arriva in un punto per ogni fenomeno
  net/               rete
    receptors.py       recettore universale: geni, segnale, rumore, descrizione a parole
    brain.py           genoma e calcolo delle reti
    geometry.py        dove stanno i neuroni e con che probabilità si collegano
    controllers.py     recettori -> rete -> comandi
    anatomy.py         struttura di una singola rete: rami morti, distanze, genoma in testo
  evo/               evoluzione
    genetics.py        incrocio, mutazione, duplicazione, crescita dei posti
    evolution.py       ciclo delle generazioni, campioni, salvataggio
    trial.py           valutazione di una rete singola su arene fisse
    exam.py            esame fisso: 7 prove, in parallelo su più processi
  gpu/               motore di simulazione compilato con JAX (GPU)
    engine.py          un passo: sensi, rete, plasticità, fisica, come il motore NumPy
    sim.py             più mondi insieme sulla GPU, e il ritorno negli oggetti NumPy
  viz/               interfaccia (l'unica parte che usa la grafica)
    app.py             finestra, menu e corsa (pagina principale e arena)
    render.py          arena e rete dal vivo, grafici
    inspector.py       elenco di reti, rete disegnata e toccabile, genoma
    testbench.py       banco di prova di una rete singola
    widgets.py         controlli del menu
main.py              avvio: interfaccia, addestramento nel terminale, benchmark
tests/               un file di test per ciascun gruppo
docs/NOTE.md         idee e direzioni future
```

Tutta la simulazione lavora su array con una riga per agente, quindi l'intera
popolazione avanza in un solo passo. Gira su CPU.

## Personalizzare

- **Costanti**: `evonets/config.py` raccoglie fisica, energia, veleno, segnali
  emessi dagli oggetti, costi, il corpo primordiale, popolazione e valori di
  partenza delle mutazioni, ciascuna con un commento. Molte sono regolabili
  anche dal vivo con `C`.
- **Nuovo compito**: in `evonets/sim/tasks.py` un compito è un oggetto con un
  nome, `reset(world)` e `completed(world)`, che dice chi ha completato. Si
  scrive accanto a quello esistente e si passa a `Evolution(cfg, task=...)`.
- **Salvataggi**: contengono la popolazione, la storia e il campione di ogni
  generazione (circa 20 KB l'uno: una corsa di migliaia di generazioni occupa
  qualche decina di MB, in memoria e su disco). Contengono anche la
  **configurazione completa** e il **registro dell'ambiente**: quando la corsa
  è nata o è stata ripresa, con quale versione del codice (il commit git,
  "+modifiche" se c'erano modifiche non committate), e ogni costante cambiata,
  con la generazione in cui è cambiata (anche i ritocchi fatti con `C`).
  Riprendendo una corsa si riparte con la configurazione salvata, non con
  quella attuale. Il registro compare in cima alla pagina principale, nel
  menu accanto al salvataggio e nel terminale con `--load`.
  Il formato attuale è il 6 (recettori universali con adattamento e sensi
  calibrati, posti che crescono, mutazioni come geni, plasticità). Quelli dei
  formati 3, 4 e 5 si aprono convertiti (senza adattamento, senza plasticità,
  sensibilità ricalcolate come moltiplicatori a parità di quanti), con le
  costanti con cui erano nati, e il registro lo annota. Quelli precedenti avevano i
  sensi fissi e non si possono più aprire.

## Stato e prossimi passi

Fatto: simulatore, recettori universali che evolvono, reti a forma libera
senza numero massimo di neuroni (continue e a impulsi con neuroni eccitatori o
inibitori), tassi di mutazione come geni, ciclo evolutivo, interfaccia con
menu, pagina principale con statistiche e reti dei vincitori esplorabili
durante la corsa, esploratore di genomi, banco di prova per singole reti,
salvataggio con configurazione e registro dell'ambiente, esame fisso, per il
primo compito.

Da fare:

- calibrare i costi di neuroni e recettori ed esperimento di validazione dei
  recettori (specifica, sezione 10);
- l'ambiente che rende utile imparare (fase E: la logica dei neuroni c'è già);
- oggetto fisico da spingere e zona di consegna (compiti 2 e 3);
- portare sul motore GPU anche il banco di prova dal vivo (oggi usa il motore NumPy);
- esperimenti sulla forma della rete e del genoma, descritti in
  [docs/NOTE.md](docs/NOTE.md).
