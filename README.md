# Evolutionary_nets

Vita artificiale e neuroevoluzione in un'arena 2D. Una popolazione di agenti,
ognuno guidato da una propria rete neurale, vive in un ambiente con zone di
veleno in movimento. Non c'è un punteggio di fitness: **si riproduce solo chi
completa il compito**, e l'energia decide chi muore prima di riuscirci.

Il progetto è una piattaforma per sperimentare: il compito è sostituibile e
pensato per diventare via via più arduo, e la forma della rete non è fissata ma
evolve.

## Avvio

Serve [uv](https://docs.astral.sh/uv/). Le dipendenze (NumPy, pygame-ce) si
installano da sole al primo comando.

```
uv run main.py                        # interfaccia grafica: menu, corsa, esploratore delle reti
uv run main.py --evolve 300           # 300 generazioni nel terminale, senza grafica, poi salva
uv run main.py --evolve 300 --select 5  # ... poi i sopravvissuti rifanno la prova 5 volte, senza figli
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
| Selezione finale | finite le generazioni, quante prove fanno ancora i sopravvissuti, senza figli (vedi "La selezione finale"); 0 = niente. Si attiva solo con Generazioni > 0 |
| Riprendi da | un salvataggio in `runs/` da cui continuare; rete e individui vengono da lì |
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
neuroni e collegamenti medi, durata dell'ultima generazione, campioni raccolti.
In cima alla pagina una riga dice anche in che generazione è nato. **Il più
anziano è cliccabile** (riquadro bordato o riga sottolineata in cima, oppure
`O`): si apre la sua rete. Sotto, quattro grafici per generazione:

- successi (linea sottile: ogni generazione; spessa: media delle ultime 10);
- dimensione media delle reti: neuroni interni e collegamenti;
- sopravvissuti e generazioni vinte di fila dal più anziano: quanto a lungo un
  genitore continua a vincere;
- quota di neuroni inibitori (reti a impulsi) o durata della generazione (reti continue).

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
| Agente giallo | quello che stai seguendo, con i suoi raggi visivi |
| Anello bianco | sopravvissuto di una generazione precedente |
| Bianco pieno | ha completato il compito |
| Grigio | morto |

| Nel pannello della rete | Significato |
|---|---|
| Colonna sinistra | i sensi: 4 pallini per raggio (il bordo indica cosa guarda), poi i sensi del corpo |
| Cerchio centrale | i neuroni interni di quell'individuo |
| Destra | i motori |
| Pallino arancione / azzurro | neurone attivo in positivo / negativo |
| Linea arancione / azzurra | collegamento che eccita / inibisce; si illumina quando passa segnale |

| Tasto | Azione |
|---|---|
| `+` / `-` | velocità da x1 a x16 |
| `T` | turbo |
| clic | segui un altro agente |
| `M` | guida manuale dell'agente seguito (WASD o frecce) |
| `V` | raggi visivi sì/no |
| `C` | elenco delle costanti; `Tab` sceglie, `,` `.` regolano |
| `O` | segui il più anziano: il pannello a destra mostra la sua rete |
| `F`, `I`, `Spazio`, `S`, `Esc` | come nella pagina principale |

### Come si legge una rete

Vale per la pagina principale e per l'esploratore. Sensi in colonna a
sinistra, motori a destra, **neuroni interni su un'ellisse** in mezzo: così i
collegamenti tra interni (anelli compresi, cioè la memoria) sono corde ben
visibili invece di linee sovrapposte. Sull'arco sinistro stanno gli interni che
ascoltano soprattutto i sensi, sul destro quelli che parlano soprattutto ai
motori; su ogni arco l'ordine dall'alto in basso segue i sensi che li
alimentano, per ridurre gli incroci. Gli archi curvi sono i motori che
rimandano il loro valore agli interni; un cerchietto sopra un neurone è un
collegamento con se stesso.

Colore = segno (arancio eccita, azzurro inibisce), intensità e spessore =
forza. Nelle reti a impulsi il bordo di un neurone interno dice il suo tipo. In
grigio i **rami morti**: neuroni e collegamenti che non arrivano a nessun
motore e quindi non cambiano il comportamento. Passando su un neurone (o
cliccandolo) si accendono solo i suoi collegamenti, con una **freccia** che ne
indica il verso; il pannello dei dettagli dice anche a quanti passi è dai sensi.

### L'esploratore delle reti

`I` apre a tutto schermo lo stesso elenco e la stessa rete della pagina
principale, più il **genoma** come matrice: una riga per blocco (neurone
interno o motore), una colonna per sorgente (sensi | interni | motori). Colore =
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
| `V` | raggi visivi sì/no |
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

- Ogni agente nasce con **100 di energia** (`energy_init`). **Non si ricarica mai.**
- **Si muore in un solo modo: quando l'energia arriva a zero.** Non esiste morte
  istantanea.
- L'energia si consuma per tre motivi, che si sommano:

| Causa | Consumo al secondo |
|---|---|
| Esistere | 2,5 (`metabolism_basal`) |
| Spingere | fino a 2,5 in più, proporzionale al comando di spinta, uguale in avanti e indietro (`metabolism_move`) |
| Sterzare | fino a 0,5 in più, proporzionale al comando di sterzata (`metabolism_turn`) |
| Veleno | da 0 a 60, secondo la profondità nella zona (vedi sotto) |

- Si paga lo **sforzo comandato**, non la velocità raggiunta: spingere contro una
  parete costa come correre.
- Conseguenze: un agente fermo vive 40 secondi; uno sempre a spinta piena 20
  secondi, in cui percorre al massimo circa 600 unità (tre larghezze di arena).
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

### 7. I sensi: i 42 ingressi della rete

L'agente non conosce la propria posizione, né dov'è il bersaglio, né il proprio
orientamento nell'arena. Sa solo quello che segue.

**Vista: 36 ingressi.**

- 9 raggi (`vision_rays`) a ventaglio su 120° davanti al muso
  (`vision_fov_deg`), dal più a sinistra al più a destra. Dietro è cieco.
- Ogni raggio parte dal centro dell'agente e arriva a 30 unità (`vision_range`).
- Ogni raggio ha **4 ingressi, uno per tipo di cosa**. La rete distingue cosa
  vede da *quale* ingresso si accende:

| Ingresso | Cosa misura |
|---|---|
| parete | la parete dell'arena |
| veleno | il bordo della zona di veleno più vicina lungo il raggio |
| bersaglio | il bersaglio |
| consegna | riservato ai compiti futuri: oggi vale sempre 0 |

- Il valore è la **vicinanza**: 0 se lungo il raggio non c'è nulla entro 30
  unità, 0,5 se la cosa è a metà raggio, 1 se è a contatto.
- I quattro tipi sono indipendenti: il veleno non nasconde ciò che sta dietro.
  Lo stesso raggio può segnalare insieme veleno a 0,6 e bersaglio a 0,2.
- Se l'agente è già dentro una zona di veleno, tutti i raggi segnano veleno a 1.
  La vista dice dove inizia una zona, non quanto è pericoloso il punto in cui ci
  si trova: per quello c'è il dolore.

**Corpo: 6 ingressi.**

| Ingresso | Valore | Significato |
|---|---|---|
| velocità in avanti | da -1 a 1 | 1 = velocità massima in avanti, negativo = retromarcia |
| velocità di traverso | da -1 a 1 | quanto sta derapando |
| rotazione | da -1 a 1 | 1 = rotazione massima verso destra |
| energia | da 0 a 1 | quota di energia rimasta |
| dolore | da 0 a 1 | quanto veleno sta subendo in questo istante (0 fuori e sul bordo, 1 al centro) |
| urto | 0 o 1 | 1 se sta toccando una parete |

L'ordine degli ingressi è: i 4 del raggio 1, i 4 del raggio 2, … i 4 del raggio
9, poi i 6 del corpo. Ingressi e uscite sono fissi e uguali per tutti: sono il
corpo dell'agente, non evolvono.

### 8. La rete: struttura comune ai due tipi

**I neuroni.** Tre gruppi:

| Gruppo | Quanti | Esistono sempre? |
|---|---|---|
| sensi | 42 | sì |
| interni | da 0 a 32 (`net_hidden_max`) | no: ogni individuo ne ha un sottoinsieme |
| motori | 2 (reti continue) o 4 (reti a impulsi) | sì |

**Quanti neuroni interni ha un individuo.** Il genoma ha 32 posti. Per ogni
posto un gene dice se quel neurone esiste.

- Alla nascita di una popolazione casuale, ogni individuo sorteggia una
  percentuale tra 0 e 100% e ciascuno dei suoi 32 posti viene occupato con
  quella probabilità. Il risultato va da 0 a 32, in media 16, e varia molto da
  un individuo all'altro.
- Nelle generazioni successive il numero cambia solo per eredità e mutazione.
  Non c'è un minimo né un obiettivo: lo decide la selezione.
- Un neurone che non esiste resta a zero e tutti i suoi collegamenti sono
  ignorati. I suoi geni però restano scritti nel genoma: se una mutazione lo
  accende, nasce già con collegamenti e pesi.

**I collegamenti.** Non ci sono strati.

- Un collegamento va da una **sorgente** (un senso, un neurone interno o un
  motore) a un **destinatario** (un neurone interno o un motore). I sensi non
  ricevono collegamenti.
- Sono ammessi tutti i casi: senso → motore diretto, interno → interno, anelli,
  un neurone verso se stesso, un motore che rimanda il proprio valore a un interno.
- Collegamenti possibili: 2.584 nelle reti continue, 2.808 in quelle a impulsi.
- Alla nascita ogni collegamento possibile esiste con probabilità 15%
  (`net_conn_prob`); contano solo quelli tra neuroni esistenti. Un senso può
  restare senza alcun collegamento: per quell'individuo è come non averlo.
- Ogni collegamento ha un **peso**, sorteggiato attorno a zero (per lo più tra
  -2 e 2, `net_weight_std`).

**Il tempo nella rete.**

- Tutti i neuroni si aggiornano nello stesso istante, usando i valori del passo
  precedente. Il segnale fa un salto per passo: da senso a motore diretto arriva
  subito, passando per un neurone interno arriva un passo dopo.
- Gli anelli trattengono informazione da un passo all'altro: sono la **memoria**
  dell'agente. Non esiste altra memoria.
- Alla nascita tutti i neuroni partono da zero.
- **I pesi non cambiano durante la vita.** L'agente non impara: si impara solo
  di generazione in generazione.

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
| senso | sì | sì (il segno è del collegamento) |

- I sensi sono liberi per scelta. Con `sensors_excitatory_only` possono solo
  eccitare, come gli altri.
- La **corrente di fondo** di un neurone non è un collegamento e può essere
  positiva (neurone che tende a scattare da solo) o negativa (neurone difficile
  da accendere).

**Sensi e motori non emettono impulsi.**

- I sensi entrano come corrente continua: un valore più alto carica più in
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

Il genoma di un individuo è tutto ciò che serve a costruire la sua rete:

- per ciascuno dei 32 posti, se il neurone interno **esiste**;
- per ciascun collegamento possibile, se **esiste** e il suo **peso**;
- per ciascun neurone interno o motore, i suoi **geni propri**.

È organizzato a **blocchi, uno per neurone** (interno o motore). Il blocco di un
neurone contiene: se esiste, i suoi geni propri, e tutti i suoi collegamenti *in
ingresso* con i pesi.

### 12. La generazione

- La popolazione è di **200 individui** (`population`) e resta sempre di quella
  dimensione.
- Tutti gli individui di una generazione vivono nella stessa arena, nello stesso
  momento.
- **La generazione finisce** quando ogni individuo è morto oppure ha completato
  il compito. C'è anche un tetto di sicurezza di 60 secondi (`life_max`), che
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
3. **I posti restanti vanno ai figli**: 200 meno i sopravvissuti.

Per ogni figlio:

- si sorteggia un genitore, tutti con la stessa probabilità;
- metà dei figli (`crossover_rate`) ha anche un **secondo genitore**, sempre
  diverso dal primo; l'altra metà è copia di un solo genitore;
- con due genitori, per ogni blocco del genoma il figlio prende quello della
  madre o quello del padre, con probabilità uguale. Un neurone arriva sempre
  intero da un solo genitore;
- poi il figlio viene **mutato**.

Di conseguenza il numero di figli per genitore dipende da quanti completano: con
12 genitori sono circa 16 a testa, con 100 genitori circa 1.

### 15. Le mutazioni

Si applicano a ogni figlio, ciascuna in modo indipendente.

| Mutazione | Probabilità | Costante |
|---|---|---|
| ritocco di un peso | 10% per ogni peso, di poco (in media ±0,3) | `mut_weight_prob`, `mut_weight_std` |
| ritocco di un gene proprio di un neurone | 10% per ogni gene, di poco | le stesse |
| un collegamento assente compare | 0,2% per ogni collegamento assente | `mut_conn_add` |
| un collegamento presente sparisce | 1% per ogni collegamento presente | `mut_conn_del` |
| un neurone interno si accende o si spegne | 1% per ogni posto | `mut_neuron_toggle` |

- Un collegamento che compare usa il peso già scritto in quel punto del genoma.
- In pratica un figlio ha qualche collegamento in più e in meno del genitore, e
  circa uno su tre guadagna o perde un neurone.
- Nelle reti a impulsi il tipo di un neurone cambia quando il ritocco del suo
  gene lo porta dall'altra parte dello zero.

### 16. I casi limite

| Quanti hanno completato | Cosa succede |
|---|---|
| due o più | regole normali |
| **uno solo** | sopravvive; metà dei figli (`rescue_share`) nasce con mutazioni tre volte più frequenti e più ampie (`rescue_strength`), per ricreare varietà |
| **nessuno**, ma in passato sì | si riparte dai genitori dell'ultima generazione che ne ha avuti: tutti figli, metà con mutazioni forti; nessuno sopravvive identico |
| **nessuno**, e mai nessuno prima | popolazione di reti casuali nuova |

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
    tasks.py           compiti
  net/               rete
    sensors.py         vista e sensi del corpo
    brain.py           genoma e calcolo delle reti
    controllers.py     sensi -> rete -> comandi
    anatomy.py         struttura di una singola rete: rami morti, distanze, genoma in testo
  evo/               evoluzione
    genetics.py        incrocio e mutazione
    evolution.py       ciclo delle generazioni, campioni, salvataggio
    trial.py           valutazione di una rete singola su arene fisse
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

- **Costanti**: `evonets/config.py` raccoglie fisica, energia, veleno, vista,
  dimensione della rete, popolazione e frequenze di mutazione, ciascuna con un
  commento. Molte sono regolabili anche dal vivo con `C`.
- **Nuovo compito**: in `evonets/sim/tasks.py` un compito è un oggetto con un
  nome, `reset(world)` e `completed(world)`, che dice chi ha completato. Si
  scrive accanto a quello esistente e si passa a `Evolution(cfg, task=...)`.
- **Salvataggi**: contengono la popolazione, la storia e il campione di ogni
  generazione (circa 20 KB l'uno: una corsa di migliaia di generazioni occupa
  qualche decina di MB, in memoria e su disco). Valgono solo per la stessa
  forma di rete. Se cambi numero di
  raggi, sensi o massimo di neuroni, il caricamento viene rifiutato.

## Stato e prossimi passi

Fatto: simulatore, sensi, reti a forma libera (continue e a impulsi con neuroni eccitatori o inibitori), ciclo
evolutivo, interfaccia con menu, pagina principale con statistiche e reti dei
vincitori esplorabili durante la corsa, esploratore di genomi, banco di prova
per singole reti, salvataggio, per il primo compito.

Da fare:

- oggetto fisico da spingere e zona di consegna (compiti 2 e 3);
- ottimizzazione delle prestazioni (compilazione dei cicli pesanti, eventualmente GPU);
- esperimenti sulla forma della rete e del genoma, descritti in
  [docs/NOTE.md](docs/NOTE.md).
