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
uv run main.py                        # evoluzione dal vivo nel viewer
uv run main.py --net spike-ei         # lo stesso, con reti a impulsi (neuroni eccitatori o inibitori)
uv run main.py --evolve 300           # 300 generazioni senza grafica, poi salva
uv run main.py --load runs/pop.npz    # riprende da un salvataggio
uv run main.py --bench                # misura la velocità della simulazione
uv run pytest                         # test
```

Opzioni: `--net rate|spike-ei` (tipo di rete, default `rate`), `--agents N`
(individui, default 200), `--seed N` (stessa sequenza di
arene e reti a parità di numero), `--save FILE` (default `runs/pop.npz`).

Il flusso tipico è addestrare in fretta con `--evolve` e poi riaprire il
risultato con `--load` per guardarlo.

## Il viewer

A sinistra l'arena, a destra la rete dell'agente seguito e, sotto, il grafico
dei successi per generazione (linea sottile: ogni generazione; linea spessa: media
delle ultime 10).

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
| Destra | i due motori, spinta e sterzo |
| Pallino arancione / azzurro | neurone attivo in positivo / negativo |
| Linea arancione / azzurra | collegamento che eccita / inibisce; si illumina quando passa segnale |

| Tasto | Azione |
|---|---|
| `+` / `-` | velocità da x1 a x16 |
| `T` | turbo |
| `Spazio` | pausa |
| `S` | salva la popolazione |
| clic | segui un altro agente |
| `M` | guida manuale dell'agente seguito (WASD o frecce) |
| `V` | raggi visivi sì/no |
| `C` | elenco delle costanti; `Tab` sceglie, `,` `.` regolano |

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

Tutto ciò che è casuale dipende dal seme (`--seed`). A parità di seme e di
impostazioni, due corse danno esattamente le stesse arene, le stesse reti e gli
stessi risultati.

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
  evo/               evoluzione
    genetics.py        incrocio e mutazione
    evolution.py       ciclo delle generazioni, salvataggio
  viz/               viewer (l'unica parte che usa la grafica)
    render.py
main.py              avvio: viewer, addestramento veloce, benchmark
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
- **Salvataggi**: valgono solo per la stessa forma di rete. Se cambi numero di
  raggi, sensi o massimo di neuroni, il caricamento viene rifiutato.

## Stato e prossimi passi

Fatto: simulatore, sensi, reti a forma libera (continue e a impulsi con neuroni eccitatori o inibitori), ciclo
evolutivo, viewer con statistiche e salvataggio, per il primo compito.

Da fare:

- oggetto fisico da spingere e zona di consegna (compiti 2 e 3);
- ottimizzazione delle prestazioni (compilazione dei cicli pesanti, eventualmente GPU);
- esperimenti sulla forma della rete e del genoma, descritti in
  [docs/NOTE.md](docs/NOTE.md).
