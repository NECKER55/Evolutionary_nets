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
uv run main.py --evolve 300           # 300 generazioni senza grafica, poi salva
uv run main.py --load runs/pop.npz    # riprende da un salvataggio
uv run main.py --bench                # misura la velocità della simulazione
uv run pytest                         # test
```

Opzioni: `--agents N` (individui, default 200), `--seed N` (stessa sequenza di
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

## Come funziona

### Il mondo

- **Arena** 100 × 70 unità, con pareti che fanno rimbalzare.
- **Movimento**: due comandi tra -1 e 1, spinta e sterzata. La spinta accelera
  lungo il muso, la sterzata è un'accelerazione angolare; l'attrito frena molto
  di traverso e poco in avanti.
- **Energia**: ogni agente parte con 100 e non si ricarica. Esistere costa 5 al
  secondo (20 secondi di vita da fermo), muoversi fino a 5 in più, sterzare fino
  a 1 in più. A zero si muore.
- **Veleno**: zone circolari che si spostano e pulsano. Il danno cresce dal bordo
  (zero) al centro (60 al secondo).
- **Compito attuale**: toccare il bersaglio. Chi ci riesce è in salvo e smette di
  consumare.
- Gli agenti non si urtano tra loro: ognuno affronta la stessa prova da solo.

### I sensi (42 ingressi)

- **Vista**: 9 raggi su 120° davanti al muso, portata 30 unità. Ogni raggio ha 4
  ingressi, uno per tipo di cosa (parete, veleno, bersaglio, zona di consegna),
  che valgono 0 se non vede nulla e salgono fino a 1 a contatto. Il canale
  "consegna" è riservato ai compiti futuri e per ora resta a 0.
- **Corpo**: velocità in avanti, di traverso e di rotazione, energia residua,
  dolore (quanto veleno sta subendo), urto con una parete.

L'agente non conosce la propria posizione né la direzione del bersaglio: deve
cercarlo.

### La rete

- 42 ingressi e 2 uscite fissi, più **fino a 32 neuroni interni**; ogni individuo
  ne ha attivi un numero diverso.
- **Collegamenti liberi**: qualsiasi neurone può parlare a qualsiasi neurone
  interno o motore, anelli e se stesso compresi. Non esistono strati.
- Peso positivo = eccita, negativo = inibisce.
- Tutti i neuroni si aggiornano insieme a ogni passo partendo dai valori del
  passo precedente: gli anelli sono la memoria dell'agente.
- I pesi non cambiano durante la vita. Si impara solo di generazione in
  generazione.

### L'evoluzione

1. Una generazione vive finché tutti sono morti o hanno completato il compito.
2. **Chi ha completato è genitore** e passa identico alla generazione successiva,
   per un massimo di 5 generazioni di vita.
3. I posti restanti vanno ai **figli**, con genitori sorteggiati alla pari. Metà
   sono copie di un genitore, metà nascono da due genitori; tutti vengono mutati.
4. Ogni generazione affronta un'**arena nuova**, uguale per tutti.

L'incrocio avviene a blocchi: per ogni neurone il figlio prende da un solo
genitore l'esistenza, la soglia e tutti i collegamenti in ingresso. Le mutazioni
ritoccano pesi e soglie, aggiungono o tolgono collegamenti, accendono o spengono
neuroni interni.

Casi limite:

- **Un solo genitore**: sopravvive, e metà dei figli nasce con mutazioni più forti.
- **Nessun genitore**: si riparte dagli ultimi genitori conosciuti, mutati.
- **Nessun genitore e mai avuti**: reti casuali nuove.

Come riferimento, in una prova da 150 generazioni con le impostazioni di default
i successi sono passati dal 3% a circa il 70–80%, in meno di due minuti. La
percentuale oscilla molto da una generazione all'altra perché alcune arene sono
più difficili di altre.

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

Fatto: simulatore, sensi, reti a forma libera, ciclo evolutivo, viewer con
statistiche e salvataggio, per il primo compito.

Da fare:

- oggetto fisico da spingere e zona di consegna (compiti 2 e 3);
- ottimizzazione delle prestazioni (compilazione dei cicli pesanti, eventualmente GPU);
- esperimenti sulla forma della rete e del genoma, descritti in
  [docs/NOTE.md](docs/NOTE.md).
