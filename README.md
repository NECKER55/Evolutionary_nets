# Evolutionary_nets

Vita artificiale e neuroevoluzione in un'arena 2D. Una popolazione di agenti,
ognuno guidato da una propria rete neurale, cerca di completare un compito.
Non c'è un punteggio di fitness: **si riproduce solo chi completa il
compito**, e l'energia decide chi muore prima di riuscirci.

È una piattaforma per sperimentare. Il compito è sostituibile e diventa via via
più arduo. La forma della rete non è fissata ma evolve, e così i sensi: ogni
agente nasce con pochi **recettori universali** che possono diventare
sensibili a luce, sostanze, vibrazioni, temperatura, campo magnetico o al
proprio corpo. Non c'è un numero massimo di neuroni o di sensi: ogni cosa ha un
costo in energia, e la selezione decide cosa tenere. Anche i tassi di
mutazione sono geni.

## Documentazione

| File | Contenuto |
|---|---|
| [docs/REGOLE.md](docs/REGOLE.md) | tutte le regole: fisica, energia, compiti, sensi, reti, genoma, riproduzione, mutazioni, esame, salvataggi |
| [docs/INTERFACCIA.md](docs/INTERFACCIA.md) | il manuale dell'interfaccia grafica, schermata per schermata |
| [docs/SPEC_MONDO.md](docs/SPEC_MONDO.md) | il mondo del cibo (prede, predatori, scala di difficoltà) e lo stato dei lavori |
| [docs/SPEC_FASE_C.md](docs/SPEC_FASE_C.md), [SPEC_FASE_E.md](docs/SPEC_FASE_E.md), [SPEC_SVILUPPO.md](docs/SPEC_SVILUPPO.md) | specifiche di recettori, plasticità e sviluppo |
| [docs/NOTE.md](docs/NOTE.md) | idee future ed esperimenti fatti |

## Avvio

Serve [uv](https://docs.astral.sh/uv/). Le dipendenze (NumPy, Numba, JAX con
CUDA, pygame-ce) si installano da sole al primo comando. La prima volta JAX
scarica circa 1 GB di librerie NVIDIA.

```
uv run main.py                          # interfaccia grafica
uv run main.py --task cibo              # interfaccia, già impostata sul mondo del cibo
uv run main.py --evolve 300             # 300 generazioni nel terminale, poi salva
uv run main.py --task cibo --ladder --evolve 2000                  # mondo del cibo con scala adattiva
uv run main.py --task cibo --ladder --controls --seeds 4 --evolve 2000  # più i due controlli, 4 semi
uv run main.py --exam runs/pop.npz      # esame fisso dei più anziani di un salvataggio
uv run main.py --exam runs/pop.npz --lesion neuroni  # lo stesso, senza neuroni interni
uv run main.py --engine numpy ...       # motore NumPy di riferimento (più lento)
uv run pytest                           # test
```

Altre opzioni: `--net rate|spike-ei`, `--agents N`, `--seed N`, `--load FILE`,
`--save FILE`, `--select K`, `--exam-every K`, `--bench` (`uv run main.py -h`).

Di default la simulazione gira sul motore compilato con JAX, sulla scheda video
NVIDIA se c'è, altrimenti sulla CPU. I test verificano che dia gli stessi
risultati del motore NumPy.

## L'interfaccia in breve

Dal **menu** si sceglie la corsa (rete, individui, mondo, scala, seme,
generazioni) o si riapre un salvataggio. La **pagina principale** ha due schede:
la **panoramica** (a che punto è la scala, quanti si riproducono, l'esame, il
cervello medio, tutto a parole) e le **reti** dei vincitori, esplorabili
mentre l'evoluzione continua.

| Tasto | Azione |
|---|---|
| `V` | panoramica / reti |
| `F` | apre/chiude l'arena (la corsa continua) |
| `T` / `A` | prova la rete scelta / tutti i sopravvissuti nel banco di prova |
| `I` | reti e genomi a tutto schermo |
| `E` / `L` | esame fisso / esame con una lesione |
| `Z` | massima velocità |
| `Spazio` / `S` / `Esc` | pausa / salva / menu |

Tutto il resto è in [docs/INTERFACCIA.md](docs/INTERFACCIA.md).

## Struttura del codice

```
evonets/
  config.py          tutte le costanti, ciascuna commentata
  sim/               simulatore: non sa nulla di reti
    world.py           arena, fisica, energia
    hazards.py         veleno
    tasks.py           compiti: bersaglio (e varianti d'esame), cibo
    signals.py         fisica dei segnali: cosa arriva in un punto per ogni fenomeno
  net/               rete
    receptors.py       recettore universale
    brain.py           genoma e calcolo delle reti
    geometry.py        dove stanno i neuroni e con che probabilità si collegano
    controllers.py     recettori -> rete -> comandi
    anatomy.py         struttura di una singola rete, genoma in testo
  evo/               evoluzione
    genetics.py        incrocio, mutazione, duplicazione, crescita
    evolution.py       ciclo delle generazioni, salvataggio
    ladder.py          scala di difficoltà del mondo del cibo
    trial.py           valutazione di una rete su arene fisse
    exam.py            esame fisso e lesioni, in parallelo su più processi
  gpu/               motore compilato con JAX
  viz/               interfaccia (l'unica parte che usa la grafica)
main.py              avvio: interfaccia, terminale, benchmark
tests/               un file di test per ciascun gruppo
```

Per un **nuovo compito** basta un oggetto in `sim/tasks.py` con `name`,
`reset(world)` e `completed(world)` (dettagli nella docstring di
`ReachTarget`).

## Stato

Fatto: simulatore nei due motori, recettori che evolvono, reti a forma libera
(continue e a impulsi), plasticità, tassi di mutazione come geni, esame fisso,
mondo del cibo con scala adattiva, controlli, esame a livelli e lesioni,
interfaccia.

Prossimi passi: taratura del mondo del cibo e primo esperimento (vedi
[docs/SPEC_MONDO.md](docs/SPEC_MONDO.md), sezione 8); sviluppo del genoma
([docs/SPEC_SVILUPPO.md](docs/SPEC_SVILUPPO.md)); un ambiente che renda utile
imparare.
