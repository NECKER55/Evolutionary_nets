# Evolutionary_nets

Vita artificiale e neuroevoluzione in un'arena 2D. Una popolazione di agenti,
ognuno guidato da una propria rete neurale, cerca di completare un compito.
Non c'è un punteggio di fitness: **si riproduce solo chi completa il
compito**, e l'energia decide chi muore prima di riuscirci.

È una piattaforma per sperimentare. Il compito è sostituibile e diventa via via
più arduo. La forma della rete non è fissata ma evolve, e così i sensi: ogni
agente nasce con pochi **recettori universali** che possono diventare
sensibili a luce, sostanze, vibrazioni, temperatura, campo magnetico o al
proprio corpo. In alternativa, nella simulazione "solo cervello", i sensi sono
fissi e decisi da noi ed evolve soltanto il cervello. Non c'è un numero massimo di neuroni o di sensi: ogni cosa ha un
costo in energia, e la selezione decide cosa tenere. Anche i tassi di
mutazione sono geni.

## Documentazione

| File | Contenuto |
|---|---|
| [docs/REGOLE.md](docs/REGOLE.md) | tutte le regole: fisica, energia, compiti, sensi, reti, genoma, riproduzione, mutazioni, esame, salvataggi |
| [docs/INTERFACCIA.md](docs/INTERFACCIA.md) | il manuale dell'interfaccia grafica, schermata per schermata |
| [docs/SPEC_MONDO.md](docs/SPEC_MONDO.md) | il mondo del cibo (prede, predatori, scala di difficoltà) e lo stato dei lavori |
| [docs/SPEC_FASE_C.md](docs/SPEC_FASE_C.md), [SPEC_FASE_E.md](docs/SPEC_FASE_E.md), [SPEC_SVILUPPO.md](docs/SPEC_SVILUPPO.md) | specifiche di recettori, plasticità e sviluppo |
| [docs/SPEC_ARENA.md](docs/SPEC_ARENA.md) | l'arena continua: tutti insieme, nascite e morti, senza generazioni |
| [docs/cervello_sezioni/](docs/cervello_sezioni/cervello_sezioni.pdf) | il cervello a sezioni spiegato parte per parte, con schema e confronto con il cervello vero (LaTeX e PDF) |
| [docs/SPEC_CERVELLO_RAGIONANTE.md](docs/SPEC_CERVELLO_RAGIONANTE.md) | progetto: nuovo cervello con modello del mondo e ragionamento su due livelli, in un mondo 3D con MuJoCo, addestrato a imparare una sezione alla volta |
| [docs/SPEC_PIANIFICAZIONE.md](docs/SPEC_PIANIFICAZIONE.md) | proposta precedente, superata; restano la prova di rivalutazione e l'arena delle casse |
| [docs/NOTE.md](docs/NOTE.md) | idee future ed esperimenti fatti |

## Avvio

Serve [uv](https://docs.astral.sh/uv/). Le dipendenze (NumPy, Numba, JAX con
CUDA, pygame-ce) si installano da sole al primo comando. La prima volta JAX
scarica circa 1 GB di librerie NVIDIA.

```
uv run main.py                          # interfaccia grafica
uv run main.py --task cibo              # interfaccia, già impostata sul mondo del cibo
uv run main.py --task cibo --mode cervello --ladder --evolve 2000  # sensi fissi, evolve solo il cervello
uv run main.py --evolve 300             # 300 generazioni nel terminale, poi salva
uv run main.py --task cibo --ladder --evolve 2000                  # mondo del cibo con scala adattiva
uv run main.py --task arena --agents 750 --evolve 60               # arena continua: 60 minuti di mondo
uv run main.py --task cibo --ladder --controls --seeds 4 --evolve 2000  # più i due controlli, 4 semi
uv run main.py --exam runs/pop.npz      # esame fisso dei più anziani di un salvataggio
uv run main.py --exam runs/pop.npz --lesion neuroni  # lo stesso, senza neuroni interni
uv run main.py --engine numpy ...       # motore NumPy di riferimento (più lento)
uv run python -m evonets.arenas.mondo3d.run --confronto --minuti 10  # mondo 3D: agente perfetto e base
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
  config.py            tutte le costanti, ciascuna commentata
  gpu.py               c'è JAX? (cache delle compilazioni)
  arenas/              una cartella per arena: mondo, motore GPU, corsa, salvataggi
    signals.py           fisica dei segnali: cosa arriva in un punto per ogni fenomeno
    generational/        a generazioni: bersaglio e mondo del cibo
      world.py hazards.py tasks.py   arena, fisica, energia; veleno; compiti
      evolution.py       ciclo delle generazioni, campioni, selezione finale
      saves.py           salvataggi: formato, scrittura sicura, ripresa con le regole recenti
      ladder.py          scala di difficoltà del mondo del cibo
      trial.py exam.py   valutazione su arene fisse; esame fisso e lesioni
      capacity.py        test di capacità del cervello
      gpu_engine.py gpu_worlds.py    il passo compilato; più mondi insieme
    continuous/          arena continua: mondo unico, senza generazioni
      world.py ecology.py gpu.py     il mondo; nascite, morti, salvataggi; la GPU
    pile/                arena delle pile: tanti recinti, apprendimento dentro la vita
      world.py run.py gpu.py         il mondo; prove e misure; la GPU
      saves.py bench.py              salvataggi; banco di prova fisso
    mondo3d/             le stesse regole delle pile, con corpi e telecamere MuJoCo
      world.py bodies.py run.py   il mondo; i cablaggi dei corpi; prove e misure
  brains/              una cartella per famiglia di cervelli
    freeform/            reti a forma libera, evolvono (arene generational e continuous)
      rate.py spiking.py   i due tipi di rete: continua e a impulsi
      genome.py genetics.py   il genoma; incrocio, mutazione, duplicazione, crescita
      plasticity.py      la regola di apprendimento
      receptors.py senses.py  recettore universale; sensi fissi
      geometry.py development.py womb.py   dove stanno i neuroni; sviluppo; grembo
      controllers.py     recettori -> rete -> comandi
      anatomy.py         struttura di una singola rete, genoma in testo
      gpu.py womb_gpu.py le stesse reti e lo stesso grembo con JAX
    pile/                agenti dell'arena delle pile
      sections.py        il cervello a sezioni
      actor_critic.py guided.py   i due agenti di confronto
      agent.py attention.py value.py genes.py   base comune, attenzione, O e K, geni
      gpu.py             gli stessi agenti con JAX
    mondo3d/             agenti di controllo del mondo 3D, non cervelli
      base.py perfect.py   vede solo la profondità; legge la verità dal simulatore
    ragionante/          il cervello nuovo, staccato da quello a sezioni: per ora il motore
      motor.py brain.py  sezione 7, con neuroni nascosti e taratura; il cervello attorno a lei
  viz/                 interfaccia (l'unica parte che usa la grafica)
    app.py launcher.py   finestra; menu
    render.py inspector.py widgets.py   arena e rete dal vivo; reti e genomi; controlli
    generational/        corsa, panoramica, banco di prova
    continuous/          vista del mondo e schede
    pile/                recinto e cervello a sezioni
    mondo3d/             vista da fuori, immagine, profondità e sezione 7 (motor_panel.py)
main.py              avvio: interfaccia, terminale, benchmark
tests/               un file di test per ciascun gruppo
```

Per un **nuovo compito** basta un oggetto in `arenas/generational/tasks.py` con `name`,
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
