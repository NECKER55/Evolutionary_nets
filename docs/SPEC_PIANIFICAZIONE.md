# Pianificazione e modello del mondo (progetto)

Stato: progetto, niente codice. Estende il cervello a sezioni
(docs/SPEC_CERVELLO_SEZIONI.md) e l'arena delle pile (docs/SPEC_ARENA_PILE.md).
**Superata da docs/SPEC_CERVELLO_RAGIONANTE.md**, che ne tiene la prova di
rivalutazione e l'arena delle casse.

## Perché

Il cervello a sezioni oggi **reagisce**: guarda un oggetto, ne legge il valore e
sceglie un programma. Non ha modo di pensare "prima faccio questo, poi quello",
né di immaginare che cosa succederebbe. Aggiungere altre sezioni reattive non
colma questa lacuna.

Pianificare richiede tre capacità che oggi mancano:

| capacità | domanda | sezione proposta |
|---|---|---|
| prevedere | se faccio questo con quella cosa, che cosa succede? | **M**, modello del mondo |
| immaginare | fra le azioni possibili, quale porta dove voglio? | **P**, piano |
| tenere a mente | a che punto sono di ciò che sto facendo? | **G**, scopi |

## Il criterio: che cosa conta come pianificare

Migliorare il punteggio non basta: anche un'abitudine imparata con molte prove
può eseguire una sequenza lunga. La prova che distingue i due casi è la
**rivalutazione**: si cambia una parte del mondo in un punto che l'azione non
tocca, e si guarda **la prima scelta**, prima di ogni nuova esperienza
(Tolman 1948; Daw e colleghi 2011).

Esempio. L'essere impara separatamente due fatti: l'interruttore X apre la cassa
Q (senza premio) e la cassa Q contiene pile buone (senza passare da X). Non ha mai
vissuto X → Q → pila di seguito. Alla prima prova:

- un'**abitudine** non ha motivo di preferire X: X non è mai stato premiato;
- un **piano** collega i due fatti e sceglie X subito.

**Definizione di successo: la preferenza alla prima prova è significativamente
sopra il caso, e sparisce con la lesione di M.**

## L'arena: oggi non serve pianificare

Nell'arena delle pile basta reagire: si vede una pila, ci si va e la si morde. Un
modello del mondo non avrebbe niente da aggiungere, e il suo effetto non si
potrebbe misurare. Serve un'arena in cui la ricompensa sia a più passi di distanza.

### Arena delle casse

Un'opzione dell'arena delle pile (`good_by` e il resto restano), non un'arena nuova:

- **Casse**: 2 tipi (P, Q). Chiuse, non si vedono le pile dentro. Aperte, liberano
  pile buone per qualche secondo.
- **Interruttori**: 2 tipi (X, Y). Un morso davanti a un interruttore apre una cassa.
- **La mappa interruttore → cassa** resta fissa per un regime e cambia ogni
  `T_regola` secondi, senza segnali.
- **Quale cassa è piena** cambia separatamente, ogni `T_cassa` secondi.

La ricompensa è quindi a tre passi: interruttore, cassa, pila. Le due regole
cambiano in momenti diversi, perciò un'abitudine fissa sulla sequenza si rompe, mentre
un modello ne aggiorna un pezzo solo. Una cassa aperta sbagliata non dà niente:
costa tempo ed energia, non dolore.

Per il cervello casse e interruttori sono **oggetti** come gli altri: servono due
caratteristiche nuove in `FEATURES` (per esempio "cassa", "interruttore"), più la
loro identità. O e S li trattano senza modifiche, e il resto dei sensi non cambia.

Le regole del mondo vanno in `arenas/pile/world.py` e nella sua copia per JAX in
`arenas/pile/gpu.py`.

### Prima prova fissa

La rivalutazione non può dipendere dal caso: è una prova scriptata, come
l'esame nelle altre arene (`arenas/pile/bench.py`):

1. **Fatto 1:** l'essere prova gli interruttori e vede quali casse si aprono. Tutte
   vuote, nessun premio.
2. **Fatto 2:** riceve casse già aperte, una piena e l'altra no.
3. **Prova:** tutto chiuso, gli interruttori davanti. Si misura la prima scelta.

## Le nuove sezioni

Valgono le regole del cervello a sezioni: si fissa la struttura, non il contenuto.
I numeri sono proposte.

### M. Modello del mondo

Impara **che cosa segue a che cosa**.

**Struttura.** Una tabella a livello di oggetti, come O: per (caratteristiche
dell'oggetto, programma) prevede le caratteristiche dell'oggetto o dell'esito che
viene dopo. Lineare, cioè `6 caratteristiche × 4 programmi → caratteristiche
successive`, più una riga per "non succede niente".

**Regola.** Regola delta sull'errore di previsione, a ogni evento: lo stesso
principio di O, ma dell'oggetto invece che del valore. Non serve premio: M impara
anche senza ricompensa, ed è ciò che rende possibile la rivalutazione.

**Eventi, non passi.** M non prevede il passo dopo: prevede l'evento dopo (l'oggetto
è cambiato, si è aperta una cassa, è arrivato un esito). Pianificare a passi da 0,05 s
sarebbe inutile e costoso.

**Contesti.** Un cassetto di M per contesto di K, come per O: la mappa degli
interruttori cambia con i regimi.

**Si verifica così:** l'errore di previsione cala nella vita, ed è più alto subito
dopo un cambio di regola. Se ha un picco che K non vede, va collegato a K.

**Alternativa scartata per ora:** una rete sullo stato di B. È più generale, ma B
oggi non porta informazione su ciò che conta (docs/SPEC_ARENA_PILE.md), quindi
erediterebbe quel problema.

### P. Piano

Immagina e confronta.

**Struttura.** Non ha pesi propri. Ogni volta che E decide, parte dall'oggetto
attento e simula con M, per ogni programma, fino a `plan_depth` eventi (2 o 3). Il
valore di ogni simulazione lo dà O. L'esito è un voto per ogni programma.

```
voto(p) = valore atteso dalla catena simulata con il primo passo p
```

Il voto si somma in E, con un peso `plan_weight` (gene) spostato dall'arbitraggio.
Come C, entra direttamente nei punteggi: non deve passare dai pesi da O a E, che
restano quasi a zero.

**Costo.** Immaginare richiede tempo. Si valuta se il piano ritarda la decisione o
costa energia: altrimenti non c'è motivo di preferire l'abitudine, e il confronto
tra i due sistemi perde senso.

**Arbitraggio.** Si usa il sistema più affidabile (Daw, Niv, Dayan 2005): P pesa di
più quando M prevede bene, meno quando il suo errore è alto. Con un'abitudine solida
E vince perché costa meno.

### G. Scopi

È la "lista delle cose da fare", che oggi non esiste.

**Struttura.** Uno scopo attivo (le caratteristiche dell'oggetto cercato), tenuto
come attività, alla maniera di D, ma con tempo più lungo. Quando P trova una catena
utile, ne scrive il primo sotto-scopo. All'arrivo dell'evento atteso, lo scopo passa
al successivo. Niente pesi nuovi nella prima versione.

**Verso S.** Lo scopo aumenta il punteggio degli oggetti che gli somigliano: è
l'attenzione guidata dallo scopo. `Attention.update` ha già il parametro `extra`
per un punteggio in più: si usa quello. Senza, l'essere dimentica che cosa cerca
appena guarda altro.

**Si verifica così:** con un distrattore in mezzo a una sequenza, l'essere riprende
dal punto in cui era. Si misura confrontando con la lesione di G.

## Il flusso

```
oggetto attento ─► P ──simula con──► M (che cosa segue)
                   │                 O (quanto vale)
                   ├─► voti in E ─► programma
                   └─► G ─► S (che cosa cercare)

evento osservato ─► M impara (anche senza premio)
```

## Prerequisiti

Prima di M conviene sistemare tre punti già emersi: M impara da chi aveva davanti
quando è successo qualcosa, quindi erediterebbe gli stessi errori.

1. **Foto dell'episodio al morso.** Oggi nella foto compare anche l'altro colore nel 34%
   delle pile buone.
2. **Sorpresa all'inizio dell'esito** per C ed E, che durante il consumo vengono
   premiati mentre guardano altro nel 44% dei passi.
3. **Riflesso del morso**, se serve: mordere a vuoto costa energia e peserebbe di più
   in un'arena più lunga.

Il punto 1 è il più importante per M. Gli altri due vanno provati sul banco.

## Fasi di lavoro

Ogni fase si chiude con una misura. Se la misura fallisce, si cambia la struttura
prima di proseguire.

| fase | lavoro | misura per passare alla fase dopo |
|---|---|---|
| 0 | prerequisiti, poi arena delle casse e prova fissa | il guidato con la regola nota raccoglie molto; il cervello attuale no, e alla prova sceglie a caso |
| 1 | M da solo, senza effetto sul comportamento | l'errore di previsione cala; M prevede la mappa degli interruttori |
| 2 | P con profondità 1 e 2 | **rivalutazione superata**; la lesione di M la annulla |
| 3 | G e attenzione guidata dallo scopo | le sequenze reggono ai distrattori |
| 4 | arbitraggio e costo del pensare | l'abitudine prende il posto del piano nei regimi stabili |
| 5 | fase B: l'evoluzione tara `plan_depth`, `plan_weight` e il costo | si misura se piano e abitudine si mantengono entrambi |

Come per le sezioni attuali, la prima versione è in NumPy e la copia per JAX viene
dopo, una volta stabile la struttura. La simulazione di P ha profondità fissa, così
può diventare un ciclo compilato.

## Nel codice

| che cosa | dove |
|---|---|
| casse, interruttori, regole | `arenas/pile/world.py`, poi `arenas/pile/gpu.py` |
| prova di rivalutazione | `arenas/pile/run.py` e `arenas/pile/bench.py` |
| M e P | nuovi `brains/pile/model.py`, `brains/pile/plan.py` |
| G | nuovo `brains/pile/goals.py`, collegato a `attention.py` |
| voto del piano | `Sections.step` in `brains/pile/sections.py` |
| nuove lesioni | "M" (niente modello), "P" (niente piano), "G" (niente scopi) |
| schermata | `viz/pile/view.py` |
| geni nuovi | `Genes` in `brains/pile/genes.py` |

Aggiungere caratteristiche e geni cambia la forma del genoma e di A: i salvataggi
di oggi non si riapriranno senza una conversione (`FORMAT` in `arenas/pile/saves.py`).

## Confronti

| agente | che cosa dovrebbe fare |
|---|---|
| guidato con la regola | tetto: sa tutto |
| attore-critico | pavimento: impara solo per rinforzo |
| sezioni di oggi | dovrebbe imparare la catena lentamente, per abitudine, e fallire la prova |
| sezioni con M e P | deve superare la prova |
| sezioni con lesione di M | deve tornare a fallirla |

## Rischi

- **Troppo scritto a mano.** Una tabella `oggetto × programma` è ancora contenuto
  dato da noi. È accettabile come primo passo, per sapere se il concetto funziona;
  il passo dopo è un modello che legge da stati imparati.
- **L'abitudine basta.** Se le regole cambiano piano, E impara la catena e P non
  serve. La prova fissa protegge da questo errore di misura.
- **Il mondo è troppo povero.** Con due interruttori si può ottenere molto a caso.
  Va tarato in fase 0, prima di costruire M.
- **Rumore di attribuzione.** Se la fase 1 non impara, la causa può essere
  l'attribuzione, non M: per questo vengono prima i prerequisiti.
- **Il modello non è il piano.** M può funzionare senza che P produca effetti: le
  due fasi vanno misurate separatamente.

## Riferimenti da tenere presenti

- Tolman (1948): mappe cognitive e apprendimento latente.
- Daw, Niv, Dayan (2005) e Daw e colleghi (2011): arbitraggio, compito a due stadi.
- Sutton (1991), Dyna: un modello imparato usato per pianificare.
- Hafner e colleghi (Dreamer), Schrittwieser e colleghi (MuZero): pianificare con
  modelli imparati, su scala grande.
- Botvinick e Niv (apprendimento gerarchico): scopi e sotto-scopi.

## Decisioni aperte

1. **Ordine:** M, P e G a uno a uno come sopra, o un primo prototipo minimo di tutti e
   tre? Proposta: a uno a uno.
2. **Arena:** opzione dell'arena delle pile (proposta) o arena separata?
3. **Quanta struttura fissare in M:** tabella di oggetti (proposta) o stati imparati?
4. **Costo del pensare:** tempo, energia o nessuno nella prima versione?
5. **Spazio:** le casse richiedono di ricordare dove stanno? Il senso dello spazio manca
   (docs/SPEC_ARENA_PILE.md). Proposta: nella prima versione stanno in vista, e la
   memoria dei luoghi viene dopo.
6. **Il maestro:** gli esseri possono imparare M anche guardando il maestro?
