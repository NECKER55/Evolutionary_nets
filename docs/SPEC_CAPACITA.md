# Test di capacità del cervello

Domanda: il cervello di oggi, con la sua plasticità, è in grado di imparare
durante la vita? Se no, nessun mondo può far emergere l'apprendimento.
Codice: `evonets/evo/capacity.py`
(`uv run python -m evonets.evo.capacity --part meccanismo|evoluzione|tutto`).

## 1. Il meccanismo, sinapsi per sinapsi

Una sinapsi da un senso a un motore; i due neuroni attivi insieme per 1 s;
dopo un ritardo arriva piacere (gusto) o dolore. Si misura la sinapsi
rispetto a chi non riceve niente, con la regola vera (`brain.plasticity`).
"Distingue" = quota di cervelli in cui il piacere la rafforza più del dolore
(50% = a caso).

Prima della traccia di eleggibilità (2026-10-25):

| | ritardo 0 | ritardo 1-40 s |
|---|---|---|
| geni di partenza | 53% | 51% |
| regola giusta a mano (pre·post·M) | 100% (+1,7% / −2,1%) | 0%: piacere e dolore identici |
| geni evoluti nell'arena (164 min) | 65%, ±1,1% | 65%, ±1,1% a ogni ritardo |

- La regola modulata usava l'attività dell'istante: con un secondo di ritardo
  il segno del segnale non contava più.
- La traccia provvisoria (30 s) è dell'hebbiano e il modulatore la consolida
  con |M|: ricorda gli eventi importanti, non quale azione ha funzionato.
- Nei geni evoluti l'effetto non dipende dal ritardo: il piacere rafforza
  tutte le sinapsi insieme (un "umore", termine costante della regola).

Dopo la traccia di eleggibilità (τ_e di partenza 2 s), regola giusta a mano:
distingue nel 100% dei cervelli fino a 10 s di ritardo (+0,85% / −1,05% a
ritardo 0, +0,32% / −0,34% a 2 s, +0,11% / −0,07% a 5 s); i geni di partenza
restano a caso (la regola è nei geni: la deve trovare l'evoluzione).

## 2. L'evoluzione su un compito che si risolve solo imparando

Il mondo del cibo con metà frutti cattivi, e quale colore è cattivo cambia a
caso in ogni vita; bisogna mangiare 4 frutti buoni. La scelta è la quota di
frutti buoni sul totale dei mangiati (50% = a caso). Ogni 25 generazioni 300
individui si provano con e senza plasticità, nelle stesse arene.

**Attenzione alla misura.** La prima versione contava tutti i frutti
insieme. Chi preferisce un colore per istinto, nelle vite in cui quel colore
è il buono, vive di più e mangia di più, e pesa di più: sembra scegliere bene.
Esempio (sensi fissi, generazione 30): buono rosso 44%, buono giallo 60%, in
tutto "54-65%"; in realtà preferiscono sempre il giallo (media dei due casi
52%, il caso). Ora la scelta è la media dei due casi.

Con la misura vecchia (quindi da rileggere così):
- regola di prima, 300 generazioni (spike-ei, genoma diretto): 49-52% fino
  alla 250, poi 57-69%, uguale con e senza plasticità: quasi certamente la
  preferenza per un colore;
- controllo evoluto senza plasticità: 49-52%;
- regola nuova (traccia di eleggibilità), 500 generazioni: 47-53% con e senza
  plasticità per tutta la corsa. Usano un solo senso (1,0 in media), 0-0,2
  neuroni interni, ~7 collegamenti: per imparare quale colore è buono
  servono insieme un recettore che distingua i colori, un collegamento dal
  recettore al movimento e una regola con il segno giusto; da soli non
  servono (valle di fitness).

Con la misura bilanciata, sensi fissi, 500 generazioni:
- **velocità ×30, regola a caso** (prova A): dalla generazione 50 istinto
  "mangia il rosso" (buono rosso 86-90%, buono giallo 10-16%), scelta 0,49-0,52,
  con e senza plasticità uguali: la velocità da sola non basta;
- **velocità ×30 e regola orientata** (prova B, A = 1 per tutti): fino alla
  200 scelta ~0,51 senza differenze; dalla 225 alla 500 con plasticità ~0,545,
  senza ~0,513 (meglio in 11 misure su 12). Il guadagno è nelle vite in cui il
  buono è il giallo, contro l'istinto (21% -> 27%): impara a correggere
  l'istinto, non lo sostituisce.

Conclusione: la barriera principale era trovare la regola. Da qui la regola
innata (`plasticity_innate_rule`: A in media 1 alla partenza); la velocità
resta quella di prima. Aperto: perché l'effetto è piccolo (+3 punti).

### Regola innata e sorpresa (500 generazioni, sensi fissi)

Differenza della scelta con e senza plasticità, dalla generazione 225 alla 500
(12 misure):

| prova | velocità | sorpresa | differenza media | misure a favore |
|---|---|---|---|---|
| B | ×30 | no | +2,9 punti | 11/12 |
| D | ×1 | no | +1,4 | 12/12 |
| C | ×30 | sì | −1,5 | 0/12 |
| E | ×1 | sì | −1,7 | 1/12 |

- La regola innata funziona anche alla velocità normale (D), con metà
  dell'effetto di ×30.
- La sorpresa (critico TD, `surprise = 1`) peggiora: la plasticità diventa
  un danno, soprattutto nelle vite in cui il buono è il giallo. Il segnale,
  appreso il legame, lascia il momento del boccone (quando la sinapsi
  colore -> movimento è pronta a cambiare) e passa alla comparsa del colore,
  quando ciò che si muove nella traccia non è ancora l'avvicinamento. Resta
  nel codice spenta (`surprise = 0`) per altre prove.

### Dove si perde l'apprendimento (prova D, generazione 300, 300 vite)

- **Quanto cambia.** Alla velocità normale la parte modulata cambia la
  sinapsi di circa lo 0,3% per frutto; l'hebbiano di base la sposta del 10%
  in una vita, a caso rispetto al valore: il segnale affoga nel rumore.
  L'evoluzione ha anzi dimezzato η (motori: 0,0005 invece di 0,001) e ha
  dato alla regola termini che non dipendono dal senso (C −0,53, D +0,32).
- **Cosa cambia.** Imponendo la regola pura (A = 1) e η ×100, anche senza
  hebbiano, le sinapsi degli occhi cambiano del 30-45% in una vita, ma quelle
  dell'occhio del colore buono quanto (o più di) quelle del cattivo, e il
  comportamento quasi non cambia: dopo il primo frutto cattivo il successivo
  è cattivo nel 46% dei casi (senza plasticità 50%); nelle vite in cui
  l'istinto è sbagliato (buono il giallo) 71% contro 71%. Il cambiamento non
  colpisce il circuito che fa la scelta.
