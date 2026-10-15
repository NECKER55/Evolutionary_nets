# Specifica: lo sviluppo del cervello (genoma indiretto)

Stato: **bozza da approvare** (2026-10-14). Nulla di quanto descritto qui è
ancora nel codice.

Scelte dell'utente:

- genoma indiretto **"identità + affinità"**: i pesi non sono più scritti
  sinapsi per sinapsi, ma nascono dai marcatori dei neuroni e da una tabella di
  affinità (chemoaffinità di Sperry);
- **attività spontanea prima della nascita**, con la plasticità accesa;
- **marcatori pochi all'inizio**, che si duplicano e divergono;
- **scaling omeostatico** con i suoi geni;
- **prima un confronto**: esperimento di riferimento con il genoma diretto, poi
  lo sviluppo, poi lo stesso esperimento.

---

## 1. Riferimenti biologici

- **Chemoaffinità** (Sperry, 1963): i neuroni si collegano ai partner le cui
  molecole di superficie riconoscono. Le famiglie principali: Dscam (circa
  19.000 varianti da un gene, nel moscerino), protocaderine a grappolo (circa
  58 geni nell'uomo, espresse in combinazioni casuali), caderine (oltre 100),
  Eph ed efrine (gradienti che costruiscono le mappe topografiche), DIP e Dpr
  (coppie che decidono quali tipi di neuroni fanno sinapsi). Sono famiglie nate
  per duplicazione di geni.
- **Sviluppo in parte casuale**: individui geneticamente identici hanno cervelli
  simili ma non uguali sinapsi per sinapsi.
- **Attività spontanea** prima dell'esperienza (onde della retina): rifinisce il
  cablaggio con regole di tipo Hebb.
- **Scaling omeostatico** (Turrigiano, 1998): un neurone ridimensiona tutte le
  sinapsi che riceve per mantenere la propria attività vicino a un livello.

## 2. Cosa contiene il genoma

| Parte | Contenuto | Diretto o indiretto |
|---|---|---|
| recettori | tutti i geni di oggi + **marcatori** | diretti |
| neuroni interni | esistenza, posizione, geni propri (soglia, plasticità, tipo, modulatore) + **marcatori** + **geni dell'omeostasi** | diretti |
| motori | geni propri + **marcatori** + geni dell'omeostasi | diretti |
| **tabella di affinità** | quanto ogni marcatore sull'assone si lega a ogni marcatore sul bersaglio (con segno) | gene dell'individuo |
| marcatori esistenti | quali "geni di marcatore" esistono (posti che crescono) | gene dell'individuo |
| meccanismo di mutazione | come oggi, più il tasso di duplicazione dei marcatori; tolti comparsa e scomparsa dei collegamenti | geni dell'individuo |
| **pesi e collegamenti** | **non ci sono più**: nascono con lo sviluppo | — |

I marcatori sono quantità di espressione tra 0 e 1. All'inizio ce ne sono 2.

## 3. Lo sviluppo: come nasce un cervello

Per ogni individuo, alla nascita:

1. **Posizionamento.** Ogni neurone va dove dice il suo gene; i nervi dei
   recettori entrano dal bordo sinistro; i motori stanno a destra.
2. **Riconoscimento.** Per ogni coppia sorgente s → bersaglio t:

   ```
   affinità(s, t) = Σ_a Σ_b  marcatore_s[a] · tabella[a, b] · marcatore_t[b]
   ```

3. **Crescita degli assoni.** Il collegamento nasce con probabilità

   ```
   p = net_conn_prob · (1 − e^(−|affinità|)) · e^(−distanza / λ)
   ```

   Senza affinità non nasce nulla; con affinità forte la probabilità satura; la
   distanza conta come oggi. λ è il gene di oggi. Un neurone con se stesso usa
   `net_self_prob` al posto della parte con la distanza.
4. **Forza iniziale.** Il peso iniziale è l'affinità, limitata a ±`synapse_max`.
   Nelle reti a impulsi il segno viene dal tipo del neurone che trasmette
   (legge di Dale) e l'affinità dà la forza.
5. **Attività spontanea.** Per `prenatal_steps` passi (proposta: 300, cioè 5
   secondi) i recettori ricevono solo rumore casuale; la rete gira; la
   plasticità agisce con M = 1 (nel grembo non c'è ricompensa) secondo i geni
   della regola di ogni neurone; agisce anche l'omeostasi.
6. **Nascita.** Il cervello risultante è il **fenotipo** dell'individuo.

Il sorteggio usa il generatore della corsa: la corsa resta ripetibile con lo
stesso seme.

**Il cervello sviluppato appartiene all'individuo.** Un genitore che sopravvive
tiene il suo e non si risviluppa; i figli ne sviluppano uno nuovo. A ogni
arena si riparte dal cervello appena nato (ciò che si impara durante la vita si
perde, come oggi). I salvataggi contengono genomi e cervelli sviluppati.

## 4. Scaling omeostatico

Ogni neurone interno e ogni motore tiene una media recente della propria
attività ā (con tempo caratteristico di pochi secondi) e, a ogni passo,
moltiplica tutte le sinapsi che riceve per

```
exp(h · dt · (a* − ā))
```

- `a*` è il livello di attività desiderato, `h` la forza dell'omeostasi. Sono
  **geni del neurone**.
- Agisce prima della nascita e durante la vita.
- Conseguenze: un neurone duplicato raddoppia i suoi effetti, e i neuroni a
  valle li riportano gradualmente in un intervallo stabile. Una sinapsi che
  impara troppo viene ridimensionata insieme alle altre. Con h = 0 non succede
  nulla: l'evoluzione può spegnerla.

## 5. Mutazioni e incrocio

| Mutazione | Effetto |
|---|---|
| deriva dei marcatori di un neurone | cambia i suoi partner in modo coerente, come una cellula che cambia tipo |
| deriva della tabella di affinità | cambia il cablaggio di tutti i neuroni che esprimono quei marcatori |
| **duplicazione di un marcatore** | il nuovo marcatore parte uguale all'originale, in tutti i neuroni e nella tabella: nessun cambiamento finché non diverge |
| perdita di un marcatore | i suoi legami spariscono |
| duplicazione di un neurone o recettore | la copia ha gli stessi geni e marcatori, posizione vicina; i collegamenti si rifanno con lo sviluppo (non è più esattamente neutra: la compensa l'omeostasi) |
| accensione o spegnimento, posizione, geni propri, geni dei recettori | come oggi |

Tutti i tassi restano geni, e mutano con il meccanismo dell'individuo stesso.
Spariscono le mutazioni "compare o sparisce un collegamento", perché i
collegamenti non sono più geni.

**Incrocio:** ogni neurone arriva intero da un genitore (marcatori compresi),
ogni recettore pure, la tabella di affinità intera da un genitore, il
meccanismo di mutazione intero da un genitore. I marcatori si abbinano per
posizione, come i posti dei neuroni.

## 6. Partenza

- 2 marcatori; la tabella di affinità e i marcatori di ogni neurone, recettore
  e motore partono a caso.
- Corpo e cervello primordiali come oggi: 2 recettori generalisti, nessun
  neurone interno acceso.
- Omeostasi debole e casuale (come l'adattamento sensoriale).

## 7. Cosa cambia nel resto

- **Configurazione**: `genome = "diretto" | "sviluppo"`. Il genoma diretto resta,
  per il confronto e per riaprire le corse di oggi.
- **Salvataggi**: formato 6, con il cervello sviluppato di ogni individuo.
- **Viewer**: con 2–3 marcatori, il colore di ogni neurone mostra la sua
  identità, così i **tipi cellulari** si vedono a colpo d'occhio. L'esploratore
  mostra la tabella di affinità e, toccando un neurone, i suoi marcatori.
- **Statistiche**: numero di marcatori, e quanti gruppi di neuroni con identità
  simile (tipi) ci sono.
- **Calcolo**: lo sviluppo costa una volta per nascita, più i passi di attività
  spontanea (circa il 10% di una vita in più per individuo nato).

## 8. Il confronto

1. **Riferimento** (in corso dal 2026-10-14): genoma diretto, ambiente di oggi,
   reti continue e a impulsi, semi 31–34, 2.000 generazioni, esame ogni 100.
2. **Sviluppo**: implementazione di questa specifica, con i suoi test.
3. **Stesso esperimento** con `genome = "sviluppo"`.
4. **Si confrontano**:
   - l'esame nel tempo;
   - la dimensione delle reti;
   - quanto spesso il cervello esplode;
   - se emergono tipi cellulari e simmetrie;
   - la robustezza ai cambi di corpo (recettori duplicati che si cablano da
     soli).

## 9. Limiti noti

- La forma delle formule (affinità come somma di prodotti, probabilità che
  satura) è una scelta nostra: è la "chimica" del simulatore.
- Con pochi marcatori i pesi iniziali hanno pochi gradi di libertà (al massimo
  tanti quanti i marcatori). Li aumentano la duplicazione dei marcatori,
  l'attività spontanea e l'apprendimento.
- Non c'è crescita da una cellula sola (divisione, migrazione, gradienti): il
  genoma contiene ancora l'elenco dei neuroni. È una base per farla in futuro.
