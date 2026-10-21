# Specifica: lo sviluppo del cervello (genoma indiretto)

Stato: **approvata**, rivista il 2026-10-21 (sezioni 10 e 11). Nel codice
c'è la plasticità durante la vita (sezione 10); lo sviluppo vero e proprio
(marcatori, tabella, grembo) non ancora: il piano è nella sezione 11.

Scelte dell'utente (2026-10-14):

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
- Il cervello sviluppato e la sua plasticità costano calcolo: lo sviluppo si fa
  a ogni nascita, l'attività spontanea aggiunge circa il 10% di una vita per
  ogni nato.
- Con pochi marcatori i pesi iniziali hanno pochi gradi di libertà (al massimo
  tanti quanti i marcatori). Li aumentano la duplicazione dei marcatori,
  l'attività spontanea e l'apprendimento.
- Non c'è crescita da una cellula sola (divisione, migrazione, gradienti): il
  genoma contiene ancora l'elenco dei neuroni. È una base per farla in futuro.

---

## 10. Revisione del 2026-10-21

### Perché ora

Le prove del 2026-10-20/21 (docs/SPEC_MONDO.md, sezioni 10–13, e il controllo
con lo XOR) hanno mostrato che con il genoma diretto un neurone utile richiede
5–6 geni giusti insieme (acceso, collegato a entrambi i sensi e al motore, pesi
e soglia precisi): con la selezione "tutto o niente" nessun neurone utile in
900.000 nascite, con la selezione graduata neuroni che si accumulano ma nessuna
soluzione in 900 generazioni. Il genoma che scrive ogni sinapsi è la parte meno
naturale del simulatore, ed è proprio quella che blocca la crescita del
cervello. Polyworld (Yaeger) genera cervelli funzionanti con un genoma che
descrive regole (gruppi, densità, velocità di apprendimento) e lascia i pesi
all'apprendimento.

### Scelte dell'utente (2026-10-21)

1. **Apprendimento hebbiano di base, sempre attivo** (in tutti i casi, anche con
   il genoma diretto): oltre alla plasticità modulata (che resta, e dice quando
   imparare), ogni sinapsi effettiva cambia come

   ```
   Δw = η_h · pre · post
   ```

   con η_h **gene del neurone che riceve** (partenza debole e diversa da
   individuo a individuo, può scendere a zero). Lo tiene in un intervallo
   stabile lo scaling omeostatico (sezione 4) e il tetto `synapse_max`; nelle
   reti a impulsi vale la legge di Dale. Agisce anche nell'attività spontanea
   prima della nascita: oggi la plasticità agisce solo se c'è un neurone
   modulatore, e i modulatori in tutte le corse sono rimasti a zero.
2. **Pesi = affinità × guadagno del neurone**: ogni neurone (interno o motore)
   ha un gene "quanto pesano gli ingressi che ricevo" (come il numero di
   recettori postsinaptici, in scala logaritmica). Il peso iniziale di s → t è
   affinità(s, t) · guadagno_t, limitato a ±`synapse_max`. Più gradi di libertà
   restando indiretto: il genoma descrive i neuroni, non le sinapsi.
3. **Sensi fissi** (`sim_mode = "cervello"`): anche i marcatori dei sensi sono
   fissi (uguali per tutti, mai mutati): il cervello deve adattarsi a sensi del
   tutto dati. Si estraggono una volta, con un seme fisso del set di sensi.
   Quando un marcatore si duplica, la copia nei sensi vale quanto l'originale e
   poi non deriva. Nella simulazione completa i marcatori dei recettori
   evolvono come quelli dei neuroni.
4. Le altre parti della specifica valgono **in entrambi i tipi di simulazione**.

### Aggiornamenti dovuti al codice di oggi

- **Formato dei salvataggi**: 12 (non 6; il 9 è l'hebbiano, il 10 l'omeostasi,
  l'11 la traccia provvisoria). I salvataggi di oggi si aprono con
  `genome = "diretto"`.
- **Incrocio tra omologhi**: anche i marcatori portano identità e famiglia (come
  i geni dei neuroni). La tabella di affinità si incrocia riga per riga,
  abbinando i marcatori per identità; un marcatore di un solo genitore passa con
  probabilità 1/2. Vale lo scambio ineguale tra marcatori della stessa famiglia.
- **Riproduzione, riserve, fecondità, sesso come gene**: invariati.
- **Il controllo prima del mondo**: lo XOR con la scala (OR poi XOR), regola
  graduata e regola a soglia, genoma diretto contro sviluppo. Costa minuti.
  Domanda: con lo sviluppo nascono neuroni utili dove la codifica diretta
  falliva? Poi la prova "un neurone in più" sulla popolazione di oggi.
- **Il confronto nel mondo** (sostituisce la sezione 8): regole fisse e
  dichiarate per tutta la durata, almeno 4 semi per condizione, prima in
  modalità "solo cervello" (meno variabili), condizioni: genoma diretto,
  diretto + hebbiano di base, sviluppo. Misure decise prima: esame a livelli,
  neuroni interni usati (lesione "neuroni"), bravura vera su 20 arene ogni
  500 generazioni.

### Ordine di lavoro

1. Hebbiano di base (anche nel genoma diretto) nei due motori, con i test.
   **Fatto il 2026-10-21** (gene `hebb` in coda ai geni della plasticità,
   formato 9). Misura sulla popolazione di `corsa-20261020-200723`, evoluta
   senza: con velocità di partenza 1e-4 i maturi crollano dal 77% allo 0,3%
   (in 14.000 passi di vita la regola di Hebb, senza scaling omeostatico,
   deforma il riflesso); 1e-5 un poco dannoso; **1e-6 neutrale**, scelto come
   partenza. Lo scaling omeostatico (sezione 4) diventa quindi la condizione
   perché l'hebbiano possa crescere: va fatto prima o insieme allo sviluppo.
1b. Scaling omeostatico (sezione 4) nei due motori, con i test. **Fatto il
   2026-10-21** (geni `target` e `homeo` in coda alla plasticità, formato 10;
   ā = media di |attività| con τ = 5 s; si nasce con ā = a*, in attesa
   dell'attività spontanea). Misura sulla stessa popolazione: forza 1e-4
   neutrale (scelta come partenza), 1e-3 un poco dannosa, 1e-2 distruttiva
   (i geni convertiti chiedono a ogni neurone, motori compresi, attività 0,5).
   **L'omeostasi non salva l'hebbiano forte**: con hebbiano 1e-4 i maturi
   restano allo 0–0,3% con qualsiasi forza. L'omeostasi tiene il livello di
   attività, non il disegno dei pesi che fa il riflesso: l'hebbiano deve
   crescere insieme all'evoluzione (che sceglie a*, h, η_h insieme), non
   essere imposto a una popolazione evoluta senza.
1c. Correzioni del 2026-10-21 (scelte dall'utente, "più simile possibile alla
   natura"), dopo aver visto che le due regole erano approssimate male:
   l'hebbiano solo rafforzava (nella rete a impulsi pre e post sono ≥ 0),
   indeboliva l'inibizione invece di rafforzarla e poteva invertire il segno
   delle sinapsi dei sensi; lo scaling rafforzava anche l'inibizione di un
   neurone troppo zitto. Ora: BCM con soglia mobile θ = ā²/a* sulle
   eccitatorie, Vogels (2011) sulle inibitorie, scaling opposto per le
   inibitorie (Turrigiano), e la plasticità (anche quella modulata) cambia
   solo la forza |w|, mai il tipo della sinapsi. Stessa misura: hebbiano 1e-4
   ora 76,2% di maturi (prima 0,3%), 1e-3 5%; partenza portata a 1e-5
   (neutrale). Parte del guadagno viene dal fatto che BCM, a parità di η_h,
   cambia i pesi meno di pre·post (il fattore post − θ). Restano scorciatoie
   dichiarate: tempi compressi (τ = 5 s invece di ore), si nasce con ā = a*,
   nessuna omeostasi dell'eccitabilità propria (la soglia).
1d. Contro la sovrascrittura, come in natura (2026-10-21, scelte dell'utente):
   traccia provvisoria dell'hebbiano (early-LTP) che svanisce con un tempo che
   è gene del neurone ("persist", partenza 30 s) e si consolida solo con un
   modulatore (|M|, tagging and capture); saturazione graduale dei
   rafforzamenti (1 − |w|/synapse_max); omeostasi dell'eccitabilità (la
   corrente di fondo, stessa forza h). Formato 11. Una sola funzione
   (`brain.plasticity`) per NumPy e GPU. Misura (maturi; 76,8% senza
   plasticità): hebbiano 1e-4 77,3% (partenza ora 1e-4), 1e-3 71%, ma 5,6% se
   la traccia è resa permanente: è la traccia provvisoria che impedisce la
   sovrascrittura. Resta: l'omeostasi forte (≥ 1e-3) danneggia le popolazioni
   convertite (a* = 0,5 per tutti i neuroni), e l'hebbiano forte dello
   sviluppo arriverà con l'attività spontanea.
2. Il resto del lavoro segue il piano della sezione 11 (approvato il
   2026-10-21), che sostituisce i vecchi punti 2–5.

## 11. Piano verso un cervello che si sviluppa come in natura (2026-10-21)

Approvato dall'utente il 2026-10-21 ("come in natura": le scelte aperte le
decide l'assistente, con l'opzione più naturale). Ogni fase si chiude con test,
misura di neutralità su una popolazione evoluta, documentazione e commit. Il
mondo (un mondo in cui imparare conviene, rumore della bravura, selezione a
soglia, durata delle vite, arena unica) si affronta dopo, a parte.

**Problemi da risolvere.** Dalla tabella del confronto con la natura:
(1) il genoma scrive ogni peso; (2) mancano tipi di cellule condivisi;
(3) manca l'attività spontanea con Hebb prima della nascita; (4) mancano i
periodi critici; (5) mancano segnali di valore innati; (6) l'architettura a
regioni deve poter emergere. Lasciati indietro prima: (7) l'omeostasi danneggia
le popolazioni convertite (a* = 0,5 per tutti); (8) si nasce con ā = a*;
(9) senza selezione i tassi di mutazione derivano e il cervello si gonfia;
(10) la rete continua ha attività negative, che non esistono; (11) omeostasi
delle sinapsi e della soglia con lo stesso gene; (12) tempi compressi (limite
accettato, legato al mondo); (13) controllo XOR, colori dell'identità,
confronto con più semi.

**Fase 0 — pulizia (7, 9, 10, 11).**
- (10) La rete continua: x = tanh(...) è lo scarto dalla frequenza
  spontanea; le regole di apprendimento usano la frequenza (1 + x)/2 ≥ 0 (e i
  sensi come sono, ≥ 0). La dinamica non cambia.
- (11) Gene proprio per l'omeostasi della soglia ("intrinsic").
- (7) Taratura alla conversione: aprendo un salvataggio senza il gene
  dell'attività desiderata, una vita di prova misura l'attività media di ogni
  neurone e a* la prende come valore (la popolazione era già in equilibrio).
- (9) Misurare se la deriva dei tassi viene dal limite minimo (sponda) e se
  con la selezione i tassi scendono (barriera della deriva, Lynch).

*Fase 0 fatta il 2026-10-21.* (10) frequenza = (1 + x)/2 nelle reti
continue, `brain.frequencies`. (11) gene "intrinsic", formato 12. (7) taratura
alla conversione (`saves.calibrate`, 10 s, circa 30 s di calcolo una volta);
inoltre ā su 120 s invece di 5 (punto 12: con 5 s l'omeostasi combatteva ogni
singolo comportamento). Misura sulla popolazione di corsa-20261020-200723
(maturi 76,8% senza plasticità): omeostasi 1e-2 dal 5% (a* = 0,5) al 38%
(tarato) e al 48% (tarato, 120 s); 1e-4 neutrale. Il danno che resta è
atteso: un'omeostasi forte imposta a una popolazione evoluta senza è un grande
cambio di fisiologia (1,5–2 volte i pesi in una vita); se l'omeostasi aiuti
quando evolve insieme al resto lo dice solo una corsa da zero. (9) senza
selezione i tassi derivano senza spinta (16 semi: spostamento medio 0 entro
l'errore; con il limite a 1e-12 è uguale): il gonfiamento viene
dall'asimmetria (una famiglia sopra la soglia cresce in modo esponenziale,
sotto resta a zero), come nei genomi a selezione rilassata. Con la selezione i
tassi scendono (corsa-20261020-200723: ampiezza 0,3 → 0,026, duplicazione
0,005 → 0,002). Nessuna modifica.

**Fase 1 — sviluppo (1, 2).** La specifica delle sezioni 2–5 e 10: marcatori
(2 all'inizio), tabella di affinità, probabilità di collegamento da affinità e
distanza, peso iniziale = affinità × guadagno, duplicazione e divergenza dei
marcatori, incrocio tra omologhi dei marcatori, `genome = "diretto" |
"sviluppo"`, marcatori dei sensi fissi in modalità "cervello", salvataggi con il
fenotipo. Scelta aperta: marcatori legati alla posizione (gradienti come
Eph/efrine, che in natura fanno le mappe).

*Fase 1 fatta il 2026-10-21* (net/development.py, formato 13, 198 test).
Scelte "come in natura": marcatori a gradiente inclusi (due geni per
marcatore, lungo x e y; i recettori stanno al bordo sinistro all'altezza del
nervo, quindi i gradienti in y danno mappe topografiche, salvo in modalità
"cervello", dove i sensi non li seguono); la duplicazione di un marcatore
divide a metà riga e colonna della tabella (affinità invariata, come la
duplicazione dei neuroni); un marcatore si rompe con lunghezza 3 + 2 ×
marcatori accesi; il guadagno è un gene proprio del neurone, dopo la
plasticità; nell'incrocio l'espressione di un marcatore in una cellula segue
la cellula, cercata per identità nell'altro genitore se quello della cellula
non ha il marcatore. Partenza: 4 posti, 2 accesi, espressioni N(0; 1,5) in
scala logistica, gradienti N(0; 2), tabella N(0; 2): alla nascita circa la
metà dei collegamenti del genoma diretto, con pesi un poco più forti. Il
genoma predefinito resta "diretto" finché la Fase 6 non dice se lo sviluppo
funziona; lo si sceglie dal menu (riga "Genoma"). Velocità uguale al diretto
(1,0 s/gen contro 1,2 con 300 individui a impulsi). Restano per le fasi dopo:
l'attività spontanea (Fase 2), i colori dell'identità (Fase 5).

**Fase 2 — grembo (3, 8).** Pochi secondi di attività spontanea prima della
nascita (rumore nei sensi, scelta: indipendente o a onde correlate tra sensi
vicini), con Hebb, omeostasi e soglia; la traccia si consolida da sola (lo
sviluppo fissa ciò che si forma); ā nasce dall'attività vera. Sulla GPU.

*Fase 2 fatta il 2026-10-21* (net/womb.py e gpu/womb.py, formato 14, 202
test). Onde che scorrono lungo `r_y` (velocità 0,5 lati/s, larghezza 0,1, una
ogni 1,5 s in media, da un capo o dall'altro) più rumore 0,05; 5 s; Hebb,
omeostasi e soglia accese, plasticità modulata spenta (niente ricompensa:
M = 0, non M = 1 come nella prima bozza della sezione 3); ā come media di
tutto il grembo (tempo min(homeo_tau, tempo trascorso)); alla nascita pesi,
ā e corrente di fondo diventano fenotipo (`born_avg`, `born_bias`), la
traccia provvisoria è già nei pesi. Solo con lo sviluppo. Sulla GPU con
`lax.scan` su tutta la popolazione (i non nati fermi): stessi risultati di
NumPy, nessun costo visibile (1,06 s/gen contro 1,09 del diretto). Con le
velocità di partenza di oggi (Hebb 1e-4) il grembo cambia i pesi dello 0,6%:
risolve il punto 8 (ā vera alla nascita: 0,40 contro a* 0,50 nella rete a
impulsi), ma la rifinitura del cablaggio diventa forte solo con la Fase 3,
dove la plasticità hebbiana è massima prima della nascita.

**Fase 3 — periodi critici (4).** η_h(età) = η_h · (r + (1 − r) · e^(−età/τ_c)),
con τ_c e r geni del neurone; la plasticità modulata resta aperta.

**Fase 4 — valore innato (5).** Sensi del gusto (quando si mangia) e del dolore
(danno); in modalità "cervello" due neuroni di valore fissi (gusto → M
positivo, dolore → M negativo, come dopamina e noradrenalina). Scelta aperta:
se nella simulazione completa partono presenti o evolvono da zero.

**Fase 5 — vedere tipi e regioni (2, 6, 13).** Colore dell'identità dei
neuroni, tabella di affinità nell'esploratore, statistiche (marcatori, tipi,
uso dei modulatori). L'architettura non si impone.

**Fase 6 — verifiche (13).** Controllo con lo XOR (diretto contro sviluppo);
"un neurone in più"; confronto con più semi in modalità "cervello" (diretto,
diretto + plasticità, sviluppo, sviluppo + valore), regole bloccate, misure
decise prima. Le corse lunghe partono solo con il via dell'utente.

```
Fase 0 ──► Fase 1 ──► Fase 2 ──► Fase 3
             │                     │
             └──► Fase 5           ▼
Fase 4 (in parallelo) ──────► Fase 6 ──► Mondo
```
