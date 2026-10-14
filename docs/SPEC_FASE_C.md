# Specifica: fasi C e D — sensi, cervello e mutazioni come geni

Stato: **approvata** (2026-10-14), in corso di implementazione.
- Passo 1 (fisica dei segnali e risposta dei recettori): fatto, in
  `sim/signals.py` e `net/receptors.py`, con i test. I cicli pesanti sono
  compilati con Numba.
- Passo 2 (nuovo genoma: recettori nel genoma, posti che crescono,
  duplicazione neutra, tassi di mutazione come geni, partenza primordiale,
  salvataggi formato 3, interfaccia ed esame versione 2): fatto. Il rumore dei
  recettori riparte dal seme di ogni arena, così una corsa ripresa resta
  identica.
- Adattamento sensoriale (sezione 3.3 bis): fatto; salvataggi formato 4.
- Passo 3: i costi attuali risultano già in linea con l'obiettivo (un occhio di
  25 recettori ≈ 12% del metabolismo di base); esperimento di validazione in
  corso (ambiente facile, reti continue e a impulsi, 4 semi ciascuna).

Decisioni già prese (2026-10-14):

- recettore **generalista**: nessun tipo di senso definito da noi, solo la fisica;
- partenza **solo primordiale**: pochi recettori generalisti, cervello minimo;
- luce in **mezzo torbido**: si attenua con la distanza;
- sensibilità pagata con il **rumore**, come per i fotoni;
- ogni "tetto" fissato da noi diventa un gene o un costo, il più possibile come in natura;
- l'ambiente lo cambia l'utente, a mano.

Le scelte ancora aperte sono raccolte nell'ultima sezione.

---

## 1. Principio

| Lo decidiamo noi (è la fisica) | Lo decide l'evoluzione (è biologia) |
|---|---|
| quali fenomeni esistono: luce, sostanze, meccanica, temperatura, magnetismo, grandezze del corpo | quanti recettori, dove, puntati dove |
| come si propagano | a cosa è sensibile ogni recettore, e quanto |
| cosa emette ogni oggetto del mondo | quanti neuroni, dove, collegati come |
| quanta energia costa ogni cosa | quanto mutano i figli |
| la regola di selezione, la dimensione della popolazione | |

Un "occhio", un "orecchio", un "naso" non esistono nel codice: sono nomi che
daremo noi a gruppi di recettori che l'evoluzione avrà specializzato.

---

## 2. La fisica dei segnali

Ogni fenomeno ha un **asse della qualità** `q` in [0, 1]: il colore per la luce,
la "forma" della molecola per le sostanze, la frequenza per la meccanica, la
temperatura per il caldo, quale grandezza per l'interno. Un recettore è
sintonizzato su un punto di quell'asse.

**Cosa emette ogni oggetto** (colore, molecola e quantità, frequenza e ampiezza
delle vibrazioni, calore) è una costante della configurazione. Aggiungere il
suono al bersaglio, l'odore al veleno o una sorgente di calore significa
cambiare un numero, senza toccare il codice dei sensi.

### 2.1 Luce (mezzo torbido)

- **Cosa emettono gli oggetti.** Ogni oggetto ha uno spettro di riflessione,
  cioè una campana sull'asse del colore. Tutto è illuminato in modo uniforme.

  | Oggetto | Colore (centro, larghezza) | Albedo | Comportamento |
  |---|---|---|---|
  | pareti | 0,50, larga (0,30) | 0,4 | opache |
  | bersaglio | 0,85, stretta (0,06) | 1,0 | opaco |
  | veleno | 0,25, stretta (0,06) | 0,6 | nube che brilla e non nasconde ciò che sta dietro, come oggi |

  I valori sono indicativi: si fissano nella calibrazione (sezione 10).
- **Attenuazione nel mezzo.** La luce che arriva da distanza `d` è moltiplicata
  per `exp(-d / L_luce)`, con `L_luce` circa 12 u. Un oggetto vicino è più
  luminoso: è così che la distanza entra nel segnale, senza telemetro.
- **Radianza lungo una direzione θ**, alla qualità q:

  ```
  R(θ, q) = S_opaco(q) · albedo · exp(-d_opaco / L_luce)
          + Σ_nubi  ρ · S_nube(q) · L_luce · [exp(-s_entrata / L_luce) - exp(-s_uscita / L_luce)]
  ```

  - `d_opaco` è la distanza del primo oggetto opaco lungo la direzione. Le
    pareti ci sono sempre, quindi ogni direzione colpisce qualcosa.
  - `s_entrata` e `s_uscita` sono i punti in cui la direzione entra ed esce da
    una nube, tagliati a `d_opaco`. Il termine di una nube è l'integrale esatto
    della sua luce lungo il tratto attraversato.
  - `ρ` è la densità luminosa della nube.

### 2.2 Sostanze

- **Cosa emettono le sorgenti.** Ogni sorgente emette una molecola, cioè un
  punto `q_m` sull'asse delle sostanze. La concentrazione è un campo statico
  calcolato con una formula, senza simulare la diffusione.
- **La tossina del veleno** (q = 0,15). Dentro una zona, la concentrazione è
  uguale all'intensità tossica di oggi (`profondità ** hazard_falloff`), quindi
  il danno non cambia. Fuori dalla zona:

  ```
  c = perdita · exp(-distanza_dal_bordo / L_sostanze)
  ```

  `perdita` vale **0 di default**: oggi il veleno non ha odore fuori dalla zona.
  Dargli un odore è un cambio d'ambiente, e quindi una tua scelta.
- **Il danno** è la concentrazione di tossina nel punto in cui sta l'agente:
  identico a oggi. Il "dolore" non esiste più come senso a parte. È un
  recettore sensibile alla tossina.

### 2.3 Meccanica

- **Pressione da contatto** (frequenza 0). Quando l'agente tocca una parete, un
  recettore che sta sul corpo all'angolo β riceve:

  ```
  p = max(0, n_β · n_parete)
  ```

  dove `n_β` è la normale del corpo nel punto del recettore e `n_parete` punta
  verso la parete. Il lato che tocca sente, quello opposto no.
- **Vibrazioni a distanza.** Le sorgenti hanno una frequenza `q_v` e
  un'ampiezza; l'ampiezza cala come `exp(-d / L_meccanica)`. Di default
  **nessuna sorgente**: aggiungere il suono è un cambio d'ambiente, per esempio
  "il bersaglio vibra a 0,6".
- **Direzione.** Una vibrazione che arriva dalla direzione θ_s pesa sul
  recettore secondo la sua finestra angolare (sezione 3.2): un recettore
  puntato verso la sorgente la sente di più.

### 2.4 Temperatura

- Un campo scalare: temperatura ambiente uniforme (0,5 sull'asse) più
  eventuali sorgenti di calore che si attenuano come `exp(-d / L_calore)`. Di
  default **nessuna sorgente**: senza sorgenti la temperatura è uguale ovunque
  e non dà informazione, ma il fenomeno esiste e l'ambiente può usarlo.
- Qui la qualità è la temperatura stessa. Un recettore risponde
  `T(temperatura locale)`: è massimo quando la temperatura è vicina al suo μ,
  come i recettori del caldo e del freddo veri (TRPV1, TRPM8).

### 2.5 Magnetismo

- Un campo uniforme con direzione e intensità, come quello di un pianeta.
  Intensità **0 di default**: dare un campo magnetico al mondo è un cambio
  d'ambiente.
- Un recettore risponde `max(0, B · direzione del recettore)`: è una bussola,
  come negli uccelli migratori e nelle tartarughe marine. Il magnetismo non ha
  un asse della qualità, quindi la sintonia non conta.

### 2.6 Interno (grandezze del corpo)

Le grandezze del corpo stanno in punti fissi dell'asse. Le grandezze con segno
sono divise in due metà, perché un recettore dà solo valori positivi (come la
frequenza di scarica di un neurone vero):

| q | grandezza | valore (da 0 a 1) |
|---|---|---|
| 0,05 | energia rimasta | energia / energia iniziale |
| 0,35 | velocità in avanti | max(0, v_avanti) / v_max |
| 0,45 | velocità all'indietro | max(0, -v_avanti) / v_max |
| 0,60 | derapata a destra | max(0, v_lat) / v_max |
| 0,70 | derapata a sinistra | max(0, -v_lat) / v_max |
| 0,85 | rotazione a destra | max(0, ω) / ω_max |
| 0,95 | rotazione a sinistra | max(0, -ω) / ω_max |

L'ordine sull'asse è una nostra scelta, e non è neutro: una sintonia che deriva
passa per le grandezze vicine. Le grandezze legate al movimento sono vicine tra
loro, l'energia sta lontana.

### 2.7 Fenomeni reali non inclusi, e perché

| Fenomeno | Perché no, per ora |
|---|---|
| campo elettrico (squali, pesci elettrici) | in natura lo emettono solo esseri viventi: servirà quando gli agenti potranno percepirsi tra loro |
| gravità, accelerazione verticale, pressione idrostatica | in un mondo 2D visto dall'alto non esistono; arrivano con il 3D |
| ciclo giorno-notte, stagioni | sono variazioni dell'ambiente (luce e temperatura che cambiano nel tempo), non fenomeni nuovi: le può aggiungere l'ambiente |
| comandi dei propri muscoli (propriocezione dei muscoli) | la rete può già leggere i propri motori attraverso i collegamenti che partono da essi: sarebbe un doppione |
| danno da urto | urtare una parete oggi non fa danno; se un giorno lo farà, la pressione è già un segnale |

Lo spazio delle molecole è semplificato a un asse solo: in natura lo spazio
degli odori ha molte dimensioni. Si potrà allargare a due assi se servirà.

---

## 3. Il recettore

### 3.1 Geni

Ogni recettore è un blocco di geni:

| Gene | Forma | Significato |
|---|---|---|
| acceso | sì/no | se il recettore esiste |
| sensibilità `h_p` | 6 numeri (uno per fenomeno) | `g_p = exp(h_p)` "quanti" raccolti per unità di segnale; molto negativo = insensibile a quel fenomeno |
| sintonia `μ_p` | 6 numeri in [0, 1] | centro della campana sull'asse della qualità |
| larghezza `σ_p` | 6 numeri (logaritmo) | quanto è selettivo |
| posizione sul corpo | vettore unitario | in che punto del bordo del corpo sta (conta per sostanze e contatto) |
| direzione | vettore unitario, relativo al muso | dove guarda (luce, vibrazioni) |
| apertura | logaritmo dell'angolo | stretta = "pixel", larga = "macchia". Da 1° a 360° |
| posizione nel cervello | y in [0, 1] | dove entra il suo nervo; x = 0, sul bordo dei sensi |

Direzioni e posizioni sono **vettori**, non angoli: in 3D avranno una
componente in più e i genomi 2D si convertiranno con quella componente a zero.

### 3.2 Segnale

Per ogni fenomeno `p` il recettore raccoglie un segnale `s_p` già pesato dalla
sua sintonia. La sintonia è `T(q) = exp(-(q - μ)² / 2σ²)`, con picco 1:

- **luce**: media su M = 5 direzioni campione distribuite nell'apertura:

  ```
  s_luce = media_k ∫ T(q) R(θ_k, q) dq
  ```

  Gli integrali sul colore sono prodotti di campane e si calcolano esatti.
  Normalizzati, danno la quota della luce di un oggetto che il recettore
  raccoglie:

  ```
  albedo · σ / sqrt(σ² + w²) · exp(-(μ - c)² / 2(σ² + w²))
  ```

  dove `c` e `w` sono centro e larghezza del colore dell'oggetto. Una sintonia
  larga raccoglie tutta la luce dell'oggetto, ma anche quella degli altri
  colori.
  Le 5 direzioni campione sono un'approssimazione dichiarata: un'apertura
  larga è campionata più grossolanamente.
- **sostanze**: `s_sost = Σ_m T(q_m) · c_m(punto del recettore)`
- **meccanica**: pressione (a q = 0) più le vibrazioni, ciascuna pesata da
  `T(q_v)` e dalla finestra angolare;
- **temperatura**: `T(temperatura nel punto del recettore)`
- **magnetismo**: `max(0, B · direzione)`
- **interno**: `Σ_grandezze T(q_k) · valore_k`

### 3.3 Risposta: quanti, rumore, saturazione

Il recettore conta "quanti" (fotoni, molecole, deformazioni) e il conteggio è
rumoroso, come nella retina:

```
n  = Σ_p g_p · s_p                      quanti attesi in questo passo
ñ  = n + sqrt(n + n_buio) · ε           ε ~ N(0, 1), nuovo a ogni passo
r  = clip(ñ / n_max, 0, 1)              risposta che entra nella rete
```

- `n_buio` ≈ 1 è il rumore di fondo, che esiste anche senza segnale (nei
  bastoncelli veri è l'agitazione termica dei pigmenti).
- `n_max` ≈ 100 è la saturazione. Corrisponde alla frequenza massima di
  scarica di un neurone, limitata dal periodo refrattario: è un limite
  biofisico, non un tetto scelto da noi.

Conseguenze, tutte da natura:

- **Più sensibilità significa meno rumore relativo**: va come `1/√n`.
- **Troppa sensibilità satura** con i segnali forti, che non si distinguono
  più. È il compromesso tra bastoncelli e coni.
- **Un recettore sensibile a più fenomeni li somma**: un segnale ambiguo. Da qui
  la spinta a specializzarsi.
- **La sensibilità non è un doppione dei pesi sinaptici.** Un peso grande
  amplifica anche il rumore; solo la sensibilità lo riduce.

Il rumore usa un generatore casuale proprio del controllore, quindi la corsa
resta ripetibile con lo stesso seme.

### 3.3 bis Adattamento sensoriale (aggiunto il 2026-10-14)

Ogni recettore tiene una media recente `m` dei suoi quanti (esponenziale, con
tempo caratteristico τ) e risponde

```
r = clip((ñ − sott · m) / (n_max + div · m), 0, 1)
```

- `sott` in [0, 1]: adattamento sottrattivo, come la metilazione dei recettori
  chimici di *E. coli*: a segnale costante la risposta torna a zero, resta la
  risposta ai cambiamenti (chemiotassi).
- `div` ≥ 0: adattamento divisivo, come i fotorecettori (legge di Weber): la
  risposta segue il contrasto rispetto al livello medio e non satura.
- τ, `sott` e `div` sono geni del recettore, ereditati e mutati come gli altri.
  Alla partenza sono deboli e casuali (τ ≈ 1 s, `sott` e `div` tra 0 e 0,2).
- `m` parte dalla prima lettura di ogni arena (nessuno nasce "abbagliato").

### 3.4 Costo

```
energia/s per recettore acceso = c_cellula + c_sens · Σ_p g_p
```

- `c_cellula` è il mantenimento della cellula: un recettore inutile costa poco
  ma costa.
- `c_sens` è il prezzo della sensibilità: un recettore preciso costa.
- **Taratura di partenza** (da verificare con le misure): un "occhio" di circa
  25 recettori ben sensibili dovrebbe costare circa il 10% del metabolismo di
  base, come nella mosca (Laughlin, Niven).

---

## 4. Il cervello senza tetto (fase D)

### 4.1 Neuroni: costo invece di massimo

- **Ogni neurone interno acceso costa** `c_neurone` energia/s: un neurone vero
  consuma anche a riposo (Attwell e Laughlin, 2001). Il costo del cablaggio
  resta com'è.
- **La scorta di posti cresce da sola.** Quando un individuo usa più dell'80%
  dei posti, tutta la popolazione riceve 8 posti nuovi, spenti. È un dettaglio
  di memoria, invisibile all'evoluzione. Vale lo stesso per i recettori.
- **Costo di calcolo, da dire chiaramente.** Le reti sono matrici piene. Il
  calcolo cresce con il cervello **più grande** della popolazione, anche per
  chi ne ha uno piccolo. Se diventa un problema si passa a reti sparse: cambia
  il calcolo, non il modello.

### 4.2 Duplicazione: l'unico modo per avere un neurone o un recettore in più

Oggi un neurone si accende o si spegne. In più ci sarà la **duplicazione**,
il modo in cui crescono davvero genomi e cervelli:

1. il blocco viene copiato in un posto libero (il primo libero);
2. la posizione nel cervello si sposta di poco dall'originale;
3. si copiano anche i **collegamenti in uscita**, e i pesi in uscita di
   originale e copia vengono **dimezzati**.

Così la duplicazione è **neutra** nel comportamento: originale e copia insieme
fanno quello che faceva l'originale da solo. Nel costo invece no: la copia si
paga (neurone o recettore in più, più cablaggio). È un piccolo svantaggio che
la copia deve ripagare diventando utile, come in natura. Nelle reti continue è esatto; in quelle a impulsi
è approssimato. Una mutazione neutra non viene eliminata subito, e la copia ha
tempo di divergere: è la via naturale dell'innovazione.

Per un recettore duplicato, le due copie hanno rumore indipendente: la loro
somma è già un poco **meno rumorosa**. La duplicazione di un recettore è quindi
leggermente vantaggiosa da subito, come in natura.

### 4.3 Omologia (cosa si incrocia con cosa)

L'incrocio lavora per posizione: il posto k di un genitore con il posto k
dell'altro. Le duplicazioni scrivono nel primo posto libero, quindi in una
popolazione che discende da antenati comuni la stessa duplicazione finisce
nello stesso posto. Due duplicazioni indipendenti nello stesso posto, in
linee diverse, si mescolerebbero male. Il figlio probabilmente andrebbe peggio
e la selezione lo eliminerebbe, come gli incroci tra varianti incompatibili in
natura.

Alternativa più costosa: un'etichetta d'origine per ogni duplicazione, come i
numeri d'innovazione di NEAT. **Proposta**: partire per posizione e misurare
quanto spesso l'incrocio produce figli incompatibili.

---

## 5. Le mutazioni come geni (fase D)

Ogni individuo porta un piccolo blocco di **geni del proprio meccanismo di
mutazione**, in forma di logaritmi:

| Gene | Sostituisce la costante di oggi |
|---|---|
| probabilità di ritocco di un peso o gene, e ampiezza | `mut_weight_prob`, `mut_weight_std` |
| comparsa e scomparsa dei collegamenti | `mut_conn_add`, `mut_conn_del` |
| accensione e spegnimento di neuroni e recettori | `mut_neuron_toggle` |
| duplicazione di neuroni e recettori | nuovo |
| spostamento nel cervello, e quanto | `mut_pos_prob`, `mut_pos_std` |
| deriva dei geni dei recettori | nuovo |
| portata tipica dei collegamenti nuovi λ | `net_conn_range` |

- **Come evolvono.** Sono geni come gli altri e mutano con lo stesso
  meccanismo: con la probabilità e l'ampiezza di ritocco dell'individuo stesso,
  in scala logaritmica. In natura i tassi dipendono dalle proteine che copiano e
  riparano il DNA, i cui geni mutano come tutti gli altri (geni mutatori,
  ipotesi della barriera della deriva, Lynch). Una nascita usa i tassi
  ereditati.
  *Correzione del 2026-10-14*: la prima versione ritoccava ogni tasso a ogni
  nascita di un fattore e^(0,2·N(0,1)) (`meta_tau`). Era troppo veloce: in
  centinaia di generazioni i tassi derivavano di oltre 100 volte per puro
  caso, la duplicazione salì al 2,5% per neurone e in 2 corse su 8 il cervello
  esplose (220 neuroni, costo 17 volte il metabolismo di base). `meta_tau` è
  stato tolto.
- **Nell'incrocio** il blocco si eredita intero da un genitore.
- **Un minimo fisico.** Nessun tasso scende sotto un errore minimo di
  copiatura (1e-5): la copiatura perfetta non esiste.
- **Cosa resta fissato da noi, onestamente.** I valori di partenza, il
  minimo fisico di copiatura, il soccorso
  quando i genitori sono 0 o 1, la dimensione della popolazione e la regola di
  selezione.

---

## 6. Il cervello: dove entrano i sensi

- I recettori entrano dal bordo sinistro (x = 0) alla y del loro gene. Il gene
  deriva, quindi recettori simili possono raggrupparsi: le mappe
  topografiche emergono da sole (sezione 4.2).
- I motori restano fissi sul bordo destro, due per le reti continue e quattro
  per quelle a impulsi.
- Geometria, probabilità di collegamento che cala con la distanza e costo del
  cablaggio restano come sono.

---

## 7. La partenza primordiale

| | Valore iniziale |
|---|---|
| recettori | 2, ai lati del muso (±30°), apertura 120° |
| sensibilità | uguale su tutti e 6 i fenomeni, bassa (g ≈ 5): risponde a tutto, con molto rumore |
| sintonia | μ = 0,5 e σ larga (0,4) su ogni fenomeno |
| neuroni interni | 0 accesi (scorta iniziale: 8 posti spenti) |
| collegamenti | dai recettori ai motori, con la probabilità geometrica di oggi; pesi casuali |
| geni delle mutazioni | i valori attuali di `config.py`, con un po' di variazione tra individui |

Tutti gli individui partono con questa forma, con geni leggermente diversi tra
loro.

**Ambiente iniziale consigliato** (la scelta resta tua): niente veleno,
bersaglio di raggio 6, energia 150. Misurato oggi con agenti ciechi: completa
il 2,1%, cioè circa 4 genitori per generazione, senza estinzioni.

---

## 8. Cosa cambia nel resto del programma

- **Configurazione.**
  - Spariscono `vision_rays`, `vision_fov_deg`, `vision_range`,
    `net_hidden_max` e le costanti `mut_*`, che diventano geni o valori
    iniziali.
  - Entrano le costanti della fisica dei segnali (`L_luce`, spettri degli
    oggetti, `n_buio`, `n_max`) e i costi (`c_cellula`, `c_sens`,
    `c_neurone`).
  - Ogni costante nuova va classificata per l'esame: il test lo impone.
- **Salvataggi.** Formato 3. I salvataggi precedenti contengono agenti con i
  sensi fissi: per l'evoluzione non si aprono più, come già deciso per la
  geometria.
- **Esame.** Le prove restano le stesse: misurano il comportamento, non la
  struttura. Ma la vista cambia natura (da telemetro a mezzo torbido), quindi
  l'esame passa alla **versione 2** e i punteggi non si confrontano con la
  versione 1.
- **Interfaccia.**
  - Ogni recettore è descritto con parole ricavate dai suoi geni, per esempio
    "luce 0,85 stretta, avanti-sinistra, apertura 15°" oppure "generalista".
  - Nell'arena si disegnano le aperture dei recettori al posto dei raggi.
  - Nuove statistiche: recettori per individuo, **quota di recettori
    specializzati** (un fenomeno che domina gli altri), costo dei sensi, valori
    medi dei geni delle mutazioni.
- **Reti a impulsi.** I recettori entrano come corrente, come i sensi di oggi.

---

## 9. In 3D

Il modello resta lo stesso: fenomeni, sintonia, quanti e rumore, costi,
duplicazioni, geni delle mutazioni. Cambia soltanto la geometria, che starà in
un modulo solo:

- posizioni sul corpo su una sfera, direzioni 3D, aperture a cono (le
  direzioni campione si distribuiscono nel cono);
- leggi di attenuazione: la luce resta `exp(-d/L)`; sostanze e vibrazioni in
  3D calano più in fretta con la distanza;
- il cervello può restare un quadrato o diventare un cubo: è una scelta
  indipendente.

---

## 10. Come si verifica

1. **Test della fisica**: ogni formula su casi calcolabili a mano. Per esempio
   una nube attraversata per intero, il contatto da un lato solo, la sintonia
   esattamente sul colore del bersaglio.
2. **Test del rumore**: media e varianza della risposta su molti passi devono
   seguire `n` e `n + n_buio`; la saturazione deve intervenire a `n_max`.
3. **Test della duplicazione neutra**: una rete continua con un neurone
   duplicato deve dare le stesse uscite di prima.
4. **Ripetibilità**: stesso seme, stessa corsa, anche con il rumore.
5. **Calibrazione**, prima di decidere i numeri:
   - il costo dei sensi di un "occhio" deve essere circa il 10% del
     metabolismo di base;
   - la velocità di calcolo con la partenza primordiale non deve essere molto
     peggiore di oggi (obiettivo: meno del doppio del tempo per generazione).
6. **Esperimento di validazione**: 4 semi, partenza primordiale, ambiente
   facile, alcune migliaia di generazioni. Si misura:
   - se compaiono recettori specializzati, cioè la quota di recettori
     specializzati nel tempo;
   - se i recettori sensibili alla luce si moltiplicano e si stringono, verso
     un "occhio";
   - come evolvono i tassi di mutazione;
   - l'esame (versione 2) nel tempo.

   Il risultato possibile, e da dichiarare, è anche un **fallimento**: per
   esempio l'evoluzione si ferma alla strategia "cieco che cerca bene". In quel
   caso il rimedio previsto è rendere l'ambiente più difficile a mano.

### Ordine di lavoro

1. Fisica dei segnali e risposta dei recettori (con i test 1–2).
2. Nuovo genoma: blocchi dei recettori, scorte che crescono, duplicazioni,
   geni delle mutazioni (test 3–4).
3. Partenza primordiale, costi e calibrazione (punto 5).
4. Interfaccia ed esame versione 2.
5. Esperimento di validazione (punto 6) e resoconto in `docs/NOTE.md`.

---

## 11. Scelte ancora aperte

1. **Corpo e motori.** Restano fissi: due motori (spinta e sterzo), corpo di
   raggio 1. Far evolvere anche il corpo (dimensione, forza dei motori, altri
   attuatori) è possibile con lo stesso schema (geni più costi). Proposta: in
   una fase successiva, per non cambiare troppe cose insieme.
2. **Omologia**: per posizione (proposta) o con etichette d'origine (sezione 4.3).
3. **Odore del veleno fuori dalla zona** (`perdita`): 0 di default. È un
   cambio d'ambiente, quindi lo decidi tu, ora o più avanti.
4. **Numeri di partenza**: spettri, L_luce, costi, sensibilità iniziale. Li
   propongo dopo la calibrazione, con le misure.
