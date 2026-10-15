# Specifica: un mondo nuovo, dal basico al complesso

Stato: **approvata** (2026-10-15). Fatti i punti 1 e 2 dell'ordine di lavoro
(mondo nel motore NumPy e nel motore GPU, con i test di equivalenza); il
compito predefinito resta `bersaglio` finché non ci sono scala ed esame.

Scelte dell'utente:

- **compito**: si riproduce chi, nella sua vita, mangia almeno N cibi buoni;
- **progressione adattiva** lungo una scala di difficoltà, senza mai far
  estinguere la popolazione;
- **in cima alla scala**: prede che scappano e predatori;
- **due popolazioni di controllo**: una sempre nel mondo complesso, una sempre
  nel mondo basico.

Motivo: nel mondo di oggi il compito è toccare il bersaglio una volta, quindi
chi nasce vicino vince per fortuna e "sporca" la selezione. E il mondo si
risolve con un riflesso (un occhio che gira, esame 46%): un cervello più grande
non serve, quindi non nasce.

---

## 1. Il compito: nutrirsi

- Nell'arena c'è **cibo**: piccoli dischi di un colore (rosso, 0,85 sull'asse
  della luce) che, toccati, vengono **mangiati**. Danno energia
  (`food_energy`, proposta: 40, così nel mondo basico nutrirsi conviene
  chiaramente) e contano per il compito.
- Il cibo mangiato **ricompare** in un punto a caso dell'arena, lontano
  dall'agente che l'ha mangiato.
- **Si riproduce chi mangia almeno N cibi** nella vita (N lungo la scala, da 1
  a 5). Arrivato a N l'agente è in salvo, come oggi chi tocca il bersaglio.
- Gli agenti nascono ad almeno 20 unità da qualsiasi cibo (`food_min_dist`) e
  40 dai predatori.
- La vita ha un tetto (`life_max`, proposta: 120 s), perché mangiando l'energia
  può durare a lungo.

**Perché toglie la fortuna:** una volta si può essere fortunati, cinque volte
di fila no. In natura il successo riproduttivo dipende da quanto ci si nutre in
tutta la vita, non da un incontro fortunato.

**Ognuno vive la propria prova**, come oggi. Ogni agente ha la **sua copia** di
cibo, prede e predatori, che partono uguali per tutti all'inizio della
generazione. Se un agente mangia un cibo, sparisce solo nella sua copia. Niente
competizione tra agenti: è un'altra scelta d'ambiente, che si potrà fare più
avanti.

## 2. Le manopole della difficoltà

Ogni manopola va da 0 (basico) a 1 (complesso). I valori indicati sono
proposte, da tarare con le misure.

| Manopola | 0 (basico) | 1 (complesso) | Cosa richiede |
|---|---|---|---|
| **abbondanza** | 12 cibi, raggio 4 | 3 cibi, raggio 2 | cercare in modo efficiente |
| **cibi necessari** (N) | 1 | 5 | affidabilità |
| **prede** | cibo fermo | il cibo si accorge dell'agente entro 15 u e fugge a 12 u/s, fermandosi contro le pareti (la deriva lenta quando l'agente è lontano è rimandata alla taratura) | inseguire, prevedere, aggirare |
| **predatori** | nessuno | 3 predatori che inseguono l'agente entro 40 u a 15 u/s (lui arriva a 30 a spinta piena, ma consuma molto); a contatto tolgono 60 energia al secondo | accorgersi in tempo, fuggire, scegliere tra mangiare e scappare |

**Percepire prede e predatori** non richiede sensi nuovi:

- il **cibo** è rosso (0,85) per la vista;
- i **predatori** hanno un loro colore (proposta: 0,05, quasi blu) e
  **vibrano** muovendosi (frequenza 0,4): si possono vedere e anche sentire,
  utile nell'acqua torbida;
- il veleno resta **spento** in tutta la scala (non scelto per la cima); le
  costanti ci sono, se un giorno vorrai riaccenderlo.

## 3. La progressione adattiva

- Le manopole salgono **una alla volta**, in quest'ordine: abbondanza e cibi
  necessari (insieme), poi prede, poi predatori.
- A ogni generazione si guarda la quota di chi completa il compito, mediata
  sulle ultime 10 generazioni:
  - **sopra il 30%** la manopola corrente sale di 0,02, cioè almeno 50
    generazioni per percorrerla tutta;
  - **sotto il 10%** scende di 0,02;
  - tra i due resta ferma;
  - se in una generazione **nessuno** completa scende di 0,05, e la
    generazione dopo riparte dagli ultimi genitori (come oggi): non ci si
    estingue mai.
- Le soglie sono una nostra scelta: servono solo a non far estinguere la
  popolazione e a non lasciarla in un mondo troppo facile.
- Ogni movimento delle manopole finisce nel **registro dell'ambiente** con la
  sua generazione. L'interfaccia mostra le manopole e il loro andamento.

## 4. Come si misura l'intelligenza

1. **Esame a livelli** (esame versione 4). 8 livelli fissi lungo la scala (da
   tutte le manopole a 0 a tutte a 1), con arene e semi fissi. Ogni esame dà un
   **profilo**: quanto la popolazione riesce a ogni livello. Diventare più
   intelligenti vuol dire migliorare sui livelli difficili **senza perdere**
   quelli facili.
2. **Due popolazioni di controllo**, nello stesso lotto sulla GPU, con lo stesso
   seme di partenza:
   - **sempre complesso**: tutte le manopole a 1 dall'inizio (con la sola regola
     anti-estinzione). Dice se la scala serve davvero;
   - **sempre basico**: tutte a 0. Dice quanto conta il tempo da solo.
3. **Prove di lesione** su richiesta (riga di comando e interfaccia): togliere
   recettori, neuroni interni o la plasticità e rimisurare il profilo. Dicono
   cosa gli agenti usano davvero.
4. **Fortuna sotto controllo**: per ogni generazione si registra se completare
   dipende dalla distanza iniziale dal cibo. Se la dipendenza è forte, la
   fortuna sta ancora pesando.

## 5. Cosa cambia nel codice

- **Il mondo** guadagna oggetti per agente: cibo (posizione, attivo o mangiato,
  velocità), predatori (posizione, velocità), contatori dei cibi mangiati. La
  fisica di prede e predatori è semplice: fuga o inseguimento in linea retta,
  rimbalzo sulle pareti, velocità massima.
- **I sensi**: la luce di cibo e predatori (dischi colorati, uno per agente: la
  stessa geometria del bersaglio di oggi), le vibrazioni dei predatori. Nessun
  gene nuovo nei recettori.
- **Il motore GPU e quello NumPy**: entrambi, con i test di equivalenza come per
  la migrazione.
- **L'evoluzione**: la progressione adattiva, e più popolazioni (scala +
  controlli) nello stesso lotto.
- **L'esame versione 4**, a livelli.
- **L'interfaccia**: cibo, prede e predatori nell'arena; le manopole e il
  profilo dell'esame nella pagina principale; la scia del percorso
  dell'agente seguito.
- Il mondo di oggi (bersaglio e veleno) **resta disponibile** come compito
  separato, per riaprire le corse vecchie.

## 6. Ordine di lavoro

1. Mondo nuovo nel motore NumPy (cibo, prede, predatori, compito), con i test.
2. Lo stesso nel motore GPU, con i test di equivalenza.
3. Scala e progressione adattiva; più popolazioni nello stesso lotto.
4. Esame a livelli, prove di lesione, misura della fortuna.
5. Interfaccia.
6. Taratura: con una popolazione primordiale nel mondo basico, deve completare
   qualcuno ma non troppi (proposta: tra il 5% e il 30%), poi si regolano i
   valori della tabella.
7. Primo esperimento: scala adattiva più i due controlli, qualche migliaio di
   generazioni, 4 semi.

## 7. Limiti noti

- **Ognuno ha la sua copia del mondo**: niente competizione per il cibo né
  vita di gruppo. È una semplificazione voluta, per ora.
- **Prede e predatori sono "stupidi"**: regole fisse, non evolvono. La
  coevoluzione (predatori che evolvono anch'essi) è possibile più avanti.
- **Le soglie della progressione** sono strumenti di addestramento, non natura.
