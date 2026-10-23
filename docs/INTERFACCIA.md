# L'interfaccia

Il manuale completo dell'interfaccia grafica (`uv run main.py`). Le regole della simulazione
sono in [REGOLE.md](REGOLE.md).

Una finestra sola con quattro schermate: menu, corsa, esploratore delle reti e
banco di prova. **Una corsa aperta continua in sottofondo** anche mentre si
guardano l'esploratore o il banco di prova; lì `B` la mette in pausa o la fa
ripartire, e una riga in basso dice a che generazione è arrivata.

## Il menu

Si sceglie la corsa e la si lancia con Invio (o il pulsante).

| Riga | Significato |
|---|---|
| Modalità | **veloce**: si parte dalla pagina principale (statistiche e reti), l'arena si apre con `F`; **massima velocità**: solo statistiche, ridisegnate due volte al secondo, tutto il resto del tempo all'evoluzione; **dal vivo**: si parte con l'arena a schermo. Il risultato è lo stesso in tutte e tre |
| Rete | continua o a impulsi con neuroni eccitatori/inibitori |
| Simulazione | **completa** (evolvono sensi e cervello) o **solo cervello** (sensi fissi decisi da noi, vedi [REGOLE.md](REGOLE.md), punto 0) |
| Individui | dimensione della popolazione |
| Mondo | **bersaglio e veleno** (il mondo di prima) o **nutrirsi**: cibo, prede e predatori, si riproduce chi mangia almeno N cibi e il veleno è spento ([SPEC_MONDO.md](SPEC_MONDO.md)) |
| Scala | solo nel mondo del cibo: **adattiva** (le manopole della difficoltà salgono se va bene e scendono se va male) oppure ferma al mondo **basico** o al **complesso**, come le popolazioni di controllo. Le tre insieme nello stesso lotto si lanciano dal terminale con `--controls` |
| Casualità | **seme casuale** (uno nuovo a ogni corsa) o **seme fisso** (quello della riga Seme). In entrambi i casi il seme usato compare nel menu, in cima alla corsa e nel salvataggio |
| Seme | il seme fisso; con il seme casuale mostra quello appena estratto |
| Generazioni | dopo quante fermarsi e salvare; 0 = senza limite |
| Esame ogni | ogni quante generazioni fare l'esame fisso ai 20 più anziani ([REGOLE.md](REGOLE.md), punto 20); 0 = mai. Default 50 |
| Selezione finale | finite le generazioni, quante prove fanno ancora i sopravvissuti, senza figli ([REGOLE.md](REGOLE.md), punto 19); 0 = niente. Si attiva solo con Generazioni > 0 |
| Riprendi da | un salvataggio in `runs/` da cui continuare; rete, individui e tutte le costanti vengono da lì. Invio o clic aprono l'elenco: i salvataggi sono raggruppati per giorno (Oggi, Ieri, poi le date), dal più recente, ognuno con l'ora e un riepilogo (generazione, individui, genoma, rete, sensi, mondo e posizione della scala, ultimo esame). Scrivendo si filtra su nome e riepilogo (per esempio "sviluppo impulsi"); Su/Giù, Pag e rotella scorrono, Invio o clic sceglie, Esc chiude senza cambiare |
| Salva in | dove salvare; per una corsa nuova propone un nome con data e ora, così non sovrascrive nulla |

Con un salvataggio scelto, "Esplora reti e genomi" lo apre nell'esploratore
senza far ripartire l'evoluzione; da lì si possono anche provare le sue reti.

Su/Giù o Tab scelgono la riga, Sinistra/Destra (o il clic) cambiano il valore,
numeri e percorso si scrivono da tastiera. Nei numeri le cifre si aggiungono in
coda: per scriverne uno nuovo si cancella prima con Backspace.

Con **Mondo = arena continua** (docs/SPEC_ARENA.md) compare la riga
**Tetto** (quanti esseri vivi al massimo); **Individui** è quanti ce ne sono
all'inizio, **Generazioni** diventa **Minuti di mondo** ed **Esame ogni** si
conta in minuti (10 di partenza). La scala e la selezione finale non ci sono.

## L'arena continua

Una sola schermata. A sinistra la **vista che segue un essere** (giallo,
con la scia; gli altri esseri sono verdi, il cibo rosso, le macchie fertili
ombre verdi più chiare quanto più sono fertili); a destra la **mappa intera**
con il riquadro della vista, poi le misure (vivi, nati, colonizzatori, morti,
ultimo minuto, ultimo esame) e i dati dell'essere seguito. Se l'essere
seguito muore si passa al più anziano.

| Tasto | Azione |
|---|---|
| `Spazio` | pausa |
| `1` `2` `3` | velocità: tempo reale, x4, x16 |
| `Z` | turbo: quanto più veloce si può |
| rotella, `+` `-` | zoom |
| `Tab` / `O` / clic | segui il prossimo (dal più anziano) / il più anziano / quello cliccato |
| `E` | esame fisso adesso, su 20 adulti a caso |
| `S` | salva |
| `Esc` | torna al menu (salva se ci sono novità) |

L'esploratore delle reti non c'è ancora per l'arena.

## La corsa: la pagina principale

In alto la generazione, la rete, la popolazione e lo stato della generazione
in corso. La pagina ha **due schede**, che si cambiano con `V` o con un clic in
alto a destra.

### Panoramica

È la scheda di partenza: cosa sta succedendo, scritto a parole. Ogni sezione
ha un titolo e una frase che dice come leggere i numeri.

| Sezione | Cosa dice |
|---|---|
| **A che punto è la scala** (mondo del cibo) | una barra con i tre stadi: 1. trovare il cibo, 2. catturare le prede, 3. sfuggire ai predatori. La parte piena è il percorso fatto, il triangolo la posizione di adesso, la barretta azzurra sotto il tratto percorso nelle ultime 100 generazioni. Sotto, lo stadio a parole, il mondo di questa generazione (cibi in arena, quanti ne servono, prede, predatori) e un giudizio sull'andamento: sale, scende, si muove o è ferma |
| **Il compito** (mondo del bersaglio) | il compito e il registro dell'ambiente |
| **Quanti si riproducono** | le ultime 300 generazioni: in grigio ogni generazione, in azzurro la media di 10. Nella scala adattiva la fascia verde (30–50%) è la zona in cui la scala resta ferma: sopra sale, sotto scende |
| **La scala nel tempo** (mondo del cibo) | la posizione da 0 a 3 in tutta la corsa, con il nome di ogni stadio |
| **Esame: cosa sanno fare** | l'ultimo esame: una barra per livello con la percentuale scritta sopra e, sotto, cosa c'è nel livello ("3 su 8" = 3 cibi da mangiare su 8 in arena, "fuga 9" = prede che scappano a 9 u/s, "2 pred." = due predatori). La tacca bianca è il primo esame, il punto viola l'ultimo esame con lesione. Sotto, le differenze rispetto al primo esame e al precedente (sotto i 5 punti si dice "stabile") e i livelli mai superati |
| **Il cervello medio e la popolazione** | neuroni interni (e quanti erano all'inizio), collegamenti, recettori usati per fenomeno, quota di inibitori, la fortuna (quanti completano tra chi nasce vicino al cibo e chi lontano) e il più anziano, con il link alla sua rete |

### Reti

A sinistra tre elenchi (`1`, `2`, `3` o `Tab`; da panoramica portano qui):

- **campioni**: uno per ogni generazione già conclusa con almeno un vincitore.
  Il campione **non è scelto a caso**: è il vincitore più anziano di quella
  generazione (quello che ha vinto più prove di fila) e, a parità di età, il
  nato prima (numero più basso). Accanto, la percentuale di successi di quella
  generazione. Si aggiorna da solo: se è scelta la riga più in alto, si segue
  l'ultimo campione arrivato;
- **sopravvissuti**: i vincitori dell'ultima generazione conclusa, che ora
  corrono di nuovo (dopo una selezione finale: i rimasti);
- **tutti**: tutta la popolazione in gara, sopravvissuti più figli appena nati
  (età 0, non ancora messi alla prova).

Sopravvissuti e tutti sono una fotografia: `R` la rifà. La rete scelta è
disegnata accanto (vedi "Come si legge una rete").

In cima alla pagina una riga dice sempre quanti sono i sopravvissuti e quanti i
figli nuovi della generazione in corso.

Sopra ogni elenco un'intestazione dice cosa c'è in ogni colonna:

| Colonna | Significato |
|---|---|
| gen | generazione di cui è campione |
| succ | percentuale della popolazione che ha completato il compito in quella generazione |
| num. | numero dell'individuo ([REGOLE.md](REGOLE.md), punto 18) |
| età | prove vinte di fila |
| neur | neuroni interni esistenti |
| colleg | collegamenti effettivi |

Al centro la rete scelta. **A destra, il neurone toccato**: chi è (senso, interno eccitatorio o
inibitorio, motore), i suoi geni propri e **tutti** i suoi collegamenti, in
ingresso ("riceve da") e in uscita ("manda a"), ciascuno con il peso e se eccita
o inibisce, dal più forte. Se sono tanti si scorrono con la rotella.

| Tasto | Azione |
|---|---|
| Su/Giù, clic | scegli la rete nell'elenco |
| clic su un neurone | lo fissa nel pannello di destra (un altro clic lo libera); passarci sopra basta per vederlo |
| `T` | **prova da sola la rete scelta** (banco di prova) |
| `A` | **mette in arena tutti i sopravvissuti insieme** (banco di prova) |
| `Tab` / `1` / `2` / `3` | campioni / sopravvissuti / tutti |
| `Invio` | a esperimento concluso, riapre il riepilogo |
| `O`, o clic sul più anziano | mostra la rete del più anziano nella scheda Reti (nell'arena: lo segue) |
| `P` | rami morti in grigio o nascosti |
| `I` | esploratore a tutto schermo, con il genoma |
| `F` | apre l'arena (la corsa continua) |
| `E` | **esame fisso adesso** dei 20 più anziani (gira su altri core; va avanti anche in pausa) |
| `L` | **esame con lesione**: si sceglie con un numero cosa togliere (i neuroni interni, la plasticità o un senso) e i 20 più anziani rifanno l'esame così; gli originali non cambiano |
| `V` | panoramica / reti |
| `Z` | massima velocità sì/no: mostra la panoramica, ridisegna due volte al secondo e dà il resto del tempo all'evoluzione |
| `Spazio` | pausa |
| `S` | salva la popolazione |
| `Esc` | torna al menu, salvando se ci sono generazioni nuove (lo stesso alla chiusura della finestra) |

La pagina principale lascia all'interfaccia una parte del tempo per restare
reattiva; con `Z` (o la modalità "massima velocità" nel menu) la corsa va più
veloce, con lo stesso risultato.

## La corsa: l'arena

`F` apre l'arena in turbo senza fermare l'evoluzione; `F` di nuovo torna alla
pagina principale. A sinistra gli agenti, a destra la rete dell'agente seguito
e, sotto, i successi delle ultime 300 generazioni. In alto a sinistra la
generazione, lo stadio della scala e il mondo di adesso (mondo del cibo),
quanti hanno completato e l'agente seguito (all'apertura il più anziano), con
i cibi mangiati su quelli necessari e la distanza dal predatore più vicino.
L'elenco dei tasti si apre con `H`. **Ogni agente ha la sua copia del mondo**: cibo e predatori disegnati
sono quelli dell'agente seguito, e cambiano quando se ne segue un altro.

| Nell'arena | Significato |
|---|---|
| Cerchi verdi sfumati | veleno: drena energia, poco al bordo e molto al centro |
| Cerchio azzurro | bersaglio |
| Dischi rossi | cibo (nel mondo del cibo). Con le prede accese, un anello sottile mostra fin dove la preda si accorge dell'agente e fugge |
| Alone rosso tenue | la nube d'odore di ogni cibo, fino alla sua portata; più tenue mentre si sta formando |
| Cerchietto rosso vuoto con alone | la traccia d'odore di un cibo appena mangiato, che svanisce |
| Dischi blu | predatori, con l'area in cui si accorgono dell'agente e lo inseguono |
| Scia gialla | il percorso recente dell'agente seguito (si azzera cambiando agente o generazione) |
| Agente giallo | quello che stai seguendo, con i suoi recettori: per ognuno due linee grigie ai bordi del suo campo e una linea al centro, più lunga e più accesa quanto più risponde; il colore dice a cosa è sensibile (vedi sotto) |
| Anello bianco | sopravvissuto di una generazione precedente |
| Bianco pieno | ha completato il compito |
| Grigio | morto |

| Nel pannello della rete | Significato |
|---|---|
| Colonna sinistra | i recettori accesi, all'altezza in cui il loro nervo entra nel cervello; il bordo dice a cosa sono sensibili: giallo luce, verde sostanze, arancio meccanica, rosso temperatura, viola magnetismo, grigio interno, bianco generalista |
| Centro | i neuroni interni di quell'individuo, dove stanno nel cervello |
| Destra | i motori |
| Pallino arancione / azzurro | neurone attivo in positivo / negativo |
| Linea arancione / azzurra | collegamento che eccita / inibisce; si illumina quando passa segnale |

| Tasto | Azione |
|---|---|
| `+` / `-` | velocità da x1 a x16 |
| `T` | turbo |
| clic | segui un altro agente |
| `M` | guida manuale dell'agente seguito (WASD o frecce) |
| `V` | recettori disegnati sì/no |
| `C` | elenco delle costanti; `Tab` sceglie, `,` `.` regolano |
| `O` | segui il più anziano: il pannello a destra mostra la sua rete |
| `H` | elenco dei tasti sì/no |
| `F`, `I`, `Spazio`, `S`, `Esc` | come nella pagina principale |

## Come si legge una rete

Vale per la pagina principale, per l'esploratore e per il pannello della rete
nell'arena. Ogni neurone è disegnato **dove sta davvero nel cervello** (vedi il
[REGOLE.md](REGOLE.md), punto 8): recettori sul bordo sinistro, motori sul destro, neuroni interni nella
loro posizione dentro il riquadro, che è il cervello. Un collegamento lungo sul
disegno è lungo anche nel cervello, e costa energia in proporzione. Un
cerchietto sopra un neurone è un collegamento con se stesso.

Colore = segno (arancio eccita, azzurro inibisce), intensità e spessore =
forza. Nelle reti a impulsi il bordo di un neurone interno dice il suo tipo. In
grigio i **rami morti**: neuroni e collegamenti che non arrivano a nessun
motore e quindi non cambiano il comportamento. Passando su un neurone (o
cliccandolo) si accendono solo i suoi collegamenti, **in entrata e in uscita**,
con una **freccia a metà** che ne indica il verso. Il colore è sempre l'effetto
di chi trasmette: arancio se eccita, azzurro se inibisce. Nelle reti a impulsi
lo decide il tipo del neurone di partenza, quindi le uscite di un neurone
inibitorio sono tutte azzurre e in entrata si vede chi lo eccita e chi lo
inibisce. Anche i collegamenti dei rami morti prendono il colore, solo più
tenue. Il pannello dei dettagli dice la posizione del neurone, a
quanti passi è dai recettori e, per ogni collegamento, peso, effetto e
lunghezza. Per un recettore mostra tutti i suoi geni: dove sta sul corpo, dove
guarda, l'apertura e, per ogni fenomeno, la sensibilità g, la quota sul totale
e la sintonia (centro ± larghezza). In cima alla rete ci sono il cablaggio e
quanta energia costano in tutto corpo e cervello al secondo.

**Con lo sviluppo** (genoma "sviluppo") il colore di ogni cellula dice la sua
**identità**: la miscela dei colori dei marcatori che esprime, più luminosa
quanto più li esprime. Ogni marcatore ha il colore della sua famiglia, quindi
un marcatore e le sue copie hanno lo stesso colore, in tutti gli individui
della stessa discendenza. Due neuroni dello stesso colore si legano agli
stessi partner: sono dello stesso **tipo**. Nell'esploratore è il colore
interno dei neuroni (e un pallino dentro i motori e i recettori); nel pannello
dell'arena un anello attorno ai neuroni interni. In cima alla rete si leggono
quanti marcatori ha l'individuo e quanti tipi di neuroni interni; nei
dettagli di un neurone, quanto esprime ogni marcatore.

## L'esploratore delle reti

`I` apre a tutto schermo lo stesso elenco e la stessa rete della pagina
principale, più il **genoma** come matrice: una riga per blocco (neurone
interno o motore), una colonna per sorgente (recettori | interni | motori). Colore =
peso effettivo, grigio = gene latente (collegamento scritto nel genoma ma con un
capo spento), righe scure = neuroni spenti. Passando col mouse su una cella se
ne legge il valore; il clic su una riga sceglie quel neurone.

Con lo sviluppo, sotto la matrice c'è la **tabella di affinità**: quanto ogni
marcatore sull'assone (righe) si lega a ogni marcatore sul bersaglio
(colonne), arancio se il collegamento nasce eccitatorio, azzurro se
inibitorio, più intenso se più forte; ai bordi il colore di ogni marcatore e,
a destra, i suoi gradienti lungo x e y (come Eph ed efrine). Con lo sviluppo la
matrice sopra è il cervello **sviluppato** (dopo il grembo), non un gene.

`T` prova la rete scelta, `X` scrive il genoma completo (blocco per blocco, con i
geni latenti) in un file di testo in `runs/genomi/`, `Esc` torna indietro. Il
campione di ogni generazione finisce nel salvataggio, quindi tutto questo si
può fare anche dopo, aprendo il salvataggio dal menu.

## Fine dell'esperimento

Quando le generazioni richieste sono finite (e l'eventuale selezione finale è
conclusa) la corsa si ferma e sulla pagina principale compare un
**riepilogo**: quante generazioni, i successi dell'ultima, **quanti
sopravvissuti ci sono e chi sono** (i loro numeri), il più anziano, l'esito di
ogni prova della selezione, dove sono stati salvati. Sotto, i pulsanti per le
cose da fare dopo:

| Pulsante | Cosa fa |
|---|---|
| Guarda i sopravvissuti in arena (`A`) | li mette tutti insieme nel banco di prova, per vedere come si comportano |
| Esplora le loro reti (`2`) | chiude il riepilogo e mostra l'elenco dei sopravvissuti con le loro reti |
| Continua l'evoluzione (`Spazio`) | riprende da dove si era fermata, senza limite |
| Chiudi (`Invio`) | chiude il riepilogo; `Invio` lo riapre |

## Il banco di prova

Una rete sola, o un gruppo, nell'arena fuori dall'evoluzione: nessuno nasce e
nessuno viene eliminato, si osserva soltanto.

- **una rete**: un campione o un singolo individuo, scelto con `T` dalla pagina
  principale o dall'esploratore;
- **un gruppo**: tutti i sopravvissuti insieme, con `A` (o dal riepilogo di
  fine esperimento). Si segue un agente col clic o con `Tab`; la sua rete è dal
  vivo nel pannello di destra. In basso a destra una riga per rete: quante
  arene viste ha completato e, dopo `E`, la sua percentuale nella
  valutazione, le migliori in alto.

- Le arene sono **sempre le stesse** (arena 0, 1, 2, ..., con semi propri,
  diversi da quelli dell'evoluzione): reti diverse si confrontano sulle stesse
  prove. A fine arena si passa da sola alla successiva; in basso a destra un
  quadratino per arena vista (azzurro = completato).
- `E` lancia una **valutazione**: 20 arene fisse, con 50 copie della rete per
  arena (10 copie a testa per un gruppo) che partono da punti diversi. Dà la
  percentuale di prove completate (per ogni rete, in un gruppo), il tempo medio
  per arrivare e, per una rete sola, una barra per arena. Va avanti a pezzi,
  senza bloccare la schermata.

| Tasto | Azione |
|---|---|
| `Spazio` | pausa |
| `+` / `-` | velocità da x1 a x64 |
| `N` / `R` | arena successiva / rifai questa |
| `A` | passa da sola alla successiva sì/no |
| `E` | valutazione su 20 arene (50 copie, 10 a testa per un gruppo) |
| clic, `Tab` | segui un altro agente del gruppo |
| `M` | guida manuale (WASD o frecce) |
| `V` | recettori disegnati sì/no |
| `B` | pausa/ripresa della corsa in sottofondo |
| `Esc` | torna indietro |
