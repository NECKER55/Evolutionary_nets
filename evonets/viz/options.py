"""Che cosa significa ogni opzione del menu. `HELP`: che cos'è la riga; `DETAILS`:
che cosa vuol dire il valore scelto. Il menu li mostra a lato della riga."""

SECTIONS_BRAIN = "A sensi, B nucleo stabile, C memoria veloce, D memoria di lavoro, E scelta dell'azione, L memoria lunga, " \
                 "U incertezza, V critico, F motore."

HELP = {
    "mode": "Quanto si vede mentre gira. Il risultato della corsa è lo stesso: cambia solo quanto in fretta arriva.",
    "sim": "Che cosa può cambiare tra una generazione e l'altra.",
    "genome": "Come il DNA descrive il cervello.",
    "net": "Il tipo di neurone. In entrambi i casi i neuroni interni si collegano in ogni direzione, anelli compresi. "
           "All'inizio sono spenti: ne nascono di nuovi per duplicazione, e ognuno costa energia.",
    "agents": "Quanti esseri nella popolazione. Più ce ne sono, più varietà e più lenta ogni generazione.",
    "task": "In quale mondo vivono gli esseri. Da qui dipendono le pagine e le opzioni che seguono.",
    "ladder": "Solo nel mondo del cibo: come si muove la difficoltà.",
    "cap": "Solo nell'arena continua: quanti esseri vivi al massimo. La popolazione vera la decide il cibo; "
           "il tetto evita che esploda. 'Individui' è quanti si parte.",
    "chance": "Da dove viene il caso della corsa. Il seme usato viene sempre salvato, quindi ogni corsa si può rifare.",
    "seed": "Stesso seme e stesse impostazioni danno la stessa corsa. Attivo solo con il seme fisso.",
    "gens": "Dopo quante generazioni, o minuti di mondo, fermarsi e salvare. 0 significa senza limite: si ferma quando esci.",
    "exam_every": "Ogni quanto i 20 più anziani fanno l'esame fisso (7 prove, 8 livelli nel mondo del cibo). "
                  "Serve solo a misurare, non cambia la corsa. 0 significa mai.",
    "select": "Finite le generazioni, i sopravvissuti rifanno la prova questo numero di volte, senza figli: "
              "chi fallisce è eliminato. 0 significa niente.",
    "p_agent": "Chi guida gli esseri nell'arena delle pile.",
    "p_phase": "Che cosa succede quando un essere muore.",
    "p_lesion": "Toglie una sezione al cervello a sezioni, per vedere a che cosa serve confrontando i risultati.",
    "p_good": "Che cosa distingue le pile buone dalle cattive. I ruoli si scambiano ogni tanto: l'essere deve accorgersene.",
    "p_teacher": "Solo per il cervello a sezioni: il primo essere di ogni recinto è guidato a mano, e gli altri imparano "
                 "il valore delle cose anche guardando ciò che gli succede. M lo toglie o lo rimette.",
    "p_encl": "Quanti recinti girano insieme. Ogni recinto è una copia indipendente, utile per fare medie; se ne guarda uno alla volta.",
    "p_beings": "Quanti esseri in ogni recinto. Competono per le stesse pile.",
    "m_agent": "Chi guida gli esseri. Gli agenti si misurano tra base (pavimento) e perfetto (soffitto).",
    "m_body": "Come le uscite del cervello diventano movimento. Il cervello non lo sa: lo scopre tarandosi. "
              "Vale per gli agenti col motore (sezione 7).",
    "m_encl": "Quanti recinti nello stesso mondo, indipendenti. Con la telecamera ogni recinto in più rallenta.",
    "m_beings": "Quanti esseri in ogni recinto. Tab li cambia durante la corsa.",
    "load": "Un salvataggio da cui ripartire. Rete, individui e costanti vengono da lì, e quelle righe si bloccano.",
    "save": "Dove salvare: tasto S, fine delle generazioni, uscita dalla corsa.",
}

DETAILS = {
    "mode": {"fast": "Parti dalle statistiche e dalle reti; F apre l'arena quando vuoi. Il compromesso consigliato.",
             "max": "Solo statistiche, nessun disegno: la più rapida per le corse lunghe.",
             "live": "L'arena sempre a schermo: bella da guardare, ma disegnare rallenta."},
    "sim": {"evoluzione": "Evolvono insieme sensi e cervello: gli esseri inventano da soli che cosa percepire.",
            "cervello": "Tutti hanno gli stessi 10 sensi, che non cambiano. Evolve solo il cervello: utile per isolare il problema."},
    "genome": {"diretto": "Ogni sinapsi ha il suo peso scritto nel genoma: semplice, ma il genoma cresce con la rete.",
               "sviluppo": "I pesi nascono dai marcatori delle cellule e da una tabella di affinità, come nello sviluppo "
                           "biologico: geni pochi, che descrivono regole invece di sinapsi."},
    "net": {"rate": "Ogni neurone è un numero tra -1 e 1: la tanh della somma pesata dei suoi ingressi, a ogni passo. Ogni "
                    "collegamento ha il suo segno. Due motori. Veloce e stabile: da preferire per iniziare.",
            "spike-ei": "Ogni neurone accumula carica e la perde col tempo; a soglia 1 spara un impulso (0 o 1) e si azzera. "
                        "Ognuno è tutto eccitatorio o tutto inibitorio, come in natura. Quattro motori a coppie opposte, "
                        "come muscoli. Costa circa il doppio."},
    "task": {"bersaglio": "Raggiungere il bersaglio evitando il veleno. Si lavora a generazioni.",
             "cibo": "Mangiare per riprodursi: si riproduce chi mangia almeno N cibi. Ci sono prede e predatori, niente veleno. "
                     "A generazioni.",
             "arena": "Tutti insieme senza generazioni: nascite e morti continue, la popolazione la decide il cibo.",
             "pile": "Recinti 2D con pile buone e cattive che si scambiano. Sensi fissi: si misura l'apprendimento dentro la vita.",
             "mondo3d": "Le stesse regole delle pile in 3D con MuJoCo: corpi veri, telecamera e profondità. "
                        "Qui si costruisce il cervello ragionante."},
    "ladder": {"scala": "La difficoltà sale se la popolazione va bene e scende se va male.",
               "basico": "Sempre il mondo più facile: controllo per confrontare.",
               "complesso": "Sempre il mondo più difficile: controllo per confrontare."},
    "chance": {"random": "Un seme nuovo a ogni corsa: ogni volta un risultato diverso.",
               "fixed": "Usa il seme della riga sotto: la corsa si può ripetere identica."},
    "p_agent": {"sezioni": "Il progetto: un cervello a parti con compiti diversi. " + SECTIONS_BRAIN,
                "attore-critico": "Il riferimento classico: un critico stima quanto vale la situazione, un attore sceglie "
                                  "l'azione (TD continuo). Dice quanto bene si può fare senza sezioni.",
                "guidato": "Regole a mano che sanno colore buono e zona. Non impara: serve a sapere quanto rende il mondo."},
    "p_phase": {"A": "Si rinasce col cervello di partenza, senza ciò che si era imparato. Nessuna evoluzione: si misura solo "
                     "l'apprendimento.",
                "B": "Si rinasce coi geni di una coppia che si è accoppiata: evoluzione più apprendimento."},
    "p_lesion": {"": "Cervello intero.",
                 "B": "Senza nucleo stabile: le sezioni leggono i sensi direttamente.",
                 "D": "Senza memoria di lavoro: dimentica subito ciò che ha appena visto.",
                 "C": "Senza memoria veloce: non impara in fretta le novità.",
                 "L": "Senza memoria lunga: niente ricordi che durano.",
                 "U": "Senza incertezza: non si accorge che il mondo è cambiato.",
                 "V": "Senza critico: impara dalla ricompensa grezza, senza previsioni."},
    "p_good": {"colore": "Una tinta è buona, l'altra cattiva: rosse o gialle.",
               "grandezza": "Contano solo grandi o piccole; il colore non dice niente. Prova che si impara qualunque regola."},
    "p_teacher": {"": "Nessun maestro: ognuno impara da ciò che gli succede.",
                  "fantasma": "Nessuno lo vede e mangia la sua copia delle pile: si può imparare da lui senza che rubi cibo.",
                  "visibile": "Un compagno vero, che si mangia davvero le pile buone."},
    "m_agent": {"ragionante": "Il motore (sezione 7) con sopra un comandante finto che legge la verità. Per vedere la rete e la "
                              "taratura.",
                "vedente": "Motore più rete nostra: 19 ms per vista. Conosce solo le pile rosse, gialle e gli esseri. "
                           "Il comandante è finto ma sa solo ciò che vede.",
                "vedente finto": "Come il vedente, ma legge gli oggetti dal simulatore e vede attraverso i muri. "
                                 "Serve a provare il resto, non a misurare.",
                "vedente pronto": "Motore più SlimSAM e DINOv2: riconosce oggetti mai visti, 3 s per vista. Il mondo lo aspetta. "
                                  "Richiede: uv run --group percezione main.py",
                "perfetto": "Regole a mano che leggono la verità, anche dietro le spalle. Il soffitto: nessun cervello, "
                            "non usa la telecamera.",
                "base": "Vaga guardando solo la profondità e morde ciò che ha davanti. Il pavimento."},
    "m_body": {"spinta, sterzo, morso": "Ogni uscita fa una cosa sola: il corpo più semplice.",
               "due ruote e morso": "Due uscite per le ruote: per andare dritto vanno mosse insieme, per girare al contrario.",
               "uscite in disordine": "Due ruote e morso, ma con le uscite in un ordine diverso.",
               "morso al contrario": "Due ruote, e la bocca si apre con un'uscita negativa.",
               "quattro motori e morso": "Cinque uscite che si dividono il lavoro: più sinapsi da imparare, taratura meno precisa.",
               "tutto mescolato": "Ogni uscita muove un po' di tutto: nessuna fa una cosa sola."},
}
