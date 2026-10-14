from dataclasses import dataclass


@dataclass
class Config:
    """Tutte le costanti fisiche in un posto solo. Unità: u (lunghezza), s (tempo)."""

    # Arena
    arena_w: float = 200.0
    arena_h: float = 140.0
    dt: float = 1.0 / 60.0

    # Agente (spinta + rotazione). Velocità limite ≈ thrust_accel / lin_drag.
    agent_radius: float = 1.0
    thrust_accel: float = 60.0  # u/s² a comando pieno
    reverse_factor: float = 0.5  # la retromarcia spinge meno
    lin_drag: float = 2.0  # 1/s, lungo l'asse dell'agente
    lat_drag: float = 8.0  # 1/s, trasversale: più alto = meno derapata
    turn_accel: float = 40.0  # rad/s² a comando pieno
    ang_drag: float = 6.0  # 1/s
    wall_restitution: float = 0.3

    # Energia: unica riserva dell'agente, a zero si muore. Non si ricarica, quindi
    # è anche la durata della vita: energy_init / metabolism_basal secondi da
    # fermo, meno se ci si muove o si attraversa il veleno. 150 e non 100: con
    # 100 anche una rete buona fallisce spesso per sfortuna e la selezione è
    # rumorosa; con 150 le reti evolute riescono meglio anche se poi messe alla
    # prova con 100 (vedi docs/NOTE.md, "Energia iniziale").
    energy_init: float = 150.0
    metabolism_basal: float = 2.5  # energia/s solo per esistere
    metabolism_move: float = 2.5  # energia/s in più a spinta piena (avanti o indietro)
    metabolism_turn: float = 0.5  # energia/s in più a sterzata piena
    # Costo del cablaggio: energia/s per ogni unità di lunghezza di
    # collegamento effettivo (il cervello è un quadrato di lato 1). Una rete
    # iniziale tipica, lunga in tutto ~60, paga ~0.25/s: un decimo del
    # metabolismo di base. Come in natura, un filo lungo deve valere il suo prezzo.
    wire_cost: float = 0.004

    # Fisica dei segnali (sim/signals.py), per i recettori universali. Ogni
    # fenomeno ha un asse della "qualità" in [0, 1]: colore, molecola,
    # frequenza, temperatura. Cosa emette ogni oggetto è scritto qui: dare un
    # suono al bersaglio o un odore al veleno è cambiare un numero.
    # Luce: mezzo torbido, si attenua come exp(-d / light_attenuation).
    light_attenuation: float = 12.0  # u
    light_samples: int = 5  # direzioni campione nell'apertura di un recettore
    wall_color: float = 0.5
    wall_color_width: float = 0.3
    wall_albedo: float = 0.4
    target_color: float = 0.85
    target_color_width: float = 0.06
    target_albedo: float = 1.0
    hazard_color: float = 0.25
    hazard_color_width: float = 0.06
    hazard_glow: float = 0.05  # luce emessa dal veleno per unità di lunghezza attraversata
    # Sostanze: la tossina del veleno (dentro la zona = intensità del danno),
    # più un odore che si allontana come exp(-d / smell_range).
    toxin_molecule: float = 0.15
    hazard_smell: float = 0.0  # tossina che esce dalla zona; 0 = inodore fuori
    target_molecule: float = 0.6
    target_smell: float = 0.0
    smell_range: float = 10.0  # u
    # Meccanica: pressione da contatto (frequenza 0) e vibrazioni a distanza.
    vibration_range: float = 20.0  # u
    target_vibration: float = 0.0  # ampiezza; 0 = silenzioso
    target_vibration_freq: float = 0.6
    hazard_vibration: float = 0.0
    hazard_vibration_freq: float = 0.3
    # Temperatura: ambiente uniforme più eventuali sorgenti di calore.
    ambient_temperature: float = 0.5
    heat_range: float = 15.0  # u
    target_heat: float = 0.0
    hazard_heat: float = 0.0
    # Magnetismo: campo uniforme, come quello di un pianeta. 0 = nessun campo.
    magnetic_field: float = 0.0
    magnetic_angle_deg: float = 90.0
    # Recettori: rumore di fondo e saturazione, in "quanti" per passo.
    receptor_dark: float = 1.0
    receptor_saturation: float = 100.0
    # Costo del corpo e del cervello, energia/s: ogni recettore acceso (la
    # cellula), la sua sensibilità (somma delle g di tutti i fenomeni) e ogni
    # neurone interno acceso. Al posto di un numero massimo, un prezzo.
    receptor_cell_cost: float = 0.002
    receptor_sens_cost: float = 0.00005
    neuron_cost: float = 0.01

    # Rumore dei sensi: deviazione standard di un disturbo casuale aggiunto a
    # ogni ingresso della rete a ogni passo. 0 = sensi perfetti (lo usa l'esame).
    sensor_noise: float = 0.0

    # Rete: neuroni interni liberi di collegarsi in qualsiasi direzione
    # (anelli compresi), senza numero massimo: ognuno costa (neuron_cost) e se
    # ne aggiungono per duplicazione. Si parte primordiali (docs/SPEC_FASE_C.md,
    # sezione 7): nessun neurone interno acceso e pochi recettori generalisti.
    # "rate" = neuroni continui (tanh), ogni collegamento col suo segno;
    # "spike-ei" = a impulsi, ogni neurone interno è tutto eccitatorio o tutto
    # inibitorio.
    net_type: str = "rate"
    net_weight_std: float = 1.0
    # Posti iniziali (spenti) per neuroni interni e recettori. Non sono limiti:
    # quando qualcuno ne usa più di pool_fill, tutti ricevono pool_step posti.
    net_hidden_slots: int = 8
    receptor_slots: int = 4
    pool_fill: float = 0.8
    pool_step: int = 8
    # Il corpo primordiale: recettori accesi, a ventaglio davanti, generalisti
    # (sensibilità uguale su tutti i fenomeni, sintonia larga).
    receptor_initial: int = 2
    receptor_initial_spread_deg: float = 60.0
    receptor_initial_aperture_deg: float = 120.0
    receptor_initial_gain: float = 5.0
    receptor_initial_width: float = 0.4
    # Adattamento sensoriale alla partenza: tempo caratteristico (s) e
    # intensità massima, sottrattiva e divisiva, sorteggiate tra 0 e questo valore.
    receptor_initial_tau: float = 1.0
    receptor_initial_adaptation: float = 0.2
    # Geometria (vedi net/geometry.py): il cervello è un quadrato di lato 1,
    # sensi a sinistra, motori a destra. Alla nascita un collegamento esiste con
    # probabilità net_conn_prob * exp(-distanza / net_conn_range): locale quasi
    # sempre, lungo di rado. Con questi valori una rete casuale ha in media
    # tanti collegamenti quanti ne avrebbe con il 15% uniforme, ma lunghi la
    # metà, e quasi sempre almeno un percorso dai sensi ai motori.
    net_conn_prob: float = 0.75  # probabilità a distanza 0
    net_conn_range: float = 0.3  # portata tipica di un collegamento (valore iniziale: poi è un gene)
    net_self_prob: float = 0.05  # un neurone collegato a se stesso: raro, come in natura
    # Plasticità (docs/SPEC_FASE_E.md): le sinapsi cambiano durante la vita
    # secondo una regola i cui coefficienti sono geni del neurone che riceve,
    # modulata da M, che i neuroni modulatori rilasciano e che si diffonde nel
    # cervello calando come exp(-d / modulation_range). Ciò che si impara non si
    # eredita. plasticity = False la spegne del tutto (per confronti).
    plasticity: bool = True
    plasticity_initial_rate: float = 0.001  # η0: velocità di partenza (il gene la moltiplica)
    plasticity_initial_rule: float = 0.5  # deviazione dei coefficienti A, B, C, D alla partenza
    modulatory_initial_share: float = 0.1  # quota di neuroni che nasce modulatrice
    modulation_range: float = 0.2  # quanto lontano arriva il modulatore, in lati del cervello
    synapse_max: float = 4.0  # forza massima di una sinapsi: le sinapsi vere saturano
    # Solo "spike-ei": passi di rete per ogni passo di fisica, e guadagno della
    # corrente in ingresso ai neuroni interni.
    spike_substeps: int = 4
    spike_gain: float = 3.0
    # Quota di neuroni interni inibitori alla nascita, e se
    # anche i sensi possono soltanto eccitare (False = liberi di fare entrambe le cose).
    spike_inhibitory_share: float = 0.2
    sensors_excitatory_only: bool = False

    # Generazioni. Una generazione finisce quando tutti sono morti o hanno
    # completato il compito. Si riproduce solo chi lo ha completato.
    population: int = 200
    life_max: float = 90.0  # s, tetto di sicurezza se il metabolismo è spento
    # Generazioni che un individuo può vivere al massimo; 0 = nessun limite: un
    # genitore resta finché continua a completare il compito.
    max_age: int = 0
    # Con un solo genitore, o nessuno (si riparte dagli ultimi genitori noti),
    # questa quota di figli nasce con mutazioni più forti.
    rescue_share: float = 0.5
    rescue_strength: float = 3.0

    # Riproduzione. Il genoma è una fila di blocchi: uno per neurone interno o
    # motore (esistenza, geni propri, posizione, collegamenti in ingresso),
    # uno per recettore, uno per il meccanismo di mutazione.
    crossover_rate: float = 0.5  # quota di figli con due genitori (gli altri: copia)
    # Mutazioni: questi sono solo i valori di PARTENZA. Ogni individuo ha i
    # suoi tassi come geni, che mutano come tutti gli altri geni (con la
    # probabilità e l'ampiezza di ritocco dell'individuo stesso) e non scendono
    # sotto mutation_floor (copiatura mai perfetta).
    mut_weight_prob: float = 0.1  # probabilità che un peso o gene venga ritoccato
    mut_weight_std: float = 0.3  # ampiezza del ritocco
    # Probabilità che un collegamento assente compaia, tra due neuroni a
    # distanza 0; cala con la distanza come alla nascita.
    mut_conn_add: float = 0.01
    mut_conn_del: float = 0.01  # probabilità che un collegamento presente sparisca
    mut_neuron_toggle: float = 0.01  # probabilità che un neurone o recettore si accenda o si spenga
    mut_dup: float = 0.005  # probabilità che un neurone o recettore acceso si duplichi
    mut_pos_prob: float = 0.1  # probabilità che un neurone (o un nervo) si sposti nel cervello
    mut_pos_std: float = 0.03  # di quanto, in lati del cervello
    mut_rec_drift: float = 0.05  # deriva dei geni dei recettori (scala: genetics.DRIFT_SCALE)
    mutation_floor: float = 1e-5

    # Compito
    target_radius: float = 3.0

    # Pericoli (zone tossiche in deriva armonica). Il danno cresce dal bordo
    # (zero) al centro (hazard_drain) come profondità ** hazard_falloff.
    hazard_drain: float = 60.0  # energia/s al centro
    hazard_falloff: float = 2.0  # >1 = bordi miti e nucleo severo
    n_hazards: int = 12
    hazard_radius_min: float = 5.0
    hazard_radius_max: float = 12.0
    hazard_omega_min: float = 0.2  # rad/s
    hazard_omega_max: float = 0.8
    hazard_pulse: float = 0.15  # ampiezza relativa della pulsazione del raggio
    hazard_speed: float = 1.0  # moltiplicatore globale (leva del curriculum)
    spawn_margin: float = 3.0  # distanza minima dai pericoli allo spawn

    # Esame (evo/exam.py): ogni quante generazioni metterlo in automatico ai
    # più anziani della popolazione; 0 = mai (resta il tasto E). Non cambia
    # l'evoluzione: è solo una misura. Gira in processi separati, su tutti i
    # core tranne due: circa 12 s su 12 core (un minuto e più su uno solo).
    exam_every: int = 50


# Costanti regolabili a runtime dal viewer (Tab per scegliere, [ ] per variare).
TUNABLES = (
    "thrust_accel",
    "lin_drag",
    "lat_drag",
    "turn_accel",
    "ang_drag",
    "wall_restitution",
    "metabolism_basal",
    "metabolism_move",
    "wire_cost",
    "hazard_speed",
    "hazard_drain",
    "hazard_falloff",
    "light_attenuation",
)
