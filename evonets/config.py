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

    # Campo visivo
    vision_rays: int = 9
    vision_fov_deg: float = 120.0  # apertura totale del ventaglio
    vision_range: float = 30.0  # u
    # Rumore dei sensi: deviazione standard di un disturbo casuale aggiunto a
    # ogni ingresso della rete a ogni passo. 0 = sensi perfetti (lo usa l'esame).
    sensor_noise: float = 0.0

    # Rete: fino a net_hidden_max neuroni interni, liberi di collegarsi in
    # qualsiasi direzione (anelli compresi). Alla nascita ogni individuo ne ha
    # attivi un numero casuale e ogni collegamento possibile esiste con
    # probabilità net_conn_prob.
    # "rate" = neuroni continui (tanh), ogni collegamento col suo segno;
    # "spike-ei" = a impulsi, ogni neurone interno è tutto eccitatorio o tutto
    # inibitorio.
    net_type: str = "rate"
    net_hidden_max: int = 32
    net_weight_std: float = 1.0
    # Geometria (vedi net/geometry.py): il cervello è un quadrato di lato 1,
    # sensi a sinistra, motori a destra. Alla nascita un collegamento esiste con
    # probabilità net_conn_prob * exp(-distanza / net_conn_range): locale quasi
    # sempre, lungo di rado. Con questi valori una rete casuale ha in media
    # tanti collegamenti quanti ne avrebbe con il 15% uniforme, ma lunghi la
    # metà, e quasi sempre almeno un percorso dai sensi ai motori.
    net_conn_prob: float = 0.75  # probabilità a distanza 0
    net_conn_range: float = 0.3  # portata tipica di un collegamento
    net_self_prob: float = 0.05  # un neurone collegato a se stesso: raro, come in natura
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

    # Riproduzione. Il genoma è una fila di blocchi, uno per neurone interno o
    # motore: il blocco contiene se il neurone esiste, la sua soglia e tutti i
    # suoi collegamenti in ingresso con i pesi.
    crossover_rate: float = 0.5  # quota di figli con due genitori (gli altri: copia)
    mut_weight_prob: float = 0.1  # probabilità che un singolo peso o soglia venga ritoccato
    mut_weight_std: float = 0.3  # ampiezza del ritocco
    # Probabilità che un collegamento assente compaia, tra due neuroni a
    # distanza 0; cala con la distanza come alla nascita (in media ~0.2%).
    mut_conn_add: float = 0.01
    mut_conn_del: float = 0.01  # probabilità che un collegamento presente sparisca
    mut_neuron_toggle: float = 0.01  # probabilità che un neurone interno si accenda o si spenga
    mut_pos_prob: float = 0.1  # probabilità che un neurone interno si sposti
    mut_pos_std: float = 0.03  # di quanto, in lati del cervello

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
    "vision_fov_deg",
    "vision_range",
)
