from dataclasses import dataclass


@dataclass
class Config:
    """Tutte le costanti fisiche in un posto solo. Unità: u (lunghezza), s (tempo)."""

    # Arena
    arena_w: float = 100.0
    arena_h: float = 70.0
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
    # fermo, meno se ci si muove o si attraversa il veleno.
    energy_init: float = 100.0
    metabolism_basal: float = 5.0  # energia/s solo per esistere
    metabolism_move: float = 5.0  # energia/s in più a spinta piena (avanti o indietro)
    metabolism_turn: float = 1.0  # energia/s in più a sterzata piena

    # Campo visivo
    vision_rays: int = 9
    vision_fov_deg: float = 120.0  # apertura totale del ventaglio
    vision_range: float = 30.0  # u

    # Rete: fino a net_hidden_max neuroni interni, liberi di collegarsi in
    # qualsiasi direzione (anelli compresi). Alla nascita ogni individuo ne ha
    # attivi un numero casuale e ogni collegamento possibile esiste con
    # probabilità net_conn_prob.
    net_hidden_max: int = 32
    net_conn_prob: float = 0.15
    net_weight_std: float = 1.0

    # Generazioni. Una generazione finisce quando tutti sono morti o hanno
    # completato il compito. Si riproduce solo chi lo ha completato.
    population: int = 200
    life_max: float = 60.0  # s, tetto di sicurezza se il metabolismo è spento
    max_age: int = 5  # generazioni che un individuo può vivere al massimo
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
    mut_conn_add: float = 0.002  # probabilità che un collegamento assente compaia
    mut_conn_del: float = 0.01  # probabilità che un collegamento presente sparisca
    mut_neuron_toggle: float = 0.01  # probabilità che un neurone interno si accenda o si spenga

    # Compito
    target_radius: float = 3.0

    # Pericoli (zone tossiche in deriva armonica). Il danno cresce dal bordo
    # (zero) al centro (hazard_drain) come profondità ** hazard_falloff.
    hazard_drain: float = 60.0  # energia/s al centro
    hazard_falloff: float = 2.0  # >1 = bordi miti e nucleo severo
    n_hazards: int = 5
    hazard_radius_min: float = 4.0
    hazard_radius_max: float = 8.0
    hazard_omega_min: float = 0.2  # rad/s
    hazard_omega_max: float = 0.8
    hazard_pulse: float = 0.15  # ampiezza relativa della pulsazione del raggio
    hazard_speed: float = 1.0  # moltiplicatore globale (leva del curriculum)
    spawn_margin: float = 3.0  # distanza minima dai pericoli allo spawn


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
    "hazard_speed",
    "hazard_drain",
    "hazard_falloff",
    "vision_fov_deg",
    "vision_range",
)
