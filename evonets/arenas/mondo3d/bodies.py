"""I corpi del mondo 3D: come le uscite del cervello diventano movimento."""

# Da K uscite a (spinta, sterzo, bocca): nome -> (cablaggio (3, K), nomi delle uscite).
BODIES = {
    "spinta, sterzo, morso": (((1, 0, 0), (0, 1, 0), (0, 0, 1)), ("spinta", "sterzo", "morso")),
    "due ruote e morso": (((.5, .5, 0), (-.5, .5, 0), (0, 0, 1)), ("ruota sx", "ruota dx", "morso")),
    "uscite in disordine": (((0, .5, .5), (0, .5, -.5), (1, 0, 0)), ("morso", "ruota dx", "ruota sx")),
    "morso al contrario": (((.5, .5, 0), (-.5, .5, 0), (0, 0, -1)), ("ruota sx", "ruota dx", "morso")),
    "quattro motori e morso": (((.25, .25, .25, .25, 0), (-.25, .25, -.25, .25, 0), (0, 0, 0, 0, 1)),
                               ("motore 1", "motore 2", "motore 3", "motore 4", "morso")),
    "tutto mescolato": (((-.6, .35, .2), (.4, .8, -.1), (.3, -.2, .9)), ("uscita 1", "uscita 2", "uscita 3")),
}
FELT = ("velocità", "rotazione", "bocca")  # ciò che il corpo sente di sé, tra -1 e 1
