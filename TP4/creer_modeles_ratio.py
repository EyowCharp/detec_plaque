"""
Modèles à PROPORTIONS CONSERVÉES (dossier modeles_ratio/) : voir normalisation.normaliser_ratio.
À lancer une fois : python creer_modeles_ratio.py
"""
import os
import cv2 as cv
import creer_modeles as base
from normalisation import normaliser_ratio

DOSSIER = "modeles_ratio"

if __name__ == "__main__":
    os.makedirs(DOSSIER, exist_ok=True)
    police = base.trouver_police()
    for c in base.CHIFFRES + base.LETTRES:
        cv.imwrite(os.path.join(DOSSIER, f"{c}.png"), normaliser_ratio(base.caractere_recadre(c, police)))
    print(len(base.CHIFFRES + base.LETTRES), "modèles créés dans", DOSSIER + "/")
