"""
Étape 0 : crée les images de référence (modèles) des caractères d'une plaque française.
33 caractères : 10 chiffres + 23 lettres (le format SIV n'utilise jamais I, O, U).
Chaque modèle : 30 x 60 px, caractère BLANC sur fond NOIR, recadré au plus juste.
À lancer une seule fois : python creer_modeles.py

Police : sans-serif "bâtons" épaisse (Arial Bold), proche de celle des plaques.
(La police intégrée d'OpenCV, putText, barre le 0 d'une diagonale : inutilisable ici.)
"""
import os
import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LARGEUR, HAUTEUR = 30, 60                 # taille commune à tous les modèles
CHIFFRES = "0123456789"
LETTRES = "ABCDEFGHJKLMNPQRSTVWXYZ"       # 26 lettres sans I, O, U = 23
DOSSIER = "modeles"
POLICES = [                               # première police trouvée
    r"C:\Windows\Fonts\arialbd.ttf",                              # Windows
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",  # équivalent d'Arial (Linux)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def trouver_police(taille=150):
    for chemin in POLICES:
        if os.path.exists(chemin):
            return ImageFont.truetype(chemin, taille)
    raise SystemExit("Aucune police trouvée : adapte la liste POLICES")


def caractere_recadre(caractere, police):
    """Étapes 1 à 3 : dessiner, binariser, recadrer au plus juste (blanc sur noir)."""
    toile = Image.new("L", (300, 300), 0)
    ImageDraw.Draw(toile).text((60, 40), caractere, fill=255, font=police)
    toile = np.array(toile)
    _, toile = cv.threshold(toile, 127, 255, cv.THRESH_BINARY)
    x, y, w, h = cv.boundingRect(toile)
    return toile[y:y + h, x:x + w]


def creer_modele(caractere, police):
    # 1. On écrit le caractère en blanc sur une grande image noire
    toile = Image.new("L", (300, 300), 0)
    ImageDraw.Draw(toile).text((60, 40), caractere, fill=255, font=police)
    toile = np.array(toile)
    # 2. On binarise (blanc pur / noir pur), comme le seront les caractères découpés
    _, toile = cv.threshold(toile, 127, 255, cv.THRESH_BINARY)
    # 3. On recadre au plus juste autour du caractère (boundingRect)
    x, y, w, h = cv.boundingRect(toile)
    toile = toile[y:y + h, x:x + w]
    # 4. On redimensionne à la taille commune. C'est EXACTEMENT ce que fera l'étape 9
    #    sur les caractères découpés dans la plaque : modèles et caractères sont donc comparables.
    modele = cv.resize(toile, (LARGEUR, HAUTEUR), interpolation=cv.INTER_AREA)
    _, modele = cv.threshold(modele, 127, 255, cv.THRESH_BINARY)
    return modele


if __name__ == "__main__":
    os.makedirs(DOSSIER, exist_ok=True)
    police = trouver_police()
    for c in CHIFFRES + LETTRES:
        cv.imwrite(os.path.join(DOSSIER, f"{c}.png"), creer_modele(c, police))
    print(len(CHIFFRES + LETTRES), "modèles créés dans", DOSSIER + "/")
