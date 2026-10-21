"""
Normalisation d'un caractère binaire (blanc sur noir, recadré au plus juste) en 30 x 60.

UNE SEULE fonction par méthode, utilisée à la fois pour fabriquer les modèles
(creer_modeles*.py) et pour les caractères découpés dans la plaque (lecture.py) :
c'est ce qui garantit que les deux sont comparables.
"""
import numpy as np
import cv2 as cv

LARGEUR_MODELE = 30
HAUTEUR_MODELE = 60


def normaliser_etire(caractere):
    """Étire le caractère pour remplir 30 x 60. Défaut : un « 1 » étroit devient une barre épaisse."""
    c = cv.resize(caractere, (LARGEUR_MODELE, HAUTEUR_MODELE), interpolation=cv.INTER_AREA)
    _, c = cv.threshold(c, 127, 255, cv.THRESH_BINARY)
    return c


def normaliser_ratio(caractere):
    """Garde les proportions : hauteur 60, largeur proportionnelle, centré sur une toile 30 x 60."""
    h, w = caractere.shape[:2]
    largeur = max(2, min(LARGEUR_MODELE, round(w * HAUTEUR_MODELE / h)))
    c = cv.resize(caractere, (largeur, HAUTEUR_MODELE), interpolation=cv.INTER_AREA)
    _, c = cv.threshold(c, 127, 255, cv.THRESH_BINARY)
    toile = np.zeros((HAUTEUR_MODELE, LARGEUR_MODELE), np.uint8)
    x0 = (LARGEUR_MODELE - largeur) // 2
    toile[:, x0:x0 + largeur] = c
    return toile
