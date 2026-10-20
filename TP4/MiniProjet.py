"""
TP4 - Mini-projet : Détection de plaques et identification des caractères

Partie A (étapes 1 à 6) : trouver la plaque dans la photo et la redresser.

Utilisation :  python TP4/MiniProjet.py
"""

import sys
import numpy as np
import cv2 as cv


# ---------------------------------------------------------------
# Paramètres (regroupés ici pour pouvoir les régler facilement)
# ---------------------------------------------------------------
LARGEUR_TRAVAIL = 800          # largeur à laquelle on ramène toutes les images

# Forme d'une plaque candidate (étape 6a)
LONGUEUR_MIN = 0.08            # longueur min de la plaque : 8 % de la largeur de l'image
LONGUEUR_MAX = 0.60            # longueur max de la plaque : 60 % de la largeur de l'image
EPAISSEUR_MIN = 10             # hauteur min de la plaque, en pixels
RATIO_MIN = 2.0                # rapport longueur / hauteur (plaque française : 520 / 110 = 4,7,
RATIO_MAX = 6.5                # mais la perspective l'écrase souvent)

LARGEUR_PLAQUE = 520           # taille de la plaque une fois redressée
HAUTEUR_PLAQUE = 110

# Ce qu'on considère comme un caractère (étape 6c)
HAUTEUR_CARACTERE_MIN = 0.30   # un caractère fait entre 30 % ...
HAUTEUR_CARACTERE_MAX = 0.90   # ... et 90 % de la hauteur de la plaque
LARGEUR_CARACTERE_MIN = 0.02   # et au moins 2 % de sa largeur
NB_CARACTERES_MIN = 4          # en dessous, le candidat n'est pas une plaque


# ---------------------------------------------------------------
# Étape 1 : Charger et redimensionner l'image
# ---------------------------------------------------------------
def charger_image(chemin):
    image = cv.imread(chemin)
    if image is None:
        print(f"Impossible de lire l'image '{chemin}'. Vérifiez le chemin.")
        sys.exit(1)
    return image


def redimensionner(image):
    hauteur, largeur = image.shape[:2]
    echelle = LARGEUR_TRAVAIL / largeur
    nouvelle_hauteur = int(hauteur * echelle)

    image_redim = cv.resize(image, (LARGEUR_TRAVAIL, nouvelle_hauteur))

    # On renvoie aussi l'échelle : elle servira à l'étape 12 pour
    # replacer le résultat sur l'image d'origine
    return image_redim, echelle


# ---------------------------------------------------------------
# Étape 2 : Passer en niveaux de gris
# ---------------------------------------------------------------
def niveaux_de_gris(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


# ---------------------------------------------------------------
# Étape 3 : Débruiter avec un filtre bilatéral
# ---------------------------------------------------------------
def debruiter(gris):
    taille_voisinage = 11   # diamètre du voisinage pris en compte
    sigma_couleur = 17      # écart d'intensité max pour qu'un voisin compte dans la moyenne
    sigma_espace = 17       # influence de la distance au pixel central
    return cv.bilateralFilter(gris, taille_voisinage, sigma_couleur, sigma_espace)


# ---------------------------------------------------------------
# Étape 4 : Détecter les contours (Canny)
# ---------------------------------------------------------------
def detecter_bords(flou):
    seuil_bas = 30
    seuil_haut = 200
    return cv.Canny(flou, seuil_bas, seuil_haut)


# ---------------------------------------------------------------
# Étape 5 : Trouver les formes
# ---------------------------------------------------------------
def trouver_formes(bords):
    contours, _ = cv.findContours(bords, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    return contours


def dessiner_formes(image, contours):
    resultat = image.copy()
    cv.drawContours(resultat, contours, -1, (0, 0, 255), 1)   # tous en rouge
    return resultat


# ---------------------------------------------------------------
# Étape 6a : Garder les formes qui ont l'allure d'une plaque
# ---------------------------------------------------------------
def a_la_forme_d_une_plaque(rectangle, largeur_image):
    # minAreaRect renvoie : (centre), (largeur, hauteur), angle
    _, (w, h), _ = rectangle
    longueur = max(w, h)    # le plus grand côté, quel que soit l'angle
    epaisseur = min(w, h)   # le plus petit côté

    if epaisseur < EPAISSEUR_MIN:
        return False
    if not (LONGUEUR_MIN * largeur_image <= longueur <= LONGUEUR_MAX * largeur_image):
        return False

    ratio = longueur / epaisseur
    return RATIO_MIN <= ratio <= RATIO_MAX


def chercher_candidats(contours, largeur_image):
    candidats = []
    for contour in contours:
        # Plus petit rectangle (éventuellement tourné) qui contient tout le contour
        rectangle = cv.minAreaRect(contour)
        if a_la_forme_d_une_plaque(rectangle, largeur_image):
            candidats.append(rectangle)
    return candidats


def dessiner_rectangle(image, rectangle, couleur, texte=None):
    coins = np.int32(cv.boxPoints(rectangle))
    cv.drawContours(image, [coins], -1, couleur, 2)
    if texte is not None:
        x, y = coins.min(axis=0)
        cv.putText(image, texte, (int(x), int(y) - 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, couleur, 2)
    return image


# ---------------------------------------------------------------
# Étape 6b : Redresser un candidat
# ---------------------------------------------------------------
def ordonner_coins(points):
    somme = points.sum(axis=1)               # x + y
    difference = np.diff(points, axis=1)     # y - x

    haut_gauche = points[np.argmin(somme)]
    bas_droit = points[np.argmax(somme)]
    haut_droit = points[np.argmin(difference)]
    bas_gauche = points[np.argmax(difference)]

    return np.array([haut_gauche, haut_droit, bas_droit, bas_gauche], dtype=np.float32)


def redresser_plaque(image, rectangle):
    coins_source = ordonner_coins(cv.boxPoints(rectangle))

    # Où l'on veut envoyer chaque coin : les 4 coins d'un rectangle droit
    coins_destination = np.array([
        [0, 0],
        [LARGEUR_PLAQUE - 1, 0],
        [LARGEUR_PLAQUE - 1, HAUTEUR_PLAQUE - 1],
        [0, HAUTEUR_PLAQUE - 1],
    ], dtype=np.float32)

    matrice = cv.getPerspectiveTransform(coins_source, coins_destination)
    return cv.warpPerspective(image, matrice, (LARGEUR_PLAQUE, HAUTEUR_PLAQUE))


# ---------------------------------------------------------------
# Étape 6c : Vérifier qu'un candidat contient bien des caractères
# ---------------------------------------------------------------
def binariser_plaque(plaque):
    gris = cv.cvtColor(plaque, cv.COLOR_BGR2GRAY)

    # Otsu choisit le seuil tout seul ; INV met les caractères (sombres) en blanc
    _, binaire = cv.threshold(gris, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

    # "Pot de peinture" noir lancé depuis le bord : efface tout ce qui touche
    # le bord de l'image (cadre de la plaque, carrosserie), mais pas les caractères
    binaire = cv.copyMakeBorder(binaire, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=255)
    cv.floodFill(binaire, None, (0, 0), 0)
    return binaire[1:-1, 1:-1]   # on retire la bordure ajoutée


def est_un_caractere(x, y, w, h):
    hauteur_ok = HAUTEUR_CARACTERE_MIN * HAUTEUR_PLAQUE <= h <= HAUTEUR_CARACTERE_MAX * HAUTEUR_PLAQUE
    largeur_ok = LARGEUR_CARACTERE_MIN * LARGEUR_PLAQUE <= w < h   # plus haut que large
    return hauteur_ok and largeur_ok


def compter_caracteres(binaire):
    contours, _ = cv.findContours(binaire, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    nb = 0
    for contour in contours:
        if est_un_caractere(*cv.boundingRect(contour)):
            nb += 1
    return nb


def choisir_plaque(image, candidats):
    meilleur = None   # (nb_caracteres, aire, rectangle)
    for rectangle in candidats:
        nb = compter_caracteres(binariser_plaque(redresser_plaque(image, rectangle)))
        _, (w, h), _ = rectangle
        # On garde le candidat avec le plus de caractères (à égalité : le plus grand)
        if meilleur is None or (nb, w * h) > meilleur[:2]:
            meilleur = (nb, w * h, rectangle)

    if meilleur is None or meilleur[0] < NB_CARACTERES_MIN:
        return None, 0
    return meilleur[2], meilleur[0]


def dessiner_candidats(image, candidats):
    resultat = image.copy()
    for rectangle in candidats:
        nb = compter_caracteres(binariser_plaque(redresser_plaque(image, rectangle)))
        dessiner_rectangle(resultat, rectangle, (0, 165, 255), str(nb))   # en orange
    return resultat


# ---------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------
if __name__ == "__main__":
    chemin = "TP4/img_voitures/voiture_01.jpg"

    # Partie A
    image = charger_image(chemin)
    image_redim, echelle = redimensionner(image)
    gris = niveaux_de_gris(image_redim)
    flou = debruiter(gris)
    bords = detecter_bords(flou)
    contours = trouver_formes(bords)
    candidats = chercher_candidats(contours, image_redim.shape[1])
    rectangle, nb_caracteres = choisir_plaque(image_redim, candidats)

    print(f"{len(contours)} formes trouvées, {len(candidats)} candidats retenus.")

    cv.imshow("1 - Image redimensionnee", image_redim)
    cv.imshow("2 - Niveaux de gris", gris)
    cv.imshow("3 - Debruitage (bilateral)", flou)
    cv.imshow("4 - Contours (Canny)", bords)
    cv.imshow("5 - Toutes les formes", dessiner_formes(image_redim, contours))
    cv.imshow("6a - Candidats (nb de caracteres)", dessiner_candidats(image_redim, candidats))

    if rectangle is None:
        print("Aucune plaque trouvée. Regardez les fenêtres 4 à 6a pour comprendre pourquoi.")
    else:
        print(f"Plaque trouvée : {nb_caracteres} caractères détectés.")
        plaque = redresser_plaque(image_redim, rectangle)
        image_plaque = dessiner_rectangle(image_redim.copy(), rectangle, (0, 255, 0))
        cv.imshow("6b - Plaque choisie", image_plaque)
        cv.imshow("6c - Plaque redressee", plaque)
        cv.imshow("6d - Plaque binarisee (verification)", binariser_plaque(plaque))

    print("Appuyez sur une touche dans une fenêtre d'image pour quitter.")
    cv.waitKey(0)
    cv.destroyAllWindows()
