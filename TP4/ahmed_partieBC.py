"""
TP4 - Mini-projet : Détection de plaques et identification des caractères

Partie B (étapes 8-9)  : isoler et normaliser chaque caractère de la plaque.
Partie C (étapes 10-12) : reconnaître les caractères, appliquer le format, afficher.

Ce fichier réutilise la partie A de Lucas (lucas_partieA.py) : on lui prend la plaque
redressée et binarisée (520 x 110, caractères en blanc sur fond noir).

Utilisation :  python ahmed_partieBC.py images/voitures/voiture_01.jpg
"""

import os
import numpy as np
import cv2 as cv

import lucas_partieA as A    # étapes 1 à 7 : trouver, redresser et binariser la plaque


# ---------------------------------------------------------------
# Paramètres (regroupés ici pour pouvoir les régler facilement)
# ---------------------------------------------------------------
TOLERANCE_HAUTEUR = 0.15       # un caractère a la même hauteur que les autres, à 15 % près
NB_CARACTERES = 7              # format SIV : AA-123-AA

LARGEUR_MODELE = 30            # taille commune des caractères et des modèles
HAUTEUR_MODELE = 60

DOSSIER = os.path.dirname(os.path.abspath(__file__))

LETTRES = "ABCDEFGHJKLMNPQRSTVWXYZ"   # sans I, O, U (jamais utilisés sur une plaque SIV)
CHIFFRES = "0123456789"
# Position (0 à 6) -> alphabet autorisé : AA-123-AA
ALPHABET_POSITION = [LETTRES, LETTRES, CHIFFRES, CHIFFRES, CHIFFRES, LETTRES, LETTRES]


# ---------------------------------------------------------------
# Étape 8 : Trouver chaque caractère
# ---------------------------------------------------------------
def trouver_caracteres(binaire):
    """Renvoie les rectangles (x, y, w, h) des caractères, triés de gauche à droite."""
    # RETR_EXTERNAL : contours extérieurs seulement, sinon le trou du 8 ou du A compterait
    contours, _ = cv.findContours(binaire, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    rectangles = [cv.boundingRect(c) for c in contours]

    # Premier tri : les règles de Lucas (hauteur entre 30 et 90 % de la plaque, plus haut que large)
    rectangles = [r for r in rectangles if A.est_un_caractere(*r)]
    if not rectangles:
        return []

    # Deuxième tri : tous les vrais caractères ont la même hauteur.
    # Les bandes bleues (F à gauche, département à droite) sont plus basses : on les écarte.
    hauteur_mediane = float(np.median([h for (_, _, _, h) in rectangles]))
    rectangles = [r for r in rectangles
                  if abs(r[3] - hauteur_mediane) <= TOLERANCE_HAUTEUR * hauteur_mediane]

    # Tri par x croissant : on lit de gauche à droite
    return sorted(rectangles, key=lambda r: r[0])


# ---------------------------------------------------------------
# Étape 9 : Normaliser chaque caractère (30 x 60)
# ---------------------------------------------------------------
# Une SEULE fonction par méthode, utilisée pour fabriquer les modèles ET pour les
# caractères de la plaque : c'est ce qui garantit qu'ils sont comparables.
def normaliser_etire(caractere):
    """Version initiale : étire le caractère pour remplir 30 x 60 (un « 1 » devient une barre épaisse)."""
    c = cv.resize(caractere, (LARGEUR_MODELE, HAUTEUR_MODELE), interpolation=cv.INTER_AREA)
    _, c = cv.threshold(c, 127, 255, cv.THRESH_BINARY)
    return c


def normaliser_ratio(caractere):
    """Version finale : garde les proportions (hauteur 60), centré sur une toile noire 30 x 60."""
    h, w = caractere.shape[:2]
    largeur = max(2, min(LARGEUR_MODELE, round(w * HAUTEUR_MODELE / h)))
    c = cv.resize(caractere, (largeur, HAUTEUR_MODELE), interpolation=cv.INTER_AREA)
    _, c = cv.threshold(c, 127, 255, cv.THRESH_BINARY)
    toile = np.zeros((HAUTEUR_MODELE, LARGEUR_MODELE), np.uint8)
    x0 = (LARGEUR_MODELE - largeur) // 2
    toile[:, x0:x0 + largeur] = c
    return toile


def normaliser(binaire, rectangle, fonction=normaliser_ratio):
    """Découpe le caractère (slicing) et le met à la taille des modèles (30 x 60)."""
    x, y, w, h = rectangle
    return fonction(binaire[y:y + h, x:x + w])


# ---------------------------------------------------------------
# Étape 10 : Comparer aux modèles
# ---------------------------------------------------------------
def charger_modeles(nom_dossier="modeles_ratio"):
    modeles = {}
    for c in CHIFFRES + LETTRES:
        image = cv.imread(os.path.join(DOSSIER, nom_dossier, f"{c}.png"), cv.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Modèle manquant : {nom_dossier}/{c}.png")
        modeles[c] = image
    return modeles


def score_pixels(caractere, modele):
    """Version initiale : proportion de pixels identiques (le fond noir commun gonfle le score)."""
    return float(np.mean(caractere == modele))


def score(caractere, modele):
    """IoU (intersection / union) des pixels blancs : 1.0 = superposition parfaite, 0 = rien en commun.
    Plus sévère que « pixels identiques » : seul le tracé des caractères compte, pas le fond."""
    a, b = caractere > 0, modele > 0
    return float(np.logical_and(a, b).sum() / max(1, np.logical_or(a, b).sum()))


# ---------------------------------------------------------------
# Étape 11 : Appliquer le format AA-123-AA
# ---------------------------------------------------------------
def reconnaitre(caractere, modeles, position, fonction_score=score):
    """Renvoie (caractère lu, score) : on ne compare qu'aux modèles autorisés à cette position."""
    meilleur, meilleur_score = "?", -1.0
    for c in ALPHABET_POSITION[position]:    # position 2 -> chiffres seulement, etc.
        s = fonction_score(caractere, modeles[c])
        if s > meilleur_score:
            meilleur, meilleur_score = c, s
    return meilleur, meilleur_score


def lire_plaque(binaire, modeles, fonction_norm=normaliser_ratio, fonction_score=score):
    """Étapes 8 à 11 : de la plaque binarisée au texte. Renvoie (texte, rectangles, scores)."""
    rectangles = trouver_caracteres(binaire)
    if len(rectangles) != NB_CARACTERES:
        return None, rectangles, []      # il n'y a pas 7 caractères : on ne devine pas
    lettres, scores = [], []
    for position, rectangle in enumerate(rectangles):
        c, s = reconnaitre(normaliser(binaire, rectangle, fonction_norm), modeles, position, fonction_score)
        lettres.append(c)
        scores.append(s)
    t = "".join(lettres)
    return f"{t[:2]}-{t[2:5]}-{t[5:]}", rectangles, scores


# ---------------------------------------------------------------
# Étape 12 : Afficher le résultat
# ---------------------------------------------------------------
def dessiner_resultat(image_originale, rectangle_plaque, echelle, texte):
    """Dessine la plaque et le texte lu sur l'image d'ORIGINE (coordonnées / echelle)."""
    resultat = image_originale.copy()
    coins = np.int32(cv.boxPoints(rectangle_plaque) / echelle)
    cv.polylines(resultat, [coins], True, (0, 255, 0), max(2, int(3 / echelle)))
    x, y = coins.min(axis=0)
    cv.putText(resultat, texte, (int(x), max(30, int(y) - 10)), cv.FONT_HERSHEY_SIMPLEX,
               1.0 / echelle, (0, 255, 0), max(2, int(3 / echelle)))
    return resultat


def dessiner_caracteres(binaire, rectangles, texte):
    """Plaque binarisée avec un cadre autour de chaque caractère, et la lettre lue dessous."""
    resultat = cv.cvtColor(binaire, cv.COLOR_GRAY2BGR)
    for i, (x, y, w, h) in enumerate(rectangles):
        cv.rectangle(resultat, (x, y), (x + w, y + h), (0, 255, 0), 1)
        if texte:
            lettre = texte.replace("-", "")[i]
            cv.putText(resultat, lettre, (x + 2, y + h + 14), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    return resultat


# ---------------------------------------------------------------
# Programme principal : toute la chaîne, étapes 1 à 12
# ---------------------------------------------------------------
def traiter_image(chemin, modeles, fonction_norm=normaliser_ratio, fonction_score=score):
    # Partie A (Lucas) : trouver, redresser et binariser la plaque
    image = A.charger_image(chemin)
    image_redim, echelle = A.redimensionner(image)
    flou = A.debruiter(A.niveaux_de_gris(image_redim))
    contours = A.trouver_formes(A.detecter_bords(flou))
    candidats = A.chercher_candidats(contours, image_redim.shape[1])
    rectangle, _ = A.choisir_plaque(image_redim, candidats)
    if rectangle is None:
        return None, image, None, None, []
    plaque = A.redresser_plaque(image_redim, rectangle)
    binaire = A.binariser_plaque(plaque)
    # Parties B et C : lire les caractères
    texte, rectangles, scores = lire_plaque(binaire, modeles, fonction_norm, fonction_score)
    return texte, image, (rectangle, echelle), binaire, (rectangles, scores, plaque)


if __name__ == "__main__":
    import sys
    chemin = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DOSSIER, "images", "voitures", "voiture_01.jpg")
    jeu = sys.argv[2] if len(sys.argv) > 2 else "modeles_ratio"
    modeles = charger_modeles(jeu)
    texte, image, plaque_info, binaire, details = traiter_image(chemin, modeles)
    if plaque_info is None:
        print("Aucune plaque trouvée.")
        sys.exit(1)
    rectangles, scores, plaque = details
    print(f"Plaque lue : {texte}   ({len(rectangles)} caractères détectés)")
    if texte:
        print("Scores     :", " ".join(f"{s:.0%}" for s in scores))
    # Phase 1 : les etapes de Lucas (partie A)
    image_redim, _ = A.redimensionner(image)
    gris = A.niveaux_de_gris(image_redim)
    flou = A.debruiter(gris)
    bords = A.detecter_bords(flou)
    contours = A.trouver_formes(bords)
    candidats = A.chercher_candidats(contours, image_redim.shape[1])
    cv.imshow("A1 - Image redimensionnee", image_redim)
    cv.imshow("A2 - Niveaux de gris", gris)
    cv.imshow("A3 - Debruitage (bilateral)", flou)
    cv.imshow("A4 - Contours (Canny)", bords)
    cv.imshow("A5 - Toutes les formes", A.dessiner_formes(image_redim, contours))
    cv.imshow("A6 - Candidats", A.dessiner_candidats(image_redim, candidats))
    cv.imshow("A7 - Plaque redressee", plaque)
    # Phase 2 : nos etapes (parties B et C), affichees en meme temps
    resultat = dessiner_resultat(image, plaque_info[0], plaque_info[1], texte or "?")
    cv.imshow("12 - Resultat", resultat)
    cv.imshow("8 - Caracteres trouves", dessiner_caracteres(binaire, rectangles, texte))
    cv.waitKey(0)
    cv.destroyAllWindows()
