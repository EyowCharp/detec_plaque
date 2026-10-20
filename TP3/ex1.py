"""
TP3 - Partie 1 : Renforcer les contours
Détection de contours avec Canny, puis renforcement avec la transformée de Hough.

Utilisation :  python tp3_partie1_contours.py mon_image.jpg
"""

import sys
import numpy as np
import cv2 as cv


# ---------------------------------------------------------------
# 1. Charger l'image
# ---------------------------------------------------------------
chemin = "img/Fleuve.jpg"
image = cv.imread(chemin)

if image is None:
    print(f"Impossible de lire l'image '{chemin}'. Vérifiez le chemin.")
    sys.exit(1)


# ---------------------------------------------------------------
# 2. Convertir en niveaux de gris
# ---------------------------------------------------------------
gris = cv.cvtColor(image, cv.COLOR_BGR2GRAY)


# ---------------------------------------------------------------
# 3. Flouter légèrement pour réduire le bruit
# ---------------------------------------------------------------
flou = cv.GaussianBlur(gris, (5, 5), 0)


# ---------------------------------------------------------------
# 4. Détection de contours avec Canny
# ---------------------------------------------------------------
seuil_bas = 50
seuil_haut = 150
contours = cv.Canny(flou, seuil_bas, seuil_haut)


# ---------------------------------------------------------------
# 5. Transformée de Hough : chercher les lignes droites
# ---------------------------------------------------------------
resolution_rho = 1             # précision de la distance, en pixels
resolution_theta = np.pi / 180 # précision de l'angle : 1 degré
seuil_votes = 150              # nombre minimum de votes pour garder une ligne

lignes = cv.HoughLines(contours, resolution_rho, resolution_theta, seuil_votes)


# ---------------------------------------------------------------
# 6. Dessiner les lignes trouvées sur une copie de l'image
# ---------------------------------------------------------------
resultat = image.copy()

if lignes is not None:
    print(f"{len(lignes)} ligne(s) détectée(s).")
    for ligne in lignes:
        rho, theta = ligne[0]

        # Point de la droite le plus proche de l'origine (coin haut-gauche)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        # On s'éloigne de 2000 pixels de part et d'autre de ce point
        # le long de la droite, pour la tracer d'un bord à l'autre de l'image
        x1 = int(x0 + 2000 * (-b))
        y1 = int(y0 + 2000 * (a))
        x2 = int(x0 - 2000 * (-b))
        y2 = int(y0 - 2000 * (a))

        cv.line(resultat, (x1, y1), (x2, y2), (0, 0, 255), 2)  # rouge, épaisseur 2
else:
    print("Aucune ligne détectée. Essayez de baisser 'seuil_votes'.")


# ---------------------------------------------------------------
# 7. Afficher et sauvegarder les résultats
# ---------------------------------------------------------------
cv.imwrite("resultat_canny.png", contours)
cv.imwrite("resultat_hough.png", resultat)

cv.imshow("1 - Image originale", image)
cv.imshow("2 - Contours (Canny)", contours)
cv.imshow("3 - Lignes (Hough)", resultat)
print("Appuyez sur une touche dans une fenêtre d'image pour quitter.")
cv.waitKey(0)
cv.destroyAllWindows()