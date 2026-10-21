# TP4 — Détection de plaques d'immatriculation et lecture des caractères (R5.16)

OpenCV + Python uniquement, aucune bibliothèque d'OCR : chaque étape vient du cours.

## Lancer
```
python lecture.py images/voitures/voiture_01.jpg     # chaîne complète, affiche les étapes
python test_lecture.py                               # teste les 3 photos, compare avant / après
```
Touche quelconque dans une fenêtre pour quitter.

## Chaîne de traitement
| Partie | Étapes | Fichier |
|---|---|---|
| A — trouver la plaque | 1 charger/redimensionner · 2 gris · 3 filtre bilatéral · 4 Canny · 5 findContours · 6 candidats (minAreaRect, ratio) · redressement (warpPerspective) · choix par nombre de caractères | `lucas_partieA.py` |
| B — isoler les caractères | 7 Otsu inversé + floodFill (efface le cadre) · 8 findContours (RETR_EXTERNAL) + tri par x + filtre de hauteur · 9 normalisation 30×60 | `lecture.py`, `normalisation.py` |
| C — reconnaître | 10 comparaison aux 33 modèles (IoU) · 11 format AA-123-AA (position -> alphabet autorisé, sans I, O, U) · 12 affichage | `lecture.py` |

Modèles : `creer_modeles_ratio.py` dessine les 33 caractères (police Liberation Sans Bold, équivalent libre d’Arial, proche des plaques
« bâtons ») avec la MÊME normalisation que les caractères lus -> `modeles_ratio/`.

## Résultats (3 photos de test : voiture_01, 02, 03 — 21 caractères)
| Méthode | Caractères justes |
|---|---|
| Caractère étiré à 30×60 + pixels identiques (première version) | 16/21 (76 %) |
| Caractère étiré + IoU | 17/21 (81 %) |
| **Proportions conservées + IoU** (version finale) | **19/21 (90 %)** |

Pourquoi : étirer un « 1 » étroit sur 30 px le déforme (lu « 7 »). Garder les proportions règle ça.
L'IoU ne compte que le tracé des caractères, alors que « pixels identiques » est gonflé par le fond noir commun.

## Limites (à assumer à l'oral)
- Seulement 3 photos : 90 % n'est pas une mesure fiable, et la méthode finale a été choisie en comparant
  sur ces mêmes photos. À valider sur d'autres images.
- Erreurs restantes (voiture_02) : 6 lu 8, W lu V — caractères proches, police Liberation Sans (équivalent d’Arial) et non celle des plaques ; son « 1 » a un pied.
- Redressement par `minAreaRect` (rectangle tourné) : corrige l'inclinaison mais pas une vraie perspective.
- Les bandes bleues sont écartées par leur hauteur (plus basses que les caractères) ; une plaque sale ou
  de nuit peut faire échouer Otsu.
- Pistes : modèles tirés de vraies plaques (`modeles_reels/`, sources dans `SOURCES.md`, recadrés en étirant :
  à refaire avec proportions conservées), seuillage local, vraie transformation de perspective (approxPolyDP à 4 sommets).
