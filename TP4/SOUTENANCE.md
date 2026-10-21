# Soutenance TP4 — 10 min (22/10) — plaques d'immatriculation

Répartition conseillée : Lucas = partie A (étapes 1-7), toi = parties B et C (étapes 8-12) + démo.
Chacun doit pouvoir répondre sur tout le programme.

## 1 min — Contexte
« On a fait un lecteur de plaques d'immatriculation françaises en OpenCV et Python, sans bibliothèque d'OCR.
Le but : trouver la plaque dans une photo de voiture, puis lire ses 7 caractères.
On a choisi de tout faire avec des méthodes du cours pour pouvoir expliquer chaque étape (cf. CM1 : explicabilité). »

## 7 min — Méthode (3 parties)
### Partie A — trouver la plaque (Lucas, ~3 min)
| Étape | À dire | Fonction |
|---|---|---|
| 1 | On ramène toutes les images à 800 px de large : les réglages en pixels valent pour toutes. | imread, resize |
| 2-3 | Gris puis filtre bilatéral : il lisse le grain mais garde les bords nets de la plaque (un flou gaussien les rendrait mous). | cvtColor, bilateralFilter |
| 4 | Canny : gradient puis hystérésis (seuil bas 30, haut 200) -> image de bords. | Canny |
| 5 | On récupère toutes les formes (des centaines). | findContours |
| 6 | On garde les formes dont le rectangle englobant a un rapport longueur/hauteur de 2 à 6,5 (plaque : 520/110 = 4,7) et une taille plausible. | minAreaRect |
| 6 | Redressement : on envoie les 4 coins sur un rectangle 520x110 -> toutes les plaques ont la même taille et sont droites. | getPerspectiveTransform, warpPerspective |
| 6c | Pour choisir parmi les candidats, on garde celui qui contient le plus de caractères. | compter_caracteres |
| 7 | Otsu inversé (caractères blancs sur noir), puis floodFill depuis le bord : efface le cadre de la plaque sans toucher aux caractères. | threshold OTSU, floodFill |

### Partie B — isoler les caractères (toi, ~2 min)
| Étape | À dire | Fonction |
|---|---|---|
| 8 | findContours en RETR_EXTERNAL (sinon le trou du 8 compterait). Filtre de Lucas, puis filtre sur la hauteur : tous les vrais caractères ont la même hauteur à 15 % près, ce qui écarte les bandes bleues. Tri par x : lecture de gauche à droite. On exige 7 caractères, sinon on ne devine pas. | findContours, boundingRect |
| 9 | On découpe chaque caractère (slicing) et on le met à 30x60 **en gardant les proportions**, centré. | resize, slicing |

### Partie C — reconnaître (toi, ~2 min)
| Étape | À dire | Fonction |
|---|---|---|
| 10 | On a 33 modèles (10 chiffres + 23 lettres), fabriqués avec la même normalisation. On compare chaque caractère à chaque modèle avec l'IoU (intersection / union des pixels blancs) et on garde le plus proche. | logical_and / logical_or |
| 11 | Format AA-123-AA : à la position 3 on ne compare qu'aux chiffres, aux autres qu'aux lettres, et sans I, O, U (jamais utilisés). Une confusion 0/O devient impossible. | alphabet par position |
| 12 | On redessine le cadre sur la photo d'origine (coordonnées divisées par l'échelle) et on écrit le texte. | polylines, putText |

**Résultats (le dire avec honnêteté)**
| Méthode | Caractères justes sur 21 (3 photos) |
|---|---|
| Caractère étiré + pixels identiques | 16 (76 %) |
| Caractère étiré + IoU | 17 (81 %) |
| Proportions conservées + IoU | 19 (90 %) |
« Étirer un 1 étroit sur 30 px le déformait : il était lu 7. »

## 2 min — Démo
```
python lecture.py images/voitures/voiture_01.jpg     # lecture parfaite
python lecture.py images/voitures/voiture_02.jpg     # 2 erreurs : 6 lu 8, W lu V
```
Montrer les 3 fenêtres. Sur voiture_02, expliquer l'erreur plutôt que la cacher.

## Limites à citer
- 3 photos seulement : 90 % n'est pas une mesure fiable, et la méthode finale a été choisie sur ces mêmes photos.
- Modèles en Liberation Sans Bold (équivalent libre d’Arial), pas la police des plaques (le « 1 » a un pied) : scores assez bas (41 à 83 %).
- Redressement par minAreaRect (rectangle tourné), pas une vraie perspective.
- Plaque sale, de nuit, très inclinée ou 2 lignes : non gérées.
- Pistes : modèles tirés de vraies plaques, seuillage local, approxPolyDP à 4 sommets.

## Questions probables du prof
- **Pourquoi pas Tesseract ?** On voulait pouvoir expliquer chaque étape (explicabilité, CM1).
- **Pourquoi le filtre bilatéral ?** Il préserve les bords, contrairement au flou gaussien.
- **Que fait Otsu ?** Il choisit le seuil qui maximise la variance inter-classes, depuis l'histogramme.
- **Pourquoi l'hystérésis ?** Deux seuils : bords sûrs, et bords faibles gardés seulement s'ils touchent un bord sûr.
- **Pourquoi floodFill ?** Effacer ce qui touche le bord (cadre) sans effacer les caractères.
- **Pourquoi RETR_EXTERNAL ?** Ne pas compter les trous des 8, A, 0, etc.
- **Pourquoi des proportions conservées ?** Le « 1 » étiré ressemblait à un 7.
- **Comment améliorer ?** Voir les pistes ci-dessus.
