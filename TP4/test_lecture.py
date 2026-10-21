"""
Teste la chaîne complète sur les photos de voitures et compare AVANT / APRÈS.
Usage : python test_lecture.py
"""
import os
import cv2 as cv
import lecture as L
import normalisation as N

ATTENDU = {"voiture_01": "EE-781-RB", "voiture_02": "FK-369-WE", "voiture_03": "GE-410-PC"}
os.makedirs(os.path.join(L.DOSSIER, "sorties"), exist_ok=True)

CONFIGS = [
    ("AVANT : caractère étiré + pixels identiques", "modeles",       N.normaliser_etire, L.score_pixels),
    ("étiré + IoU",                                  "modeles",       N.normaliser_etire, L.score),
    ("APRÈS : proportions conservées + IoU",         "modeles_ratio", N.normaliser_ratio, L.score),
]
for titre, jeu, norm, sc in CONFIGS:
    modeles = L.charger_modeles(jeu)
    bons = total = 0
    print(f"=== {titre} ===")
    for nom, attendu in ATTENDU.items():
        texte, image, info, binaire, details = L.traiter_image(
            os.path.join(L.DOSSIER, "images", "voitures", nom + ".jpg"), modeles, norm, sc)
        lu = texte or "(échec)"
        n = sum(a == b for a, b in zip(lu.replace("-", ""), attendu.replace("-", ""))) if texte else 0
        bons += n; total += 7
        print(f"  {nom}: attendu {attendu} | lu {lu} | {n}/7")
        if texte and sc is L.score and norm is N.normaliser_ratio:
            rects, scores, plaque = details
            cv.imwrite(os.path.join(L.DOSSIER, "sorties", f"{nom}_resultat.png"),
                       L.dessiner_resultat(image, info[0], info[1], texte))
            cv.imwrite(os.path.join(L.DOSSIER, "sorties", f"{nom}_caracteres.png"),
                       L.dessiner_caracteres(binaire, rects, texte))
    print(f"  -> {bons}/{total} caractères justes ({bons / total:.0%})\n")
