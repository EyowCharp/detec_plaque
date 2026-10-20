import cv2
import numpy as np

# Crée une image noire de 300x500 pixels avec 3 canaux de couleur
image = np.zeros((300, 500, 3), dtype=np.uint8)

# Dessine un texte et un cercle (OpenCV utilise l'ordre BGR, pas RGB)
cv2.putText(image, "OpenCV fonctionne !", (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.circle(image, (250, 230), 40, (0, 0, 255), -1)

cv2.imshow("Test", image)
cv2.waitKey(0)            # attend l'appui sur une touche
cv2.destroyAllWindows()