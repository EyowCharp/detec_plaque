import sys
import cv2 as cv

img = cv.imread("img/Cascade.jpg")
if img is None:
    sys.exit(...)

cv.imshow("Cascade", img)

cv.waitKey(0)

cv.imwrite("Cascade_couleur.png", img)

img_gris = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imwrite("Cascade_gris.png", img_gris)