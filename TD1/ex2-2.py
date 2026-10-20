import sys
import cv2 as cv

cap = cv.VideoCapture("vid/chevaux.mp4")
if not cap.isOpened():
    sys.exit(...)

fps = cap.get(cv.CAP_PROP_FPS)

if fps > 0:
    delai = int(1000 / fps) 
else:
    delai = 40

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_gris = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    cv.imshow("Vidéo", frame_gris)

    touche = cv.waitKey(delai) & 0xFF

    if touche == ord('q'):
        break

cap.release()
cv.destroyAllWindows()