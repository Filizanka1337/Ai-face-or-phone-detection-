import cv2
import numpy as np

# Funkcja do wykrywania niebieskiego iPhone'a
def detect_blue_iphone(frame):
    # Zakresy kolorów dla niebieskiego koloru (HSV)
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    # Konwersja obrazu na przestrzeń kolorów HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Tworzenie maski dla koloru niebieskiego
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Znajdowanie konturów na podstawie maski
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Rysowanie konturów i prostokątów wokół wykrytych obiektów
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filtruj małe kontury
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame

# Otwórz kamerę komputerową
cap = cv2.VideoCapture(0)

while True:
    # Przechwytywanie klatki z kamerki
    ret, frame = cap.read()

    if not ret:
        print("Nie można uzyskać obrazu z kamery.")
        break

    # Wykrywanie niebieskiego iPhone'a
    result_frame = detect_blue_iphone(frame)

    # Wyświetlanie wyniku
    cv2.imshow('Detected Blue iPhone', result_frame)

    # Zakończ działanie po naciśnięciu klawisza 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zwolnienie kamery i zamknięcie okien
cap.release()
cv2.destroyAllWindows()
