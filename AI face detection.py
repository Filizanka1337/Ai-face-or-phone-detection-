import cv2

# Załaduj klasyfikator Haar Cascade do wykrywania twarzy
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Otwórz kamerę laptopa (domyślnie kamera o indeksie 0)
cap = cv2.VideoCapture(0)

while True:
    # Przechwyć klatkę z kamery
    ret, frame = cap.read()
    if not ret:
        break

    # Konwertuj obraz na odcienie szarości (Haar Cascade działa lepiej na szaro)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Wykryj twarze w obrazie
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Zaznacz twarze na obrazie
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Wyświetl obraz z wykrytymi twarzami
    cv2.imshow('AI Face Detection', frame)

    # Przerwij działanie programu, jeśli naciśniesz 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zwolnij zasoby i zamknij okna
cap.release()
cv2.destroyAllWindows()
