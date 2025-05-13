import cv2
from deepface import DeepFace

# Inicializa a webcam
camera = cv2.VideoCapture(0)
print("Pressione 'q' para sair.")

while True:
    ret, frame = camera.read()
    if not ret:
        print("Erro ao acessar a câmera.")
        break

    try:
        # Analisa a emoção da face detectada
        resultado = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emocao = resultado[0]['dominant_emotion']

        # Mostra o resultado na tela
        cv2.putText(frame, f'Emocao: {emocao}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    except Exception as e:
        print("Erro:", e)

    # Exibe o vídeo com a emoção detectada
    cv2.imshow("Detector de Emoções", frame)

    # Encerra com 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a câmera e fecha as janelas
camera.release()
cv2.destroyAllWindows()
