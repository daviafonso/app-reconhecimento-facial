from flask import Flask, render_template, Response
import cv2
import dlib
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)

# Caminho para os arquivos do modelo de landmarks faciais e reconhecimento facial
predictor_path = "shape_predictor_68_face_landmarks.dat"
face_recognition_model_path = "dlib_face_recognition_resnet_model_v1.dat"

# Caminho para as imagens de referência
known_faces_folder = "imagem" 

# Inicializar o detector de rosto, o preditor de landmarks faciais e o modelo de reconhecimento facial
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
face_recognition_model = dlib.face_recognition_model_v1(face_recognition_model_path)

# Função para carregar as faces conhecidas
def load_known_faces(folder_path):
    known_face_encodings = []
    known_face_names = []
    
    for image_name in os.listdir(folder_path):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            for face in faces:
                landmarks = predictor(gray, face)
                face_descriptor = np.array(face_recognition_model.compute_face_descriptor(image, landmarks))
                known_face_encodings.append(face_descriptor)
                known_face_names.append(os.path.splitext(image_name)[0])
    
    return known_face_encodings, known_face_names

# Carregar as faces conhecidas
known_face_encodings, known_face_names = load_known_faces(known_faces_folder)

# Função para encontrar o melhor match
def find_best_match(face_descriptor, known_face_encodings):
    distances = [np.linalg.norm(face_descriptor - known_face) for known_face in known_face_encodings]
    min_distance_index = np.argmin(distances)
    return min_distance_index, distances[min_distance_index]

# Função para capturar a câmera e processar frames
def gen_frames():
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_frame)

        for face in faces:
            landmarks = predictor(gray_frame, face)
            face_descriptor = np.array(face_recognition_model.compute_face_descriptor(frame, landmarks))
            match_index, distance = find_best_match(face_descriptor, known_face_encodings)
            name = known_face_names[match_index] if distance < 0.6 else "Desconhecido"
            
            welcome_message = f"Bem-vindo, {name}!" if name != "Desconhecido" else "Desconhecido"
            
            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, welcome_message, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Rota para exibir o vídeo no navegador
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Rota principal
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
