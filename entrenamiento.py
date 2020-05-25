import cv2
import os
import numpy as np
from PIL import Image
import pickle

cascPath = "C:/Users/Bryan/Desktop/Proyecto_Final/Cascades/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

reconocimiento = cv2.face.LBPHFaceRecognizer_create()  # metodo de lbp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

# declarando  etiquetas
current_id = 0
etiquetas_id = {}
y_etiquetas = []
x_entrenamiento = []

for root, dirs, archivos in os.walk(image_dir):
    for archivo in archivos:
        if archivo.endswith("png") or archivo.endswith("jpg"):
            pathImagen = os.path.join(root, archivo)
            etiqueta = os.path.basename(root).replace(" ", "-")  # .lower()

            # creando las etiquetas
            if not etiqueta in etiquetas_id:
                etiquetas_id[etiqueta] = current_id
                current_id += 1
            id_ = etiquetas_id[etiqueta]

            pil_image = Image.open(pathImagen).convert("L")
            tamanio = (550, 550)
            imagenFinal = pil_image.resize(tamanio, Image.ANTIALIAS)
            image_array = np.array(pil_image, "uint8")

            rostros = faceCascade.detectMultiScale(image_array, 1.5, 5)  # reconocer rostros
            for (x, y, w, h) in rostros:
                roi = image_array[y:y+h, x:x+w]
                x_entrenamiento.append(roi)
                y_etiquetas.append(id_)


with open("labels.pickle", 'wb') as f:
    pickle.dump(etiquetas_id, f)

reconocimiento.train(x_entrenamiento, np.array(y_etiquetas))
reconocimiento.save("entrenamiento.yml")  # entrenamiento
