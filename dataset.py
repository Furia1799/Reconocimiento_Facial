# coding=utf-8
import cv2

web_cam = cv2.VideoCapture(0)  # abrir la webcaccap


# direccion de la ruta donde se encuentra el archivo //detectar rostros de enfrente
cascPath = "C:/Users/Bryan/Desktop/Proyecto_Final/Cascades/haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascPath)  # metodo de cv indentificar rostros

count = 0  # imagenes creadas

while(True):
    _, imagen_marco = web_cam.read()  # imagen del marco

    grises = cv2.cvtColor(imagen_marco, cv2.COLOR_BGR2GRAY)  # convertir a grises

    rostro = faceCascade.detectMultiScale(grises, 1.5, 5)  # detector de rostro

    for(x, y, w, h) in rostro:  # encerrar el rostro en  un cuadro
        cv2.rectangle(imagen_marco, (x, y), (x+w, y+h), (255, 0, 0), 4)  # pintar rostro cuadro
        count += 1

        cv2.imwrite("images/Bryan/Bryan_"+str(count)+".jpg",
                    grises[y:y+h, x:x+w])  # escala grises
        cv2.imshow("Creando Dataset", imagen_marco)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    elif count >= 400:  # numero de fotos creadas
        break

# Cuando todo está hecho, liberamos la captura
web_cam.release()
cv2.destroyAllWindows()
