from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator
import matplotlib.pyplot as plt

# Charger le modèle YOLOv8
model = YOLO('yolov8n.pt')

# Charger l'image
img = cv2.imread('68747470733a2f2f756c7472616c79746963732e636f6d2f696d616765732f7a6964616e652e6a7067.jpeg')

# Redimensionner l'image à 720x1340

# Prédire les objets dans l'image
results = model.predict(img)

# Compter le nombre de personnes
num_people = 0

# Parcourir chaque résultat
for r in results:
    boxes = r.boxes
    for box in boxes:
        # Vérifier si la classe de l'objet est une personne
        if model.names[int(box.cls)] == 'person':
            num_people += 1
            # Dessiner la boîte englobante
            b = box.xyxy[0]  # obtenir les coordonnées de la boîte au format (haut, gauche, bas, droite)
            
            # Cropper l'image pour obtenir seulement la boîte englobante
            cropped_img = img[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
            cropped_img2 = cv2.resize(cropped_img, (3000, 3000))

            # Afficher l'image croppée
            plt.figure()
            plt.imshow(cv2.cvtColor(cropped_img2, cv2.COLOR_BGR2RGB))
            plt.title(f'Personne {num_people}')
            plt.show()

# Afficher le nombre de personnes
print(f'Nombre de personnes : {num_people}')
