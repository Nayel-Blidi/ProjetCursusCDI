from Images.maskImages.mainMaskImages import MaskImages
from Images.multiImages.mainMultiImages import solve_capcha_9_images 

from PIL import Image
import numpy as np
import cv2
import os


# CAS D'UTILISATION : ENIGME 4x4 (maskImage)
# - créer un objet de classe mask avec en input une np array de format cv2 (height, width, channels)
# class_obj = MaskImages(image_array=np.array)
# - récupérer la prédiction (matrice de positions) en output de runPipeline()
# roi_cells = class_obj.runPipeline()
# - roi_cells est une matrice 4x4 avec 1 pour les cellules à cliquer, et 0 sinon.

# CAS D'UTILISATION : ENIGME multi images (9 images)
# - convertir la liste en objet PIL.Image
# list_images = [Image.fromarray(image, mode="RGB") for image in list_images]
# - créer un objet solveur avec en input la liste des images PIL et le string de la catégorie cherchée 
# solveur = solve_capcha_9_images(list_images=list, target=str)
# - récupérer la variable images_to_mark qui indique l'index de list_images comprenant une image à sélectionner.
# solveur.images_to_mark

def main():

    return 0


if __name__ == "__main__":
    main()