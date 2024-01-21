import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import os

if __name__ == "__main__":

    image1 = Image.open("Bicycle_1.png").convert('RGB')
    image2 = Image.open("dog.jpg").convert('RGB')
    input_images = [image1, image2]

#%%

class solve_capcha_9_images:
    
    def __init__(self, list_images, target):
        
        self.dico = self.create_dico_categories()
        self.target = target
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.input_batches = self.preprocess_images(list_images)
        self.model = self.load_model()
        self.categories = self.evaluate(self.input_batches, self.model)
        self.images_to_mark = self.get_images_to_mark()
        
    def load_model(self):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        return model

    def preprocess_images(self, list_images):
        images = []
        for image in list_images:
            images.append(self.preprocess(image).unsqueeze(0))
        return images
    
    
    def evaluate(self, input_batches, model):
        self.model.eval()
        evaluations = []
        for batch in input_batches:
            
            with torch.no_grad():
                output = self.model(batch)
                
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Categories.txt"), "r") as f:
                categories = [s.strip() for s in f.readlines()]
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            results = []
            for i in range(top5_prob.size(0)):
                for key, values in self.dico.items():
                    if categories[top5_catid[i]] in values:
                        results.append(key)
                        
            if results == []:
                results.append(categories[top5_catid[0]])
            evaluations.append(np.unique(results)[0])
        return evaluations
    
    
    def get_images_to_mark(self):
        img_to_mark = []
        for i, cat in enumerate(self.categories):
            if cat == self.target:
                img_to_mark.append(i)
        return img_to_mark
        
    def create_dico_categories(self):
        dico = {}
        dico["bicycle"] = ["bicycle-built-for-two", "mountain bike", "tricycle"]
        dico["bridge"] = ["suspension bridge", "steel arch bridge", "viaduct"]
        dico["bus"] = ["school bus", "trolleybus"]
        dico["car"] = ["convertible", "cars", "sports car", "limousine", "racer", "minivan", "golfcart", "mobile home", "pickup", "cab", "car mirror"]
        dico["chimney"] = ["fireplace", "smokestack"]
        dico["crosswalk"] = ["pedestrian crossing", "zebra crossing", "crosswalks"]
        dico["hydrant"] = ["fire hydrant"]
        dico["motorcycle"] = ["motorcycle"]
        dico["mountain"] = ["mountain", "peak", "summit"]
        dico["palm"] = ["palm tree", "date palm"]
        dico["traffic light"] = ["traffic light"]
        return dico   
        
#%%
    

if __name__ == "__main__":
    solveur = solve_capcha_9_images(input_images, "Mountain")
    
    print(solveur.categories)

    print(solveur.images_to_mark)
        
        
        
        
        
        
        
        
        
        

