import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2

image1 = Image.open("voiture.png").convert('RGB')
image2 = Image.open("Dog.jpg").convert('RGB')
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
    
            with open("Categories.txt", "r") as f:
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
        dico["Bicycle"] = ["bicycle-built-for-two", "mountain bike", "tricycle"]
        dico["Bridge"] = ["suspension bridge", "steel arch bridge", "viaduct"]
        dico["Bus"] = ["school bus", "trolleybus"]
        dico["Car"] = ["convertible", "sports car", "limousine", "racer", "minivan", "golfcart", "mobile home", "pickup", "cab", "car mirror"]
        dico["Chimney"] = ["fireplace", "smokestack"]
        dico["Crosswalk"] = ["pedestrian crossing", "zebra crossing"]
        dico["Hydrant"] = ["fire hydrant"]
        dico["Motorcycle"] = ["motorcycle"]
        dico["Mountain"] = ["mountain", "peak", "summit"]
        dico["Palm"] = ["palm tree", "date palm"]
        dico["Traffic Light"] = ["traffic light"]
        return dico   
        
#%%

solveur = solve_capcha_9_images(input_images, "Car")

print(solveur.categories)

print(solveur.images_to_mark)
        
        
        
        
        
        
        
        
        
        

