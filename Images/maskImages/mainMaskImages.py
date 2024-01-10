# %%
import numpy as np
import pandas as pd

import os
import sys

import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

local_path = os.path.dirname(os.path.abspath(__file__)) + "/"
image_path = local_path + "Images_input/"
print(local_path)

file_list = os.listdir(image_path)
image_list = []
image_list = [file for file in file_list if file.endswith(".jpg")]
image_list = image_list + [file for file in file_list if file.endswith(".png")]

# %% DEEPLAV_V3
if "deeplab" in sys.argv:
        
    # Load the pre-trained DeepLabV3 model with a ResNet backbone
    # model = models.segmentation.deeplabv3_resnet101(weights = models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)
    model = models.segmentation.deeplabv3_resnet50(weights = models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)

    model.eval()


    class_labels = models.segmentation.deeplabv3._VOC_CATEGORIES
    print(class_labels)

    # Define the image transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    for image in image_list:
            
        input_image = Image.open(image_path + image).convert("RGB")
        input_tensor = transform(input_image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)['out'][0]
            
        # Get the mask prediction
        mask_predictions = output.argmax(0)

        # Retrieve labels for each unique value in the mask
        masked_object_labels = [class_labels[val] for val in torch.unique(mask_predictions) if val != 0]
        print(torch.unique(mask_predictions))
        # Plot the original image and mask
        plt.subplot(1, 2, 1)
        plt.imshow(input_image)
        plt.title('Original Image')

        plt.subplot(1, 2, 2)
        plt.imshow(mask_predictions, cmap='viridis')
        plt.title(f"Mask Prediction: {masked_object_labels}")
        plt.colorbar()

        plt.show()

# %% fasterrcnn_resnet50_fpn
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

if "fastercnn" in sys.argv:
    # Load pre-trained Faster R-CNN model with a ResNet;-50 backbone
    fasterrcnn_resnet50_fpn = models.detection.fasterrcnn_resnet50_fpn(weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    # fasterrcnn_resnet50_fpn = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    weights = FCN_ResNet50_Weights.DEFAULT
    transform = weights.transforms(resize_size=None)
    model = fcn_resnet50(weights=weights, progress=False)

    fasterrcnn_resnet50_fpn.eval()
    model = model.eval()

    class_labels = models.detection.faster_rcnn._COCO_CATEGORIES
    class_labels_resnet50 = models.segmentation.fcn._VOC_CATEGORIES

    print(class_labels)

    for image in image_list:

        input_image = Image.open(image_path + image).convert("RGB")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = transform(input_image).unsqueeze(0)

        print(input_tensor.size())

        # Make the prediction to get region proposals
        with torch.no_grad():
            predictions = fasterrcnn_resnet50_fpn(input_tensor)
            output = model(input_tensor)['out']

        sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
        normalized_masks = torch.nn.functional.softmax(output, dim=1)
        print(normalized_masks.shape)

        plt.subplots(1, 2)
        plt.subplot(1, 2, 1)
        plt.plot(normalized_masks[0,:,0,0])
        plt.imshow(normalized_masks[0,0,:,:])

        # Extract bounding boxes and scores
        boxes = predictions[0]['boxes'].numpy().astype(int)
        scores = predictions[0]['scores'].numpy()
        labels = predictions[0]['labels'].numpy()

        # Threshold for considering a region as a valid detection
        threshold = 0.5
        valid_boxes = boxes#[scores > threshold]

        print(predictions)
        max_pred_idx = np.argmax(scores)
        max_pred_label = labels[max_pred_idx]
        max_pred_class = class_labels[max_pred_label]

        pred_class = []
        # for pred_idx, pred in enumerate(scores):
        #     pred_label = labels[pred_idx]
        #     pred_class.append(class_labels[max_pred_label])

        # Draw rectangles on the mask
        plt.subplot(1, 2, 2)
        mask = np.zeros((input_image.size[1], input_image.size[0]))
        print(valid_boxes)
        for box_idx, box in enumerate(valid_boxes):
            mask[box[1]:box[3], box[0]:box[2]] = 1

        plt.imshow(input_image)
        plt.imshow(mask, cmap="Accent", alpha=0.5, vmin=0, vmax=1)
        plt.title(f"Most likely pred : {max_pred_class}")
        plt.suptitle(f"{pred_class}")
        plt.show()

        # break


class MaskImages():
    """
    Takes a numpy array of shape (Height, With, Channels) as input.
    Returns a 4x4 array where:
        - Null cells mean no object was detected
        - Non-null cells means the object of maximum likelyhood appears in this cell
    Multiple non-null cells may feature the target, if the object is large enough, or if it present multiple times.

    Attributes:
        - image : the image as a numpy array (H, W, C)
        - roi_cells : the matrix of Regions Of Interest
    """

    def __init__(self, image_array) -> None:
        self.image = np.array(image_array, dtype=np.uint8)
        self.height = image_array.shape[0]
        self.width = image_array.shape[1]
        self.channels = image_array.shape[2]

    def runPipeline(self):
        """
        Runs all the necessary steps to return the mask of objects of an input array.
        Returns the RoI 4x4 array.
        """
        pipeline = [
            self.loadModelWeigths(),
            self.composeTensor(),
            self.evaluateTensor(),
            self.getROICells()
        ]
        for step in pipeline:
            step

        return self.roi_cells
    
    def loadModelWeigths(self):

        self.model = models.segmentation.deeplabv3_resnet50(weights = models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
        self.model.eval()
        class_labels = models.segmentation.deeplabv3._VOC_CATEGORIES
        # print(class_labels)
        return self.model

    def composeTensor(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if self.channels == 1:
            self.image = Image.fromarray(self.image).convert("RGB")
        # self.tensor = torch.from_numpy(self.image)
        self.tensor = transform(self.image).unsqueeze(0)
        print(self.tensor.size())

        return self.tensor

    def evaluateTensor(self):
        with torch.no_grad():
            output = self.model(self.tensor)['out'][0]
            
        self.mask_predictions = output.argmax(0)
        self.pred_category = self.mask_predictions.numpy()[-1]

        self.masked_object_labels = [self.class_labels[val] for val in torch.unique(self.mask_predictions) if val != 0]

    def getROICells(self):

        self.roi_cells = np.zeros((4, 4))

        for vertical_roi in range(4):
            for horizontal_roi in range(4):
                bottom_roi_coo, top_roi_coo = vertical_roi/4*self.height, (vertical_roi+1)/4*self.height
                left_roi_coo, right_roi_coo = horizontal_roi/4*self.width, (horizontal_roi+1)/4*self.width

                roi_mask = self.mask_predictions[round(bottom_roi_coo):round(top_roi_coo), round(left_roi_coo):round(right_roi_coo)].numpy()

                if np.all(roi_mask != 0):
                    self.roi_cells[vertical_roi, horizontal_roi] = 1

        return self.roi_cells

    def plotMask(self):
        plt.subplot(1, 2, 1)
        plt.imshow(self.image)
        plt.title('Original Image')

        plt.subplot(1, 2, 2)
        plt.imshow(self.mask_predictions, cmap='viridis')
        plt.title(f"Mask Prediction: {self.masked_object_labels}")
        plt.colorbar()

        plt.show()

if __name__ == "__main__":
    image_array = np.ones((200, 300, 3))*100 
    image_array = np.random.randint(0, 255, size=(200, 200, 3))

    class_obj = MaskImages(image_array=image_array)
    roi_cells = class_obj.runPipeline()
    class_obj.plotMask()
    print(roi_cells)


