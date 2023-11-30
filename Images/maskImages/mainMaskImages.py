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

    # print(class_labels)

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
        plt.plot(normalized_masks[0,:,0,0])
        plt.imshow(normalized_masks[0,0,:,:])
        plt.show()

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
        plt.figure()
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




