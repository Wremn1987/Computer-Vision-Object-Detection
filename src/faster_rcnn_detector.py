# src/faster_rcnn_detector.py

import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
import cv2
import numpy as np
import os

# This is a simplified example. In a real scenario, you would handle custom datasets and training.

def load_faster_rcnn_model():
    """Loads a pre-trained Faster R-CNN model with a ResNet50-FPN backbone."""
    # Use the COCO trained weights for demonstration
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    model.eval() # Set the model to evaluation mode
    print("Faster R-CNN model loaded successfully.")
    return model, weights.transforms()

def detect_objects_faster_rcnn(model, transform, image_path, threshold=0.7):
    """
    Performs object detection using the loaded Faster R-CNN model.

    Args:
        model: The pre-trained Faster R-CNN model.
        transform: The transformation function for the input image.
        image_path (str): Path to the input image.
        threshold (float): Confidence threshold for displaying detections.

    Returns:
        numpy.ndarray: Image with bounding boxes and labels drawn.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None

    # Convert image to RGB (PyTorch models expect RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = transform(image_rgb)

    with torch.no_grad():
        prediction = model([img_tensor])

    output_image = image.copy()
    # COCO dataset classes (example, adjust if using different weights)
    coco_names = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
        'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A',
        'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote',
        'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    for i in range(len(prediction[0]['labels'])):
        score = prediction[0]['scores'][i].item()
        if score > threshold:
            label = coco_names[prediction[0]['labels'][i].item()]
            box = prediction[0]['boxes'][i].tolist()
            x1, y1, x2, y2 = [int(b) for b in box]

            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    output_path = f"detected_frcnn_{os.path.basename(image_path)}"
    cv2.imwrite(output_path, output_image)
    print(f"Detection result saved to {output_path}")
    return output_image

if __name__ == '__main__':
    # Create a dummy image for testing
    dummy_image_path = "data/images/test_frcnn.jpg"
    os.makedirs(os.path.dirname(dummy_image_path), exist_ok=True)
    dummy_image = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.putText(dummy_image, "Sample FRCNN Image", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(dummy_image_path, dummy_image)

    model_frcnn, transform_frcnn = load_faster_rcnn_model()
    if model_frcnn:
        detect_objects_faster_rcnn(model_frcnn, transform_frcnn, dummy_image_path)
