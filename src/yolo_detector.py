import cv2
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLODetector:
    """
    A class for performing object detection using the YOLOv3 model with OpenCV.
    This class handles loading the model, preprocessing images, running inference,
    and drawing bounding boxes on the detected objects.
    """
    def __init__(self, config_path, weights_path, classes_path, conf_threshold=0.5, nms_threshold=0.4):
        """
        Initializes the YOLO detector.
        
        Args:
            config_path (str): Path to the YOLO configuration file (.cfg).
            weights_path (str): Path to the YOLO weights file (.weights).
            classes_path (str): Path to the file containing class names (.names).
            conf_threshold (float): Confidence threshold for filtering weak detections.
            nms_threshold (float): Non-maximum suppression threshold for removing overlapping boxes.
        """
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        logger.info("Initializing YOLODetector...")
        
        # Load class names
        try:
            with open(classes_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            logger.info(f"Loaded {len(self.classes)} classes.")
        except FileNotFoundError:
            logger.error(f"Classes file not found at {classes_path}")
            raise
            
        # Generate random colors for each class
        np.random.seed(42)
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
        # Load YOLO network
        try:
            self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            logger.info("YOLO network loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLO network: {e}")
            raise
            
        # Get output layer names
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect(self, image):
        """
        Performs object detection on the given image.
        
        Args:
            image (numpy.ndarray): The input image.
            
        Returns:
            tuple: A tuple containing lists of class IDs, confidences, and bounding boxes.
        """
        height, width = image.shape[:2]
        
        # Create a blob from the image
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Run forward pass
        logger.debug("Running forward pass...")
        outputs = self.net.forward(self.output_layers)
        
        class_ids = []
        confidences = []
        boxes = []
        
        # Process outputs
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.conf_threshold:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    
        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        
        # Flatten indices if necessary (depends on OpenCV version)
        if len(indices) > 0:
            indices = indices.flatten()
        else:
            indices = []
            
        final_boxes = [boxes[i] for i in indices]
        final_confidences = [confidences[i] for i in indices]
        final_class_ids = [class_ids[i] for i in indices]
        
        logger.info(f"Detected {len(final_boxes)} objects.")
        return final_class_ids, final_confidences, final_boxes

    def draw_boxes(self, image, class_ids, confidences, boxes):
        """
        Draws bounding boxes and labels on the image.
        """
        result_image = image.copy()
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            label = str(self.classes[class_ids[i]])
            confidence = confidences[i]
            color = self.colors[class_ids[i]]
            
            # Draw rectangle
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            text = f"{label} {confidence:.2f}"
            cv2.putText(result_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        return result_image

if __name__ == "__main__":
    # Example usage (requires YOLO weights, config, and names files)
    # These files need to be downloaded separately for this to run
    print("YOLODetector class defined. To run, provide paths to YOLOv3 files.")
    # detector = YOLODetector('yolov3.cfg', 'yolov3.weights', 'coco.names')
    # image = cv2.imread('test.jpg')
    # class_ids, confidences, boxes = detector.detect(image)
    # result = detector.draw_boxes(image, class_ids, confidences, boxes)
    # cv2.imshow('Result', result)
    # cv2.waitKey(0)
