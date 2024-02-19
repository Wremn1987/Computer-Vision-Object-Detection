import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import torchvision.transforms as T
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format=\'%(asctime)s - %(levelname)s - %(message)s\')
logger = logging.getLogger(__name__)

class FasterRCNNDetector:
    """
    A class for performing object detection using the Faster R-CNN model
    with a ResNet50-FPN backbone from torchvision. This class handles
    loading a pre-trained model, fine-tuning (optional), and inference.
    """
    def __init__(self, num_classes=91, pretrained=True, device=None):
        """
        Initializes the Faster R-CNN detector.
        
        Args:
            num_classes (int): Number of output classes. Default is 91 for COCO.
            pretrained (bool): If True, loads a model pre-trained on COCO.
            device (str): The device to run the model on (‘cpu’ or ‘cuda’).
                          If None, it automatically selects ‘cuda’ if available.
        """
        self.device = device if device else (‘cuda’ if torch.cuda.is_available() else ‘cpu’)
        logger.info(f"Initializing FasterRCNNDetector on device: {self.device}")

        if pretrained:
            # Load a pre-trained model on COCO
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
            # Replace the classifier with a new one, that has num_classes which is user-defined
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            logger.info(f"Loaded pre-trained Faster R-CNN with {num_classes} classes.")
        else:
            # Create a model from scratch
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
            logger.info(f"Created Faster R-CNN model from scratch with {num_classes} classes.")

        self.model.to(self.device)
        self.transform = T.Compose([T.ToTensor()])

    def train_one_epoch(self, data_loader, optimizer, lr_scheduler=None):
        """
        Trains the model for one epoch.
        (Simplified for demonstration; a full training loop would be more complex)
        """
        self.model.train()
        logger.info("Starting training epoch...")
        for i, (images, targets) in enumerate(data_loader):
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()
            
            if i % 50 == 0:
                logger.info(f"Iteration {i}: Loss = {losses.item():.4f}")
        logger.info("Training epoch finished.")

    @torch.no_grad()
    def predict(self, image):
        """
        Performs inference on a single image.
        
        Args:
            image (PIL.Image or numpy.ndarray): The input image.
            
        Returns:
            dict: A dictionary containing ‘boxes’, ‘labels’, and ‘scores’.
        """
        self.model.eval()
        img_tensor = self.transform(image).to(self.device)
        prediction = self.model([img_tensor])
        
        # Move predictions to CPU for easier handling
        output = {
            ‘boxes’: prediction[0][‘boxes’].cpu().numpy(),
            ‘labels’: prediction[0][‘labels’].cpu().numpy(),
            ‘scores’: prediction[0][‘scores’].cpu().numpy()
        }
        logger.info(f"Detected {len(output[‘boxes’])} objects.")
        return output

    def save_model(self, path):
        """
        Saves the model’s state dictionary.
        """
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path):
        """
        Loads the model’s state dictionary.
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        logger.info(f"Model loaded from {path}")

if __name__ == "__main__":
    # Example usage (requires a custom dataset for training)
    print("FasterRCNNDetector class defined. To run, provide a dataset.")
    # detector = FasterRCNNDetector(num_classes=2) # e.g., 1 for background, 1 for custom object
    # # Dummy data for demonstration
    # class DummyDataset(torch.utils.data.Dataset):
    #     def __len__(self):
    #         return 10
    #     def __getitem__(self, idx):
    #         img = torch.rand(3, 300, 400) # Dummy image
    #         target = {
    #             ‘boxes’: torch.tensor([[50, 50, 150, 150]], dtype=torch.float32),
    #             ‘labels’: torch.tensor([1], dtype=torch.int64)
    #         }
    #         return img, target
    # 
    # data_loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=2, shuffle=True)
    # 
    # params = [p for p in detector.model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # 
    # detector.train_one_epoch(data_loader, optimizer)
    # 
    # # Example prediction
    # from PIL import Image
    # dummy_image = Image.new(‘RGB’, (400, 300), color = ‘red’)
    # predictions = detector.predict(dummy_image)
    # print(predictions)
