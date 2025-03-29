import os
from ensemble_boxes import weighted_boxes_fusion
from detect4 import load_model, predict
from ssd.default import class_names_defined

class WeaponDetector:
    def __init__(self, config_files, checkpoint_files, model_weights):
        """
        Initialize the WeaponDetector with models and weights.
        
        Args:
            config_files (list): List of model configuration files
            checkpoint_files (list): List of model checkpoint files
            model_weights (list): List of weights for each model
        """
        self.config_files = config_files
        self.model_weights = model_weights
        self.models = self._load_models(config_files, checkpoint_files)
    
    def _load_models(self, config_files, checkpoint_files):
        """Load SSD models (internal method)."""
        models = []
        for i, (config_file, checkpoint_file) in enumerate(zip(config_files, checkpoint_files)):
            try:
                print(f"Loading model {i+1}: {os.path.basename(checkpoint_file)}")
                model = load_model(config_file=config_file, checkpoint_file=checkpoint_file)
                models.append(model)
            except Exception as e:
                print(f"Error loading model {i+1}: {e}")
                models.append(None)
        return models
    
    def detect(
        self, 
        img_path: str, 
        ind_model_threshold: list, 
        conf_scaling_factor: list, 
        conf_filter=0.3, 
        iou_threshold=0.5, 
        skip_box_threshold=0.0001, 
        conf_type='box_and_model_avg'
        ):
        """Run detection on a single image."""
        if not os.path.exists(img_path):
            print(f"Error: Image file not found: {img_path}")
            return None, None, None
        
        # Run predictions
        boxes_list, labels_list, scores_list = predict_with_models(
            self.models, self.config_files, img_path, ind_model_threshold, conf_scaling_factor
        )
        
        # Apply ensemble
        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list=boxes_list,
            scores_list=scores_list,
            labels_list=labels_list,
            weights=self.model_weights,
            iou_thr=iou_threshold,
            skip_box_thr=skip_box_threshold,
            conf_type=conf_type
        )
                
        # Filter out fused detections with low confidence.
        indices = scores >= conf_filter
        boxes = boxes[indices]
        scores = scores[indices]
        labels = labels[indices]
        
        print("\nDetection Results:")
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            labelInt = int(label)
            class_name = class_names_defined.get(labelInt, f"Class {labelInt}")
            print(f"Detection {i+1}: {class_name} (Confidence: {score:.4f})")
            print(f"Bounding Box: {box}")
        
        print(labels)
        return boxes, scores, labels

def predict_with_models(models, config_files, img_path, ind_model_threshold: list, conf_scaling_factor: list):
    """
    Run prediction with multiple models on a single image.
    
    Args:
        models (list): List of loaded models
        config_files (list): List of configuration files corresponding to models
        img_path (str): Path to the input image
        output_dir (str): Directory to save output predictions
        
    Returns:
        tuple: Lists of bounding boxes, class scores, and confidence scores
    """
    final_bb = []
    final_labels = []
    final_conf = []
    
    for i, model in enumerate(models):
        if model is None:
            print(f"Skipping model {i+1} (not loaded)")
            continue
        
        print(f"\nPredicting with model {i+1}")   
        res = predict(loaded_model=model, config_file=config_files[i], image_path=img_path, threshold=ind_model_threshold[i])
        print(res)
        
        final_bb.append(res[0][0])
        final_labels.append(res[0][1].cpu().numpy().tolist())
        # Scale confidence scores
        resConfAfterScalingFactor = res[0][2].cpu().numpy().tolist() * conf_scaling_factor[i]
        final_conf.append(resConfAfterScalingFactor)
    
    return final_bb, final_labels, final_conf


# # Example usage
# if __name__ == "__main__":
#     config_files = [
#         "/home/atharvaj/Desktop/SSD/configs/efficient_net_b3_ssd300_voc0712.yaml",
#         "/home/atharvaj/Desktop/SSD/configs/mobilenet_v3_ssd320_voc0712.yaml",
#         "/home/atharvaj/Desktop/SSD/configs/vgg_ssd300_voc0712.yaml",
#         "/home/atharvaj/Desktop/SSD/configs/resnet_50_ssd300_voc0712.yaml"
#     ]
    
#     checkpoint_files = [
#         "/home/atharvaj/Desktop/SSD/models_pth/ssdEff-v1.pth",
#         "/home/atharvaj/Desktop/SSD/models_pth/ssdMob-v1.pth",
#         "/home/atharvaj/Desktop/SSD/models_pth/vgg_model_3000.pth",
#         "/home/atharvaj/Desktop/SSD/models_pth/resnet50-v1.pth"
#     ]
    
#     model_weights = [0.4915, 0.6202, 0.6111, 0.6858]
    
#     # Initialize detector once
#     detector = WeaponDetector(config_files, checkpoint_files, model_weights)
    
#     # Run detection on multiple images
#     image_paths = [
#         # "/home/atharvaj/Desktop/SSD/ip_images/blackman-gun.jpg",
#         # "/home/atharvaj/Desktop/SSD/ip_images/gun-knife-2.png",
#         "/home/atharvaj/Desktop/SSD/ip_images/L7NCZHHQ590R_jpg.rf.43725c70f4a09fd1181ee15cb1be3914.jpg"
#     ]
#     from save_results import save_results
#     for img_path in image_paths:
#         print(f"\n\nRunning detection on image: {img_path}")
#         boxes, scores, labels = detector.detect(
#             img_path=img_path,
#             conf_filter=0.2,
#             iou_threshold=0.6,
#             ind_model_threshold=0.3
#         )
#         save_results(
#             img_path, 
#             boxes, 
#             scores, 
#             labels, 
#             output_dir="/home/atharvaj/Desktop/SSD/op_images",
#         )
