import os
import cv2
import matplotlib.pyplot as plt
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
    
    def detect(self, img_path, conf_filter=0.3, ind_model_threshold=0.3, iou_threshold=0.5, skip_box_threshold=0.0001,conf_type='box_and_model_avg'):
        """Run detection on a single image."""
        if not os.path.exists(img_path):
            print(f"Error: Image file not found: {img_path}")
            return None, None, None
        
        # Run predictions
        boxes_list, labels_list, scores_list = predict_with_models(
            self.models, self.config_files, img_path, ind_model_threshold
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

'''
import torch
import torch.multiprocessing as mp

# Ensure the spawn start method is used (if not already set elsewhere)
mp.set_start_method('spawn', force=True)

def predict_one_model(args):
    """
    Helper function to run prediction for a single model.
    """
    i, model, config_file, img_path, threshold = args
    print(f"\nPredicting with model {i+1}")
    # Run the prediction using your predict function
    res = predict(loaded_model=model, config_file=config_file, image_path=img_path, threshold=threshold)
    
    # Move the results to CPU to avoid CUDA IPC issues
    boxes = res[0][0]
    scores = res[0][1].cpu().numpy().tolist()
    conf = res[0][2].cpu().numpy().tolist()
    return boxes, scores, conf

def predict_with_models(models, config_files, img_path, ind_model_threshold=0.3):
    """
    Run prediction with multiple models on a single image concurrently.

    Args:
        models (list): List of loaded models.
        config_files (list): List of configuration files corresponding to models.
        img_path (str): Path to the input image.
        ind_model_threshold (float): Threshold for individual model predictions.

    Returns:
        tuple: Lists of bounding boxes, class scores, and confidence scores.
    """
    final_bb = []
    final_scores = []
    final_conf = []
    
    # Build a list of tasks for each loaded model
    tasks = []
    for i, model in enumerate(models):
        if model is None:
            print(f"Skipping model {i+1} (not loaded)")
            continue
        tasks.append((i, model, config_files[i], img_path, ind_model_threshold))
    
    # Use a process pool to run predictions concurrently
    if tasks:
        with mp.Pool(processes=len(tasks)) as pool:
            results = pool.map(predict_one_model, tasks)
        
        # Unpack the results from each process
        for boxes, scores, conf in results:
            final_bb.append(boxes)
            final_scores.append(scores)
            final_conf.append(conf)
    
    return final_bb, final_scores, final_conf

'''

import concurrent.futures

def predict_with_models(models, config_files, img_path, ind_model_threshold=0.3):
    """
    Run prediction with multiple models on a single image concurrently using threads.
    
    Args:
        models (list): List of loaded models.
        config_files (list): List of configuration files corresponding to models.
        img_path (str): Path to the input image.
        ind_model_threshold (float): Threshold for individual model predictions.
        
    Returns:
        tuple: Lists of bounding boxes, class scores, and confidence scores.
    """
    final_bb = []
    final_scores = []
    final_conf = []
    
    def run_prediction(i, model, config_file):
        print(f"\nPredicting with model {i+1}")
        res = predict(loaded_model=model, config_file=config_file, image_path=img_path, threshold=ind_model_threshold)
        print(res)
        # Extract and move tensors to CPU as needed
        boxes = res[0][0]
        scores = res[0][1].cpu().numpy().tolist()
        conf = res[0][2].cpu().numpy().tolist()
        return (i, boxes, scores, conf)
    
    futures = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i, model in enumerate(models):
            if model is None:
                print(f"Skipping model {i+1} (not loaded)")
                continue
            futures.append(executor.submit(run_prediction, i, model, config_files[i]))
        
        # as_completed returns results in arbitrary order so we sort them by the model index.
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    results.sort(key=lambda x: x[0])
    for _, boxes, scores, conf in results:
        final_bb.append(boxes)
        final_scores.append(scores)
        final_conf.append(conf)
    
    return final_bb, final_scores, final_conf
