#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Weapon Detection System using Ensemble of SSD Models
This module provides functions for weapon detection using multiple SSD models.
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from ensemble_boxes import weighted_boxes_fusion
from detect_code.detect3 import load_model, predict
from ssd.default import class_names_defined

def load_models(config_files, checkpoint_files):
    """
    Load SSD models with the given configurations and checkpoint files.
    
    Args:
        config_files (list): List of paths to model configuration YAML files
        checkpoint_files (list): List of paths to model checkpoint/weights files
        
    Returns:
        list: List of loaded models
    """
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

def predict_with_models(models, config_files, img_path, output_dir):
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
    final_scores = []
    final_conf = []
    
    for i, model in enumerate(models):
        if model is None:
            print(f"Skipping model {i+1} (not loaded)")
            continue
        
        print(f"Predicting with model {i+1}")   
        res = predict(model, config_files[i], img_path, output_dir)
        
        final_bb.append(res[0])
        final_scores.append(res[1].cpu().numpy().tolist())
        final_conf.append(res[2].cpu().numpy().tolist())
    
    return final_bb, final_scores, final_conf

def apply_ensemble(boxes_list, scores_list, labels_list, weights, iou_thr=0.5, skip_box_thr=0.0001, conf_type='box_and_model_avg'):
    """
    Apply weighted boxes fusion to ensemble predictions from multiple models.
    
    Args:
        boxes_list (list): List of bounding boxes from each model
        scores_list (list): List of confidence scores from each model
        labels_list (list): List of class labels from each model
        weights (list): Weights for each model
        iou_thr (float, optional): IoU threshold for box fusion. Default is 0.5.
        skip_box_thr (float, optional): Threshold to skip low-confidence boxes. Default is 0.0001.
        conf_type (str, optional): Confidence type for weighted boxes fusion. Default is 'box_and_model_avg'.
        
    Returns:
        tuple: Fused boxes, scores, and labels
    """
    return weighted_boxes_fusion(
        boxes_list=boxes_list,
        scores_list=scores_list, 
        labels_list=labels_list, 
        weights=weights,
        iou_thr=iou_thr, 
        skip_box_thr=skip_box_thr,
        conf_type=conf_type
    )

def visualize_results(img_path, boxes, scores, labels, output_dir=None, save=False):
    """
    Visualize detection results on the image.
    
    Args:
        img_path (str): Path to the input image
        boxes (numpy.ndarray): Array of normalized bounding boxes [x1, y1, x2, y2]
        scores (numpy.ndarray): Array of confidence scores
        labels (numpy.ndarray): Array of class labels
        output_dir (str, optional): Directory to save the visualization
        save (bool, optional): Whether to save the visualization. Default is False.
        
    Returns:
        None
    """
    # Load the image using cv2
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image {img_path}")
        return
    
    # Convert from BGR (OpenCV default) to RGB (Matplotlib default)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create a plot
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    ax = plt.gca()
    
    # Get the image dimensions
    img_height, img_width = img.shape[:2]
    
    # Loop through the boxes, scores, and labels to draw rectangles and labels
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        x1 *= img_width
        y1 *= img_height
        x2 *= img_width
        y2 *= img_height
        
        width = x2 - x1
        height = y2 - y1
        
        # Create rectangle
        rect = plt.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        class_name = class_names_defined.get(int(label), f"Class {int(label)}")
        label_text = f"{class_name}: {score:.2f}"
        plt.text(x1, y1-2, label_text,
                color='white',
                fontsize=10,
                fontweight='bold',
                va='top',
                bbox=dict(facecolor='blue', alpha=0.5))
    
    plt.axis('off')
    
    if save and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"detection_{os.path.basename(img_path)}")
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Visualization saved to {output_path}")
    
    plt.show()

def validate_inputs(config_files, checkpoint_files, model_weights, img_path):
    """
    Validate that the input arguments are consistent.
    
    Args:
        config_files (list): List of configuration files
        checkpoint_files (list): List of checkpoint files
        model_weights (list): List of model weights
        img_path (str): Path to the input image
        
    Returns:
        bool: True if inputs are valid, False otherwise
    """
    # Check that the number of config files matches the number of checkpoint files
    if len(config_files) != len(checkpoint_files):
        print("Error: Number of config files must match number of checkpoint files")
        return False
    
    # Check that the number of weights matches the number of models
    if len(model_weights) != len(config_files):
        print("Error: Number of weights must match number of models")
        return False
    
    # Check that all files exist
    for config_file in config_files:
        if not os.path.exists(config_file):
            print(f"Error: Config file not found: {config_file}")
            return False
    
    for checkpoint_file in checkpoint_files:
        if not os.path.exists(checkpoint_file):
            print(f"Error: Checkpoint file not found: {checkpoint_file}")
            return False
    
    if not os.path.exists(img_path):
        print(f"Error: Image file not found: {img_path}")
        return False
    
    return True

def run_detection(config_files, checkpoint_files, model_weights, img_path, output_dir,
                 iou_threshold=0.5, skip_box_threshold=0.0001, conf_type='box_and_model_avg',
                 visualize=False, save_visualization=False):
    """
    Run the complete weapon detection pipeline.
    
    Args:
        config_files (list): List of model configuration files
        checkpoint_files (list): List of model checkpoint files
        model_weights (list): List of weights for each model
        img_path (str): Path to the input image
        output_dir (str): Directory to save outputs
        iou_threshold (float, optional): IoU threshold for box fusion. Default is 0.5.
        skip_box_threshold (float, optional): Threshold to skip low-confidence boxes. Default is 0.0001.
        conf_type (str, optional): Confidence type for fusion. Default is 'box_and_model_avg'.
        visualize (bool, optional): Whether to show visualization. Default is False.
        save_visualization (bool, optional): Whether to save visualization. Default is False.
        
    Returns:
        tuple: Fused boxes, scores, and labels
    """
    # Validate inputs
    if not validate_inputs(config_files, checkpoint_files, model_weights, img_path):
        return None, None, None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    print("Loading models...")
    models = load_models(config_files, checkpoint_files)
    
    # Check if at least one model was loaded successfully
    if not any(models):
        print("Error: No models were loaded successfully.")
        return None, None, None
    else:
        print("Models loaded successfully - ", len(models), " models")
        
    # Run predictions
    print(f"Running detection on image: {img_path}")
    boxes_list, scores_list, labels_list = predict_with_models(
        models, config_files, img_path, output_dir
    )
    
    # Apply ensemble
    print("Applying weighted boxes fusion...")
    boxes, scores, labels = apply_ensemble(
        boxes_list, scores_list, labels_list, 
        model_weights, iou_threshold, skip_box_threshold, conf_type
    )
    
    # Print results
    print("\nDetection Results:")
    for i, (score, label) in enumerate(zip(scores, labels)):
        class_name = class_names_defined.get(int(label), f"Class {int(label)}")
        print(f"Detection {i+1}: {class_name} (Confidence: {score:.4f})")
    
    # Visualize results if requested
    if visualize or save_visualization:
        visualize_results(
            img_path, 
            boxes, 
            scores, 
            labels, 
            output_dir,
            save_visualization
        )
    
    print("Detection completed successfully")
    return boxes, scores, labels


# Example usage (not executed when imported)
if __name__ == "__main__":
    # Example to show how to use the functions
    config_files = [
        "/home/atharvaj/Desktop/SSD/configs/efficient_net_b3_ssd300_voc0712.yaml",
        "/home/atharvaj/Desktop/SSD/configs/mobilenet_v3_ssd320_voc0712.yaml",
        "/home/atharvaj/Desktop/SSD/configs/vgg_ssd300_voc0712.yaml",
        "/home/atharvaj/Desktop/SSD/configs/resnet_50_ssd300_voc0712.yaml"
    ]
    
    checkpoint_files = [
        "/home/atharvaj/Desktop/SSD/models_pth/ssdEff-v1.pth",
        "/home/atharvaj/Desktop/SSD/models_pth/ssdMob-v1.pth",
        "/home/atharvaj/Desktop/SSD/models_pth/vgg_model_3000.pth",
        "/home/atharvaj/Desktop/SSD/models_pth/resnet50-v1.pth"
    ]
    
    model_weights = [0.4, 0.3, 0.3]
    
    # Run the detection pipeline
    boxes, scores, labels = run_detection(
        config_files=config_files,
        checkpoint_files=checkpoint_files,
        model_weights=model_weights,
        img_path="/path/to/image.jpg",
        output_dir="/path/to/output",
        visualize=False
    )