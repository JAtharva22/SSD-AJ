import argparse
import os
import logging

import torch
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image

from ssd.default import class_names_defined
from ssd.config import cfg
from ssd.modeling.detector import build_detection_model

# Adjust this dictionary to contain your class names.
CLASS_NAMES = class_names_defined

def load_model(config_file, ckpt_file, device):
    # Load and merge config
    cfg.merge_from_file(config_file)
    cfg.freeze()
    # Build the detector model
    model = build_detection_model(cfg)
    model.to(device)
    
    # Load checkpoint manually and check for the key. This avoids using CheckPointer.
    checkpoint = torch.load(ckpt_file, map_location=device)
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        raise KeyError("Checkpoint does not contain a valid model state dict. Keys found: {}".format(list(checkpoint.keys())))
    
    model.load_state_dict(state_dict)
    model.eval()
    return model

def get_inference_transform():
    # Must match the preprocessing employed during training.
    transform = transforms.Compose([
        transforms.Resize((cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[123/255.0, 117/255.0, 104/255.0],
                             std=[1/255.0, 1/255.0, 1/255.0])
    ])
    return transform


def perform_detection(model, image_path, device, conf_threshold=0.5):
    transform = get_inference_transform()
    image = Image.open(image_path).convert("RGB")
    orig_image = np.array(image)  # for drawing (in RGB)
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(image_tensor)
    
    print(predictions)
    # If predictions is a list or tuple, extract the first element.
    if isinstance(predictions, (list, tuple)):
        predictions = predictions[0]
    
    boxes = predictions["boxes"].cpu().numpy()
    scores = predictions["scores"].cpu().numpy()
    labels = predictions["labels"].cpu().numpy()
    
    # Filter by confidence threshold.
    keep = scores >= conf_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    print(f"Detected {len(boxes)} objects in image {image_path}")
    print(boxes, scores, labels)
    return orig_image, boxes, scores, labels


def draw_detections(image, boxes, scores, labels):
    # Convert image to OpenCV's BGR
    out_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(out_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{CLASS_NAMES.get(label, 'N/A')}:{score:.2f}"
        cv2.putText(out_image, text, (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return out_image

def detect_single_image(config_file, ckpt_file, image_path, output_path, conf_threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config_file, ckpt_file, device)
    orig_image, boxes, scores, labels = perform_detection(model, image_path, device, conf_threshold)
    result_image = draw_detections(orig_image, boxes, scores, labels)
    cv2.imwrite(output_path, result_image)
    print(f"Saved detection result to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect objects on a single image")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path to save output detected image")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence score threshold")
    args = parser.parse_args()
    
    detect_single_image(args.config, args.ckpt, args.image, args.output, args.threshold)

# import os
# import torch
# import cv2
# import numpy as np
# from PIL import Image
# from torchvision import transforms
# from ssd.modeling.detector import build_detection_model
# from ssd.config import cfg
# from ssd.engine.inference import do_evaluation
# from ssd.modeling.detector import build_detection_model
# from ssd.utils import dist_util
# from ssd.utils.checkpoint import CheckPointer
# from ssd.utils.dist_util import synchronize
# from ssd.utils.logger import setup_logger
# # from your_model_module import build_model  # Import your SSD model builder here
# class_names = class_names_defined

# # Example class names; update as needed to match your training classes.
# # 'grenade', 'gun', 'knife', 'pistol'

# # Define a transform that should match the preprocessing during training.
# def get_transform():
#     transform = transforms.Compose([
#         transforms.Resize((320, 320)),
#         transforms.ToTensor(),
#         # Normalize using the same pixel_mean as in training ([123,117,104] if needed).
#         transforms.Normalize(mean=[123/255.0, 117/255.0, 104/255.0],
#                              std=[1.0/255.0, 1.0/255.0, 1.0/255.0])
#     ])
#     return transform

# from ssd.config import cfg
# from ssd.modeling.detector import build_detection_model
# import torch

# def load_model(model_path, device, config_file):
#     cfg.merge_from_file(config_file)
#     cfg.freeze()
#     model = build_detection_model(cfg)
#     checkpoint = torch.load(model_path, map_location=device)
#     if 'model_state_dict' in checkpoint:
#         state_dict = checkpoint['model_state_dict']
#     elif 'model' in checkpoint:
#         state_dict = checkpoint['model']
#     else:
#         raise KeyError("Checkpoint does not contain a valid model state dictionary.")
        
#     model.load_state_dict(state_dict)
#     model.to(device)
#     model.eval()
#     return model

# def detect_and_save(image_path, model, device, save_folder, threshold=0.5):
#     # Read image using PIL
#     image = Image.open(image_path).convert("RGB")
#     orig_image = np.array(image)
#     transform = get_transform()
#     image_tensor = transform(image).unsqueeze(0).to(device)

#     # Inference
#     with torch.no_grad():
#         predictions = model(image_tensor)
    
#     # Adjust the following based on your model output format.
#     # Assume predictions is a dict with keys: boxes, labels, scores.
#     boxes = predictions['boxes'].cpu().numpy()
#     scores = predictions['scores'].cpu().numpy()
#     labels = predictions['labels'].cpu().numpy()

#     # Draw results on the original image (converted to BGR for OpenCV)
#     image_draw = cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR)
#     for box, score, label in zip(boxes, scores, labels):
#         if score < threshold:
#             continue
#         x1, y1, x2, y2 = box.astype(int)
#         cv2.rectangle(image_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         text = f"{class_names.get(label, 'N/A')}: {score:.2f}"
#         cv2.putText(image_draw, text, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5, (0, 255, 0), 2)

#     # Ensure the save folder exists
#     os.makedirs(save_folder, exist_ok=True)
#     # Build the save filename based on the original image name.
#     image_name = os.path.basename(image_path)
#     save_path = os.path.join(save_folder, f"detected_{image_name}")
#     cv2.imwrite(save_path, image_draw)
#     print(f"Saved detected image to: {save_path}")

# # if __name__ == "__main__":
# #     # Update these paths as necessary.
# #     model_path = "outputs/mobilenet_v3_ssd320_voc0712/model_final.pth"
# #     test_image = "path/to/your/test_image.jpg"
# #     save_folder = "detected_results"

# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     model = load_model(model_path, device)

# #     detect_and_save(test_image, model, device, save_folder)