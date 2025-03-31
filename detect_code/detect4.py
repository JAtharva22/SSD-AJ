import os
import cv2
import torch
from ssd.default import class_names_defined
from ssd.config import cfg
from ssd.modeling.detector import build_detection_model
from ssd.utils.checkpoint import CheckPointer
from ssd.data.transforms import build_transforms
from ssd.utils import mkdir

def load_model(config_file, checkpoint_file):
    # Load the configuration file
    a = cfg.clone()
    a.merge_from_file(config_file)
    a.freeze()

    # Build the model# Choose device and build model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_detection_model(a)
    model.to(device)
    model.eval()

    # Load the checkpoint
    checkpointer = CheckPointer(model, save_dir=a.OUTPUT_DIR)
    checkpointer.load(checkpoint_file, use_latest=True)

    return model

def preprocess_image(image, transform):
    # Apply the transformation to the image
    image = transform(image)[0]
    return image.unsqueeze(0)

def postprocess_results(results, threshold=0.1):
    boxes, labels, scores = results[0]['boxes'], results[0]['labels'], results[0]['scores']
    keep = scores > threshold
    return boxes[keep], labels[keep], scores[keep]

def predict(loaded_model, config_file, image_path, threshold=0.3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    b = cfg.clone()
    b.merge_from_file(config_file)
    b.freeze()

    # Get the transformation
    transform = build_transforms(b, is_train=False)
    results_list = []

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Input image is empty or not loaded properly.")

    # Preprocess the image
    image = preprocess_image(image, transform).to(device)

    # Run inference
    with torch.no_grad():
        results = loaded_model(image)

    # Post-process the results
    boxes, labels, scores = postprocess_results(results, threshold)

    normalized_bbox = []
    image_size = b.INPUT.IMAGE_SIZE
    for box in boxes:
        normalized_bbox.append(
            [box[0].item() / image_size, 
                box[1].item() / image_size,
                box[2].item() / image_size,
                box[3].item() / image_size]
        )

    results_list.append((normalized_bbox, labels, scores))

    return results_list

# ======================== To draw boxes ==============================

def draw_boxes(image, boxes, labels, scores, class_names):
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        # print(x1, y1, x2, y2)
        height, width, _ = image.shape
        # height, width, _ = 300, 300, 3
        print(height, width)
        x1, y1, x2, y2 = x1 * width, y1 * height, x2 * width, y2 * height
        print(x1, y1, x2, y2)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(x1, y1, x2, y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_name = class_names[label.item()]
        caption = f"{label_name}: {score:.2f}"
        cv2.putText(image, caption, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image

def draw_and_save_results(output_dir, orig_image, normalized_bbox, labels, scores, class_names, image_path):
    # Make sure the output directory exists
    mkdir(output_dir)

    result_image = draw_boxes(orig_image, normalized_bbox, labels, scores, class_names)

    # Save the result
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, result_image)
    print(f"Saved result to {output_path}")
