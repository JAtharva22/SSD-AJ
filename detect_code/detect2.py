import os
import cv2
import torch
from ssd.default import class_names_defined
from ssd.config import cfg
from ssd.modeling.detector import build_detection_model
from ssd.utils.checkpoint import CheckPointer
from ssd.data.transforms import build_transforms
from ssd.utils import mkdir

def load_model(config_file, checkpoint_file, device='cuda'):
    # Load the configuration file
    cfg.merge_from_file(config_file)
    cfg.freeze()

    # Build the model
    model = build_detection_model(cfg)
    model.to(device)
    model.eval()

    # Load the checkpoint
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(checkpoint_file, use_latest=True)

    return model

def preprocess_image(image, transform):
    # Apply the transformation to the image
    image = transform(image)[0]
    return image.unsqueeze(0)

# def postprocess_results(results, threshold=0.5):
#     boxes, labels, scores = results
#     keep = float(scores) > threshold
#     return boxes[keep], labels[keep], scores[keep]

def postprocess_results(results, threshold=0.113):
    # print(results)
    boxes, labels, scores = results[0]['boxes'], results[0]['labels'], results[0]['scores']
    keep = scores > threshold
    # print(boxes[keep], labels[keep], scores[keep])
    # print(scores)]

    return boxes[keep], labels[keep], scores[keep]


def draw_boxes(image, boxes, labels, scores, class_names):
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        # print(x1, y1, x2, y2)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_name = class_names[label.item()]
        caption = f"{label_name}: {score:.2f}"
        cv2.putText(image, caption, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image

def main(config_file, checkpoint_file, input_images, output_dir, threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = load_model(config_file, checkpoint_file, device)

    # Get the transformation
    transform = build_transforms(cfg, is_train=False)

    # Make sure the output directory exists
    mkdir(output_dir)

    # Class names
    class_names = class_names_defined

    for image_path in input_images:
        # Read the image
        image = cv2.imread(image_path)
        orig_image = image.copy()

        # Preprocess the image
        image = preprocess_image(image, transform).to(device)

        # Run inference
        with torch.no_grad():
            results = model(image)

        # Post-process the results
        boxes, labels, scores = postprocess_results(results, threshold)

        # Print bounding boxes, labels, and scores
        for box, label, score in zip(boxes, labels, scores):
            print(f"Box: {box}, Label: {class_names[label.item()]}, Score: {score:.2f}")

        # Draw the boxes on the image
        result_image = draw_boxes(orig_image, boxes, labels, scores, class_names)

        # Save the result
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, result_image)
        print(f"Saved result to {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SSD Object Detection")
    parser.add_argument("--config-file", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint-file", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--input-images", nargs="+", required=True, help="Paths to input images")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save output images")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")

    args = parser.parse_args()
    main(args.config_file, args.checkpoint_file, args.input_images, args.output_dir, args.threshold)