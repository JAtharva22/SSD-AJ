import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from ssd.modeling.detector import build_detection_model
# from your_model_module import build_model  # Import your SSD model builder here

# Example class names; update as needed to match your training classes.
class_names = {
    0: 'background',
    1: 'grenade',
    2: 'gun',
    3: 'knife',
    4: 'pistol',
}
# 'grenade', 'gun', 'knife', 'pistol'

# Define a transform that should match the preprocessing during training.
def get_transform():
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        # Normalize using the same pixel_mean as in training ([123,117,104] if needed).
        transforms.Normalize(mean=[123/255.0, 117/255.0, 104/255.0],
                             std=[1.0/255.0, 1.0/255.0, 1.0/255.0])
    ])
    return transform

def load_model(model_path, device):
    # Build the model architecture
    # For example: model = build_model()   <-- adjust to your implementation
    model = build_detection_model()  # Replace with your actual model builder function
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def detect_and_save(image_path, model, device, save_folder, threshold=0.5):
    # Read image using PIL
    image = Image.open(image_path).convert("RGB")
    orig_image = np.array(image)
    transform = get_transform()
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Adjust the following based on your model output format.
    # Assume predictions is a dict with keys: boxes, labels, scores.
    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()

    # Draw results on the original image (converted to BGR for OpenCV)
    image_draw = cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR)
    for box, score, label in zip(boxes, scores, labels):
        if score < threshold:
            continue
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(image_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{class_names.get(label, 'N/A')}: {score:.2f}"
        cv2.putText(image_draw, text, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

    # Ensure the save folder exists
    os.makedirs(save_folder, exist_ok=True)
    # Build the save filename based on the original image name.
    image_name = os.path.basename(image_path)
    save_path = os.path.join(save_folder, f"detected_{image_name}")
    cv2.imwrite(save_path, image_draw)
    print(f"Saved detected image to: {save_path}")

if __name__ == "__main__":
    # Update these paths as necessary.
    model_path = "outputs/mobilenet_v3_ssd320_voc0712/model_final.pth"
    test_image = "path/to/your/test_image.jpg"
    save_folder = "detected_results"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    detect_and_save(test_image, model, device, save_folder)