from ssd.default import class_names_defined
import os
import cv2
import matplotlib.pyplot as plt

def save_results(img_path, boxes, scores, labels, output_dir=None):
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
    
    if output_dir or output_dir == '':
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"detection_{os.path.basename(img_path)}")
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"\nVisualization saved to {output_path}")
    else:
        try:
            plt.show()
        except Exception as e:
            print(f"Error displaying plot with matplotlib: {e}")
            # Convert the plot to an image and display with OpenCV
            plt.savefig('/tmp/temp_plot.png', bbox_inches='tight', dpi=300)
            temp_img = cv2.imread('/tmp/temp_plot.png')
            cv2.imshow('Detection Results', temp_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    plt.close()
