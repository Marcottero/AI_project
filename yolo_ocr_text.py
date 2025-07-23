from ultralytics import YOLO
import os
import yaml
import pandas as pd
import easyocr
import matplotlib.pyplot as plt
from ultralytics.utils import torch_utils

# Load a pre-trained YOLOv8 model
model = YOLO('yolo11n.pt')

# Transfer learning parameters
# model.model.fuse()  # Optimize the model for inference (optional)
model.train(
    data=r'C:\Users\k1020696\CV_project\text.yaml',  # Specify the dataset
    epochs=10,                         # Number of epochs
    imgsz=1080,                        # Image size
    lr0=0.01,                          # Initial learning rate
    freeze=20,                         # Freeze the first 20 layers (backbone)
    hsv_h=0.015,                       # HSV hue augmentation
    hsv_s=0.7,                         # HSV saturation augmentation
    hsv_v=0.4,                         # HSV value augmentation
    translate=0.1,                     # random translation
    scale=0.5                          # random scaling
   )

# Save the trained model
model.save('yolov11n_text.pt')
# Pruning some weights to downsize the model
torch_utils.prune(model, sparsity=0.3)  # Prune 30% of the weights 

# Run inference on the trained model
results = model.predict(source=r'C:\Users\k1020696\CV_project\obj_det_text\train\images',  # Specify the image folder
                        conf=0.5,  # Confidence threshold
                        save=True,  # Save results
                        save_txt=True,
                        project='runs/detect_text'
                        )  # Save annotations in text format

# Print inference results
for result in results:
    print(f"Image: {result.path}")
    print(f"Detections: {result.boxes.xyxy}")  # Bbox coordinates
    print(f"Labels: {result.boxes.cls}")  # Class labels
    print(f"Confidences: {result.boxes.conf}")  # Prediction confidences

# Path to the training results log file
runs_dir = 'runs/detect_text'
# Find the latest run in the custom directory
last_run = sorted(os.listdir(runs_dir))[-1]
results_path = os.path.join(runs_dir, last_run, 'results.csv')
hyp_path = os.path.join(runs_dir, last_run, 'hyp.yaml')

# Load data from the results CSV file
df = pd.read_csv(results_path)

# Plot main metrics
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
plt.plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss')
plt.plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss during training')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
plt.plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
plt.plot(df['epoch'], df['metrics/mAP_0.5(B)'], label='mAP@0.5')
plt.plot(df['epoch'], df['metrics/mAP_0.5:0.95(B)'], label='mAP@0.5:0.95')
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.title('Evaluation metrics')
plt.legend()

plt.tight_layout()
# Save the plot as a PNG file
output_path = os.path.join(runs_dir, last_run, 'training_metrics.png')
plt.savefig(output_path)
plt.close()
print(f"Plot saved at: {output_path}")

# Initialize EasyOCR reader (English language, adjust if needed)
reader = easyocr.Reader(['it'])

# Directory to save cropped images and OCR results
cropped_dir = os.path.join(runs_dir, last_run, 'crops')
os.makedirs(cropped_dir, exist_ok=True)

ocr_results = []

for result in results:
    img = result.orig_img
    img_name = os.path.splitext(os.path.basename(result.path))[0]
    for i, box in enumerate(result.boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)
        crop = img[y1:y2, x1:x2]
        crop_path = os.path.join(cropped_dir, f"{img_name}_crop_{i}.png")
        # Save cropped image
        plt.imsave(crop_path, crop)
        # OCR on the cropped image
        ocr_out = reader.readtext(crop)
        text = " ".join([item[1] for item in ocr_out])
        ocr_results.append({
            'image': result.path,
            'crop_path': crop_path,
            'bbox': [x1, y1, x2, y2],
            'text': text
        })
        print(f"OCR for {crop_path}: {text}")

# Save OCR results to a CSV file
ocr_df = pd.DataFrame(ocr_results)
ocr_csv_path = os.path.join(runs_dir, last_run, 'ocr_results.csv')
ocr_df.to_csv(ocr_csv_path, index=False)
print(f"OCR results saved at: {ocr_csv_path}")

# Python logic for further processing or analysis of OCR results
