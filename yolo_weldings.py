from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import os

def train_and_validate(dataset_path='weld.yaml', imgsz=1020, freeze=10, epochs=10, batch=8):
    # Load pre-trained YOLO model
    model = YOLO(f'yolov8n.pt')
    
    # Train the model (no augmentation, only imgsz and freeze)
    results = model.train(
        data=dataset_path,
        imgsz=imgsz,
        freeze=freeze,
        epochs=epochs,
        batch=batch,
        val=True,           # Run validation every epoch
        plots=True,         # Save training plots
        save_json=True,     # Save validation results in COCO json
        dropout=0.2
    )
    
    # Save best model
    best_model_path = f'yolov8n_weld_best.pt'
    model.save(best_model_path)
    print(f"Best model saved as: {best_model_path}")
    return model, results

def run_inference(model, source_path, conf=0.70):
    results = model.predict(
        source=source_path,
        conf=conf,
        save=True,
        save_txt=True,
        save_conf=True
    )
    print("Inference completed.")
    return results

def plot_metrics(run_dir):
    # Plot results.png and confusion_matrix.png if present
    for fname in ["results.png", "confusion_matrix.png"]:
        fpath = os.path.join(run_dir, fname)
        if os.path.exists(fpath):
            img = plt.imread(fpath)
            plt.figure(figsize=(8, 6))
            plt.imshow(img)
            plt.title(fname)
            plt.axis('off')
            plt.show()
        else:
            print(f"Plot not found: {fpath}")

def main():
    dataset_path = r'C:\Users\k1020696\CV_project\weld.yaml'
    images_path = r'C:\Users\k1020696\CV_project\obj_det_weld\train\images'
    
    # Training e validazione
    model, results = train_and_validate(
        dataset_path=dataset_path,
        imgsz=1020,
        freeze=10,
        epochs=10,
        batch=8
    )
    # Use or explicitly ignore 'results' to avoid unused variable warning
    _ = results
    
    # Inference
    run_inference(model, images_path)

if __name__ == "__main__":
    main()