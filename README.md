# EfficientNet Backbone for 2D Technical Drawing Analysis

A multi-modal deep learning model based on EfficientNet for automated quality control and error detection in 2D technical drawings and engineering blueprints.

## 🎯 Project Overview

This project implements a specialized neural network architecture designed to identify common errors and inconsistencies in technical drawings, focusing on three critical areas:

- **Welding Symbols**: Detection of missing or incorrectly positioned welding symbols
- **Title Block Validation**: Verification of designer/validator names and completeness
- **Bill of Materials (BOM) Consistency**: Material code and part code validation

## 🏗️ Architecture

The model leverages **EfficientNet** as the backbone architecture with several key enhancements:

- **Frozen Early Layers**: Selective freezing of initial blocks for efficient transfer learning
- **Region-Specific Attention**: Specialized attention modules for different drawing regions
- **Multi-Label Classification**: Simultaneous detection of multiple error types
- **Weighted Loss Function**: Balanced training with emphasis on critical errors

### Model Classes

| Class | Description |
|-------|-------------|
| `missing_weld` | Missing welding symbols |
| `weld_error` | Incorrectly positioned welding symbols |
| `valid_name` | Validator name present and correct |
| `des_name` | Designer name present |
| `mat_cod` | Material code present |
| `part_cod` | Part code present |

## 🚀 Quick Start

### Installation

```bash
pip install torch torchvision efficientnet-pytorch numpy
```

### Basic Usage

```python
from efficientnet_backbone import EfficientNetBackbone2D, DrawingPreprocessor

# Initialize model
model = EfficientNetBackbone2D(
    model_name='efficientnet-b0',
    num_classes=6,
    freeze_layers=3,
    dropout_rate=0.2,
    pretrained=True
)

# Preprocessing
preprocessor = DrawingPreprocessor(input_size=(512, 512))

# Inference
predictions = model.predict(image_tensor, threshold=0.5)
```

### Training

```python
# Setup optimizer for trainable parameters only
optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=1e-4)

# Training loop
for batch in dataloader:
    images, targets = batch
    outputs = model(images, region_masks)
    loss, loss_components = model.compute_loss(outputs, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 📊 Features

### 🎯 Multi-Region Analysis
- **Welding Region**: Central-upper area focus for welding symbol detection
- **Title Block**: Lower-right corner analysis for validation information
- **BOM Region**: Right-side analysis for material and part codes

### 🔧 Flexible Architecture
- Configurable backbone models (EfficientNet-B0 through B7)
- Adjustable layer freezing for different training strategies
- Modular design for easy extension to new error types

### 📈 Training Optimizations
- **Selective Parameter Training**: Only unfreeze necessary layers
- **Weighted Loss Function**: Emphasis on critical error detection
- **Region-Specific Masks**: Focus computational resources on relevant areas
- **Data Augmentation**: Specialized transforms for technical drawings

## 🛠️ Model Configuration

### Available EfficientNet Variants

| Model | Parameters | Input Size | Recommended Use |
|-------|------------|------------|-----------------|
| efficientnet-b0 | 5.3M | 224x224 | Fast inference, limited data |
| efficientnet-b1 | 7.8M | 240x240 | Balanced performance |
| efficientnet-b2 | 9.2M | 260x260 | Higher accuracy |
| efficientnet-b3 | 12M | 300x300 | Production deployment |
| efficientnet-b4 | 19M | 380x380 | High-quality analysis |

### Hyperparameter Recommendations

```python
# For fine-tuning
config = {
    'learning_rate': 1e-4,
    'batch_size': 16,
    'freeze_layers': 3,
    'dropout_rate': 0.2,
    'weight_decay': 1e-5
}

# For full training
config = {
    'learning_rate': 1e-3,
    'batch_size': 8,
    'freeze_layers': 0,
    'dropout_rate': 0.3,
    'weight_decay': 1e-4
}
```

## 📁 Project Structure

```
├── efficientnet_backbone.py    # Main model implementation
├── preprocessor.py             # Image preprocessing utilities
├── training.py                 # Training scripts and utilities
├── evaluation.py               # Model evaluation and metrics
├── utils/                      # Helper functions
│   ├── data_loader.py         # Custom data loading
│   ├── visualization.py       # Result visualization
│   └── metrics.py             # Custom metrics
├── configs/                    # Configuration files
├── examples/                   # Usage examples
└── tests/                      # Unit tests
```

## 📋 Requirements
- label studio
- Python 3.8+
- PyTorch 1.9+
- torchvision 0.10+
- efficientnet-pytorch
- numpy
- PIL (Pillow)
- matplotlib (for visualization)

## 🔄 Training Pipeline

1. **Data Preparation**: Organize technical drawings with corresponding labels
2. **Preprocessing**: Apply specialized transforms for technical drawings
3. **Region Masking**: Generate attention masks for different drawing regions
4. **Model Initialization**: Load pretrained EfficientNet and freeze early layers
5. **Fine-tuning**: Train on labeled dataset with multi-label objectives
6. **Evaluation**: Assess performance on validation set
7. **Deployment**: Export model for production inference

## 📊 Performance Metrics

The model is evaluated using:
- **Precision/Recall** for each error class
- **F1-Score** for balanced performance assessment
- **Area Under Curve (AUC)** for classification confidence
- **Mean Average Precision (mAP)** for multi-label performance

## 🚀 Future Enhancements

- [ ] Integration with OCR for text-based validation
- [ ] Support for CAD file formats (DWG, DXF)
- [ ] Real-time inference optimization
- [ ] Web interface for batch processing
- [ ] Integration with PLM/CAD systems

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

---

**Note**: This model is designed specifically for technical drawing analysis and may require domain-specific fine-tuning for optimal performance on your dataset.
