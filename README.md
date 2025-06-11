# BLIP Clock Time Reading 🕐

A fine-tuned BLIP (Bootstrapping Language-Image Pre-training) model for accurate time reading from analog clock and watch images.

![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v1.12+-red.svg)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-v4.21+-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YF97tgCnoibeEJ3YgTFoYmUOhgfadBmU?usp=sharing)

## 🎯 Overview

This project fine-tunes the Salesforce BLIP model to read time from analog clocks with high accuracy. The model takes clock images as input and outputs the time in HH:MM format.

### Key Features
- **High Accuracy**: 77.85% accuracy on validation set
- **Analog Clock Focus**: Specialized for analog timepieces only
- **Easy to Use**: Simple API for predictions
- **Well Documented**: Comprehensive documentation and examples
- **Production Ready**: Optimized for inference

### Sample Results
```
Input: [Clock Image showing 2:35] + "What time is it?"
Output: "2:35" ✅

Input: [Clock Image showing 12:00] + "What time is it?"  
Output: "12:00" ✅
```

## 🚀 Quick Start

### Try it Online 🌐
**🔥 Want to test the model immediately?** Use our interactive Google Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YF97tgCnoibeEJ3YgTFoYmUOhgfadBmU?usp=sharing)

The Colab notebook includes:
- Pre-loaded model and dependencies
- Sample clock images to test
- Interactive interface for uploading your own images
- Step-by-step explanations

### Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/hsbl1234/blip-clock-time-reading.git
cd blip-clock-time-reading
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the model** (if not included)
```bash
# Model should be in models/blip-clock-final/
# If missing, contact the repository owner
```

### Basic Usage

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load model
model_path = "./models/blip-clock-final"
processor = BlipProcessor.from_pretrained(model_path)
model = BlipForConditionalGeneration.from_pretrained(model_path)

# Load and process image
image = Image.open("path/to/clock/image.jpg").convert("RGB")
question = "What time is it?"

# Make prediction
inputs = processor(image, question, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=50)
    
time_prediction = processor.decode(outputs[0], skip_special_tokens=True)
print(f"Time: {time_prediction}")
```

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 77.85% |
| **Training Images** | 9,548 |
| **Validation Images** | 1,440 |
| **Model Size** | ~1.5GB |
| **Inference Time** | ~0.1s per image (GPU) |

## 📁 Project Structure

```
blip-clock-time-reading/
├── data/                          # Dataset
│   ├── train/                     # Training images & annotations
│   ├── valid/                     # Validation images & annotations  
│   ├── test/                      # Test images & annotations
│   ├── sample_images/             # Demo images
│   └── README.md                  # Data documentation
├── models/                        # Trained models
│   ├── blip-clock-final/          # Main fine-tuned model
│   └── README.md                  # Model documentation
├── scripts/                       # Training & evaluation scripts
├── results/                       # Evaluation results
├── examples/                      # Usage examples
├── config/                        # Configuration files
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## 🔧 Advanced Usage

### Training Your Own Model

```bash
# Train from scratch
python scripts/train.py --config config/training_config.yaml

# Resume training
python scripts/train.py --resume ./models/checkpoint-1000/
```

### Evaluation

```bash
# Evaluate on test set
python scripts/evaluate.py --model ./models/blip-clock-final/ --data ./data/test/

# Generate predictions
python scripts/predict.py --model ./models/blip-clock-final/ --input ./data/sample_images/
```

### Batch Processing

```python
# Process multiple images
from scripts.batch_predict import BatchPredictor

predictor = BatchPredictor("./models/blip-clock-final/")
results = predictor.predict_folder("./path/to/images/")
```

## 📋 Supported Clock Types

✅ **Supported:**
- Analog wall clocks
- Analog wristwatches  
- Roman numeral clocks
- Vintage and modern timepieces
- Various lighting conditions

## 🛠️ Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Code Formatting

```bash
# Format code
black .
isort .

# Check linting
flake8 .
```

## 📈 Model Details

- **Base Model**: Salesforce/blip-image-captioning-base
- **Architecture**: BLIP (Vision-Language Model)
- **Parameters**: ~387M
- **Training Time**: ~10 minutes on A100
- **Input Resolution**: 224×224 pixels
- **Output Format**: HH:MM (24-hour format)

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{blip-clock-reading-2025,
  title={BLIP Fine-tuning for Analog Clock Time Reading},
  author={Hari shankar b l},
  year={2025},
  url={https://github.com/hsbl1234/blip-clock-time-reading}
}
```

## 🔄 Changelog

### v1.0.0 (June 2025)
- Initial release
- Fine-tuned BLIP model for analog clocks
- 77.85% accuracy on validation set
- Complete documentation and examples

## 🙏 Acknowledgments

- **Salesforce** for the original BLIP model
- **Hugging Face** for the Transformers library
- **PyTorch** team for the deep learning framework
- Contributors and testers

---

**Made with ❤️ for accurate time reading from analog clocks**
