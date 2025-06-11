# Models Directory

This directory contains the trained models and related artifacts for the BLIP Clock Time Reading project.

## 📁 Directory Structure

```
models/
├── README.md                    # This file
├── blip-clock-final/           # Main fine-tuned model
│   ├── pytorch_model.bin       # Model weights
│   ├── config.json            # Model configuration
│   ├── preprocessor_config.json # Processor configuration
│   ├── tokenizer.json         # Tokenizer files
│   ├── tokenizer_config.json  
│   ├── special_tokens_map.json
│   ├── vocab.txt              # Vocabulary
│   └── training_info.json     # Training statistics
├── checkpoints/               # Training checkpoints (optional)
│   ├── checkpoint-1000/
│   ├── checkpoint-2000/
│   └── checkpoint-3000/
└── original-baseline/         # Original BLIP for comparison (optional)
    └── evaluation_results.json
```

## 🤖 Model Information

### Base Model
- **Name**: Salesforce/blip-image-captioning-base
- **Type**: Vision-Language Model (VLM)
- **Architecture**: BLIP (Bootstrapping Language-Image Pre-training)
- **Parameters**: ~387M parameters
- **Input**: RGB images (224x224) + text prompts
- **Output**: Text captions/responses

### Fine-tuned Model: blip-clock-final
- **Task**: Analog clock time reading from images
- **Input Format**: Clock/watch images with question "What time is it?"
- **Output Format**: Time in HH:MM format (e.g., "2:35", "12:00")
- **Training Dataset**: 9,548 analog clock images
- **Validation Dataset**: 1,440 analog clock images  
- **Test Accuracy**: 77.85% (1,121/1,440 correct)

## 📊 Performance Metrics

| Model | Accuracy | Correct/Total | Improvement |
|-------|----------|---------------|-------------|
| Original BLIP | 0.00% | 0/1,440 | Baseline |
| Fine-tuned BLIP | 77.85% | 1,121/1,440 | +77.85% |

### Sample Predictions

**Original Model:**
- Input: Clock showing 2:35 → Output: "a clock with the time"
- Input: Clock showing 1:30 → Output: "what time is it? clock,"

**Fine-tuned Model:**
- Input: Clock showing 2:35 → Output: "2:35" ✅
- Input: Clock showing 1:30 → Output: "1:30" ✅

## 🏋️ Training Details

### Training Configuration
```json
{
  "model_name": "Salesforce/blip-image-captioning-base",
  "training_samples": 9548,
  "validation_samples": 1440,
  "epochs": 3,
  "batch_size": 8,
  "learning_rate": 3e-5,
  "optimizer": "AdamW",
  "warmup_steps": 50,
  "max_length": 32,
  "hardware": "NVIDIA A100-SXM4-40GB"
}
```

### Training Progress
| Epoch | Training Loss | Validation Loss |
|-------|---------------|-----------------|
| 1 | 0.1066 | 0.1135 |
| 2 | 0.0228 | 0.0564 |
| 3 | 0.0048 | 0.0470 |

### Training Time
- **Total Training Time**: ~10 minutes
- **Hardware**: NVIDIA A100 (40GB VRAM)
- **Training Steps**: 3,582 total steps
- **Best Model**: Epoch 3 (lowest validation loss)

## 💾 Model Files Description

### Core Model Files
- `pytorch_model.bin`: Main model weights (PyTorch format)
- `config.json`: Model architecture configuration
- `preprocessor_config.json`: Image preprocessing settings

### Tokenizer Files
- `tokenizer.json`: Main tokenizer configuration
- `tokenizer_config.json`: Tokenizer settings and special tokens
- `special_tokens_map.json`: Special token mappings (PAD, UNK, etc.)
- `vocab.txt`: Vocabulary file with all tokens

### Training Artifacts
- `training_info.json`: Complete training statistics and metadata
- `training_args.bin`: Training arguments used during fine-tuning

## 🚀 Usage Instructions

### Loading the Model
```python
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Load the fine-tuned model
model_path = "./models/blip-clock-final"
processor = BlipProcessor.from_pretrained(model_path)
model = BlipForConditionalGeneration.from_pretrained(model_path)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
```

### Making Predictions
```python
from PIL import Image

# Load image
image = Image.open("clock_image.jpg").convert("RGB")
question = "What time is it?"

# Generate prediction
inputs = processor(image, question, return_tensors="pt").to(device)
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_length=50,
        do_sample=False,
        repetition_penalty=1.2
    )

# Decode result
prediction = processor.decode(generated_ids[0], skip_special_tokens=True)
print(f"Predicted time: {prediction}")
```

## 📋 Model Capabilities

### Supported Clock Types
✅ Analog wall clocks  
✅ Analog wristwatches  
✅ Roman numeral clocks  
✅ Modern and vintage analog timepieces  
✅ Various lighting conditions  
✅ Different clock sizes and styles  

### Time Format Support
- **Input**: Any analog clock showing time
- **Output**: 24-hour format (HH:MM)
- **Range**: 00:00 to 23:59
- **Precision**: Minute-level accuracy

### Limitations
- May struggle with extremely blurry images
- Unusual or artistic clock designs might be challenging
- Ambiguous hand positions (12:00 vs 6:00) can cause errors
- Performance may degrade on clock types not in training data
- Only works with analog clocks (no digital displays)

## 🔧 Model Maintenance

### Model Size
- **Total Size**: ~1.5 GB
- **Model Weights**: ~1.4 GB
- **Tokenizer**: ~2 MB
- **Config Files**: ~10 KB

### Version Information
- **Framework**: PyTorch
- **Transformers Version**: 4.21.0+
- **CUDA Compatibility**: 11.0+
- **Python Version**: 3.7+

### Storage Requirements
- **Training**: 40GB+ VRAM recommended
- **Inference**: 4GB+ VRAM minimum
- **Disk Space**: 2GB for model storage

## 🚨 Known Issues & Solutions

### Common Issues

1. **"Missing keys" warning during loading**
   - Issue: `text_decoder.cls.predictions.decoder.bias` missing
   - Solution: This is expected and doesn't affect performance

2. **CUDA out of memory**
   - Solution: Reduce batch size or use CPU inference
   - Alternative: Use `model.half()` for FP16 inference

3. **Slow inference**
   - Solution: Ensure model is on GPU with `.to(device)`
   - Alternative: Use smaller batch sizes

### Performance Optimization
```python
# For faster inference
model.half()  # Use FP16
model.eval()  # Set to evaluation mode

# For memory efficiency
torch.cuda.empty_cache()  # Clear GPU cache
```

## 📈 Future Improvements

### Potential Enhancements
- **Data Augmentation**: Add more diverse analog clock types and angles
- **Multi-language Support**: Train on clocks with different number systems
- **Time Range**: Include seconds precision (HH:MM:SS)
- **Robustness**: Handle partially visible or damaged clocks
- **Speed**: Optimize for faster inference

### Model Variants
- **Lightweight**: Smaller model for mobile deployment
- **High-precision**: Seconds-level accuracy
- **Multi-format**: Support for both 12-hour and 24-hour output

## 📄 Citation

If you use this model in your research, please cite:

```bibtex
@misc{blip-clock-reading-2025,
  title={BLIP Fine-tuning for Analog Clock Time Reading},
  author={Hari Shankar},
  year={2025},
  note={Fine-tuned BLIP model for accurate time reading from analog clock images}
}
```

## 📞 Support

For issues or questions about this model:
- Check the main repository README
- Review the training logs in `training_info.json`
- Open an issue on the GitHub repository
- Refer to the evaluation results for performance benchmarks

---

**Last Updated**: June 2025  
**Model Version**: 1.0  
**Training Completion**: ✅ Successful  
**Specialization**: Analog clocks only
