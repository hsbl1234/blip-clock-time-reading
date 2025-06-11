# Data Directory

This directory contains the dataset used for training and evaluating the BLIP Clock Time Reading model.

## 📁 Directory Structure

```
data/
├── README.md                    # This file
├── train/                       # Training dataset
│   ├── images/                  # Training images (9,548 images)
│   │   ├── clock_001.jpg
│   │   ├── clock_002.jpg
│   │   └── ...
│   └── annotations.json         # Training labels and metadata
├── validation/                  # Validation dataset  
│   ├── images/                  # Validation images (1,440 images)
│   │   ├── val_001.jpg
│   │   ├── val_002.jpg
│   │   └── ...
│   └── annotations.json         # Validation labels
├── test/                        # Test dataset (if separate)
│   ├── images/
│   └── annotations.json
└── sample_images/               # Sample images for demonstration
    ├── analog_clock_235.jpg     # Shows 2:35
    ├── wall_clock_130.jpg       # Shows 1:30
    └── roman_clock_1200.jpg     # Shows 12:00
```

## 📊 Dataset Statistics

| Split | Images | Time Range | Formats |
|-------|--------|------------|---------|
| Training | 9,548 | 00:00 - 23:59 | Analog clocks only |
| Validation | 1,440 | 00:00 - 23:59 | Analog clocks only |
| **Total** | **10,988** | **Full 24-hour coverage** | **Analog timepieces** |

## 🕐 Clock Types Included

### Analog Clocks Only
- Wall clocks (round, square, decorative)
- Wristwatches (men's, women's, sports)
- Antique and vintage timepieces
- Roman numeral clocks
- Modern minimalist designs
- Mantle clocks
- Grandfather clocks
- Kitchen clocks
- Office clocks

### Time Distribution
- **Hours**: Even distribution across 00-23 (24-hour format)
- **Minutes**: All minute values (00-59) represented
- **Special times**: 12:00, 6:00, 3:00, 9:00 well-represented
- **Ambiguous positions**: Challenging cases included for robustness

## 📝 Annotation Format

The annotations are stored in JSON format with the following structure:

```json
{
  "images": [
    {
      "id": "clock_001",
      "filename": "clock_001.jpg",
      "width": 224,
      "height": 224,
      "time": "14:35",
      "clock_type": "analog",
      "question": "What time is it?",
      "answer": "14:35"
    }
  ]
}
```

### Annotation Fields
- `id`: Unique identifier for the image
- `filename`: Image filename
- `width`/`height`: Image dimensions (standardized to 224x224)
- `time`: Ground truth time in HH:MM format (24-hour)
- `clock_type`: "analog" (all clocks in this dataset)
- `question`: Input question (standardized to "What time is it?")
- `answer`: Expected model output (same as time field)

## 🖼️ Image Specifications

### Image Properties
- **Format**: JPG/JPEG
- **Resolution**: 224×224 pixels (resized during preprocessing)
- **Color**: RGB (3 channels)
- **Quality**: High-resolution originals, preprocessed for training

### Image Sources
- Custom captured analog clock images
- Synthetic analog clock generations
- Curated from public datasets (with proper licensing)
- Various lighting conditions and angles

## 🔄 Data Preprocessing

Images undergo the following preprocessing steps:
1. **Resize**: All images resized to 224×224 pixels
2. **Normalization**: Pixel values normalized to [0,1] range
3. **Format**: Converted to RGB if needed
4. **Quality check**: Blurry or corrupted images filtered out

## 📋 Usage Instructions

### Loading the Dataset

```python
import json
from PIL import Image
import os

def load_dataset(data_dir, split='train'):
    """Load images and annotations for a given split"""
    
    # Load annotations
    ann_path = os.path.join(data_dir, split, 'annotations.json')
    with open(ann_path, 'r') as f:
        annotations = json.load(f)
    
    # Load images
    images = []
    labels = []
    
    for item in annotations['images']:
        img_path = os.path.join(data_dir, split, 'images', item['filename'])
        image = Image.open(img_path).convert('RGB')
        
        images.append(image)
        labels.append(item['time'])
    
    return images, labels

# Usage
train_images, train_labels = load_dataset('./data', 'train')
val_images, val_labels = load_dataset('./data', 'validation')
```

### Data Validation

```python
def validate_dataset(data_dir, split):
    """Validate dataset integrity"""
    
    ann_path = os.path.join(data_dir, split, 'annotations.json')
    img_dir = os.path.join(data_dir, split, 'images')
    
    with open(ann_path, 'r') as f:
        annotations = json.load(f)
    
    missing_files = []
    for item in annotations['images']:
        img_path = os.path.join(img_dir, item['filename'])
        if not os.path.exists(img_path):
            missing_files.append(item['filename'])
    
    print(f"Dataset validation for {split}:")
    print(f"Total annotations: {len(annotations['images'])}")
    print(f"Missing files: {len(missing_files)}")
    
    return len(missing_files) == 0
```

## 🎯 Data Quality

### Quality Metrics
- **Accuracy**: All time labels manually verified
- **Consistency**: Standardized annotation format across all splits
- **Completeness**: No missing annotations or corrupted images
- **Diversity**: Balanced representation of different analog clock styles

### Quality Assurance
1. **Manual Review**: Random sample validation by human annotators
2. **Automated Checks**: Scripts to verify file integrity and format consistency
3. **Cross-validation**: Multiple annotators for ambiguous cases
4. **Edge Cases**: Intentionally included challenging examples

## ⚠️ Known Issues

### Dataset Limitations
- **Clock visibility**: Some images may have partially obscured clock faces
- **Ambiguous times**: 12:00 vs 6:00 hand positions can be challenging
- **Image quality**: Varying lighting conditions and image clarity
- **Hand overlap**: Hour and minute hands may overlap at certain times

### Handling Ambiguous Cases
- Ambiguous analog clock positions are labeled based on context clues
- When uncertain, multiple annotators reviewed the images
- Difficult cases are flagged in the metadata

## 📈 Dataset Statistics

### Time Distribution
```
Hour Distribution (0-23): Approximately uniform
Minute Distribution (0-59): Balanced across all minutes
Clock Styles:
- Traditional round clocks: 45% (4,944 images)
- Square/rectangular clocks: 25% (2,747 images)
- Roman numeral clocks: 15% (1,648 images)
- Modern/minimalist: 10% (1,099 images)
- Antique/vintage: 5% (550 images)
```

### Training Splits
- **Training**: 87% (9,548 images)
- **Validation**: 13% (1,440 images)
- **Test**: Separate test set available on request

## 🔧 Data Maintenance

### Version Control
- **Current Version**: 1.0
- **Last Updated**: June 2025
- **Format**: JSON annotations + JPG images

### File Integrity
- All images verified for corruption
- Checksums available for data validation
- Backup copies maintained

## 📄 License & Citation

### Data License
This dataset is provided for research and educational purposes. Please cite this work if you use the dataset:

```bibtex
@dataset{blip-clock-dataset-2025,
  title={Analog Clock Time Reading Dataset for BLIP Fine-tuning},
  author={Hari Shankar},
  year={2025},
  note={Dataset of 10,988 analog clock images with time annotations}
}
```

### Usage Rights
- ✅ Academic research
- ✅ Educational purposes  
- ✅ Non-commercial projects
- ❌ Commercial use without permission

## 🤝 Contributing

To contribute additional data:
1. Follow the annotation format specified above
2. Ensure image quality meets standards (224x224, clear visibility)
3. Only analog clocks (no digital displays)
4. Provide accurate time labels in HH:MM format
5. Submit via pull request with validation scripts

## 📞 Support

For dataset-related questions:
- Check annotation format examples
- Validate data integrity using provided scripts
- Open an issue for missing or corrupted files
- Contact maintainers for additional data splits

---

**Last Updated**: June 2025  
**Dataset Version**: 1.0  
**Total Images**: 10,988  
**Clock Type**: Analog only  
**Annotation Format**: JSON
