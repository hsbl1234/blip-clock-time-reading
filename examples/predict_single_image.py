#!/usr/bin/env python3
"""
Single Image Prediction Example

This script demonstrates how to use the fine-tuned BLIP model
to predict time from a single clock image.

Usage:
    python examples/predict_single_image.py --image path/to/clock.jpg
    python examples/predict_single_image.py --image path/to/clock.jpg --model ./models/blip-clock-final/
"""

import argparse
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import sys

def load_model(model_path):
    """Load the fine-tuned BLIP model and processor."""
    print(f"Loading model from: {model_path}")
    
    try:
        processor = BlipProcessor.from_pretrained(model_path)
        model = BlipForConditionalGeneration.from_pretrained(model_path)
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        print(f"Model loaded successfully on {device}")
        return processor, model, device
        
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def predict_time(image_path, processor, model, device):
    """Predict time from a clock image."""
    
    # Load and process image
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"Image loaded: {image_path} (Size: {image.size})")
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    # Prepare input
    question = "What time is it?"
    inputs = processor(image, question, return_tensors="pt").to(device)
    
    # Generate prediction
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_length=50,
            do_sample=False,
            repetition_penalty=1.2,
            pad_token_id=processor.tokenizer.pad_token_id
        )
    
    # Decode result
    prediction = processor.decode(generated_ids[0], skip_special_tokens=True)
    
    return prediction

def main():
    parser = argparse.ArgumentParser(description="Predict time from clock image")
    parser.add_argument("--image", required=True, help="Path to clock image")
    parser.add_argument("--model", default="./models/blip-clock-final/", 
                       help="Path to model directory")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
        
    if not os.path.exists(args.model):
        print(f"Error: Model directory not found: {args.model}")
        sys.exit(1)
    
    # Load model
    processor, model, device = load_model(args.model)
    
    # Make prediction
    print("\nMaking prediction...")
    prediction = predict_time(args.image, processor, model, device)
    
    if prediction:
        print(f"\n{'='*50}")
        print(f"Image: {os.path.basename(args.image)}")
        print(f"Predicted Time: {prediction}")
        print(f"{'='*50}")
    else:
        print("Failed to make prediction")

if __name__ == "__main__":
    main()
