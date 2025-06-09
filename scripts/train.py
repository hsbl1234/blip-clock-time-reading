#!/usr/bin/env python3
"""
Training script for BLIP clock time reading
"""
import sys
sys.path.append('.')

from src.dataset import load_dataset_from_dir, ClockDataset
from src.trainer import ClockTrainer
# BLIP Fine-Tuning for Clock Time Reading - Training Script
# This is the cleaned version of your Colab notebook

import os
import json
import torch
from torch.utils.data import Dataset as TorchDataset
from PIL import Image
from datasets import Dataset, DatasetDict
from transformers import (
    BlipProcessor, 
    BlipForConditionalGeneration,
    TrainingArguments, 
    Trainer
)

class ClockDataset(TorchDataset):
    """Custom dataset class for clock time reading"""
    
    def __init__(self, dataset, processor, max_length=32):
        self.dataset = dataset
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Open and preprocess image
        try:
            image = Image.open(item["image"]).convert("RGB")
        except Exception as e:
            print(f"Error loading image {item['image']}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (224, 224), color='white')

        # Create input text
        question = "What time is it?"

        # Process inputs
        inputs = self.processor(
            images=image,
            text=question,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )

        # Process labels
        labels = self.processor.tokenizer(
            item["label"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        ).input_ids

        return {
            "pixel_values": inputs["pixel_values"].squeeze(),
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }


class ClockTrainer(Trainer):
    """Custom trainer with proper loss computation for BLIP"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Fixed compute_loss method with proper signature"""
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        # Shift labels for language modeling
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Calculate loss
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def load_dataset_from_dir(dataset_path, split):
    """Load dataset from directory structure"""
    folder = os.path.join(dataset_path, split)
    data = {"image": [], "label": []}

    if not os.path.exists(folder):
        print(f"Warning: {folder} does not exist!")
        return Dataset.from_dict(data)

    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if os.path.isdir(label_folder):
            for img in os.listdir(label_folder):
                if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(label_folder, img)
                    try:
                        # Test if image can be opened
                        with Image.open(img_path) as test_img:
                            test_img.verify()
                        data["image"].append(img_path)
                        # Clean up label format (replace - with :)
                        clean_label = label.replace('-', ':')
                        data["label"].append(clean_label)
                    except Exception as e:
                        print(f"Skipping corrupted image: {img_path}")

    print(f"Loaded {len(data['image'])} images for {split} split")
    return Dataset.from_dict(data)


def main():
    # Configuration
    DATASET_PATH = "/path/to/your/dataset"  # Update this path
    OUTPUT_DIR = "./blip-clock-final"
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # Load datasets
    print("Loading datasets...")
    dataset = DatasetDict({
        "train": load_dataset_from_dir(DATASET_PATH, "train"),
        "validation": load_dataset_from_dir(DATASET_PATH, "valid"),
    })

    print("Dataset loaded successfully!")
    print(f"Train: {len(dataset['train'])}, Valid: {len(dataset['validation'])}")

    # Load BLIP model and processor
    print("Loading BLIP model and processor...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model = model.to(device)

    # Create datasets
    train_dataset = ClockDataset(dataset["train"], processor)
    val_dataset = ClockDataset(dataset["validation"], processor)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=3,
        learning_rate=3e-5,
        warmup_steps=50,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        dataloader_num_workers=4,
        fp16=True,
        dataloader_pin_memory=True,
        report_to="none",
        remove_unused_columns=False,
    )

    # Initialize trainer
    trainer = ClockTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train the model
    print("🚀 Starting training...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Epochs: {training_args.num_train_epochs}")
    print(f"Batch size: {training_args.per_device_train_batch_size}")
    print("="*50)

    training_output = trainer.train()

    # Save the model and processor
    print("\n💾 Saving fine-tuned model...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"✅ Model saved to: {OUTPUT_DIR}")

    # Save training info
    def extract_loss_from_history(log_history, loss_type):
        """Extract loss values from training history"""
        try:
            for entry in reversed(log_history):
                if loss_type in entry:
                    return entry[loss_type]
            return "N/A"
        except (IndexError, KeyError, TypeError):
            return "N/A"

    training_info = {
        "model_name": "Salesforce/blip-image-captioning-base",
        "training_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "epochs": training_args.num_train_epochs,
        "batch_size": training_args.per_device_train_batch_size,
        "learning_rate": training_args.learning_rate,
        "final_train_loss": extract_loss_from_history(trainer.state.log_history, "train_loss"),
        "final_eval_loss": extract_loss_from_history(trainer.state.log_history, "eval_loss"),
        "total_training_steps": trainer.state.global_step,
        "training_runtime": getattr(training_output, 'training_time', "N/A"),
    }

    with open(os.path.join(OUTPUT_DIR, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2)

    print("🎉 Training completed successfully!")
    print("📊 Training info saved to: training_info.json")


if __name__ == "__main__":
    main()
