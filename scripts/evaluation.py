# BLIP Fine-Tuning Evaluation Script
# Compares original vs fine-tuned model performance

import os
import json
import torch
import re
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm


def load_test_dataset(dataset_path):
    """Load test dataset from directory structure"""
    test_folder = os.path.join(dataset_path, "test")
    test_data = []

    if not os.path.exists(test_folder):
        print(f"Error: {test_folder} does not exist!")
        return []

    for label in os.listdir(test_folder):
        label_folder = os.path.join(test_folder, label)
        if os.path.isdir(label_folder):
            for img_file in os.listdir(label_folder):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(label_folder, img_file)
                    # Clean label format
                    clean_label = label.replace('-', ':')
                    test_data.append({
                        'image_path': img_path,
                        'ground_truth': clean_label
                    })

    print(f"Loaded {len(test_data)} test images")
    return test_data


def predict_time(model, processor, image_path, question="What time is it?"):
    """Generate time prediction with improved parameters"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")

        with torch.no_grad():
            # Use the processor's generate method correctly
            inputs = processor(image, question, return_tensors="pt").to(model.device)

            # Generate with proper BLIP parameters
            generated_ids = model.generate(
                **inputs,
                max_length=50,
                min_length=5,
                do_sample=False,
                repetition_penalty=1.2,
                length_penalty=1.0,
                num_beams=1,
            )

        # Decode prediction and clean up
        generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)

        # Remove the input question from output if it appears
        if question in generated_text:
            question_end = generated_text.find(question) + len(question)
            generated_text = generated_text[question_end:].strip()

        # Additional cleanup for common artifacts
        generated_text = re.sub(r'\s+', ' ', generated_text).strip()

        # If the output is still the question or empty, try alternative approach
        if not generated_text or generated_text.lower() == question.lower():
            # Try without question prompt - pure image captioning
            inputs_no_text = processor(image, return_tensors="pt").to(model.device)
            generated_ids = model.generate(
                **inputs_no_text,
                max_length=30,
                do_sample=False,
                repetition_penalty=1.2,
                num_beams=1,
            )
            generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)

            # Look for time patterns in the caption
            time_pattern = r'\b(\d{1,2})[:\s](\d{2})\b'
            match = re.search(time_pattern, generated_text)
            if match:
                generated_text = f"{match.group(1)}:{match.group(2)}"

        # Final cleanup - extract time patterns if text is still too long
        if len(generated_text) > 20:
            time_pattern = r'\b(\d{1,2})[:\s](\d{2})\b'
            match = re.search(time_pattern, generated_text)
            if match:
                generated_text = f"{match.group(1)}:{match.group(2)}"
            else:
                # Take first few tokens if no time pattern found
                tokens = generated_text.split()[:5]
                generated_text = ' '.join(tokens)

        return generated_text if generated_text else "NO_OUTPUT"

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return "ERROR"


def evaluate_model(model, processor, test_data, model_name):
    """Evaluate model on test dataset"""
    correct = 0
    total = len(test_data)
    results = []

    print(f"\n📊 Evaluating {model_name}...")
    print("=" * 40)

    for item in tqdm(test_data, desc=f"Evaluating {model_name}"):
        pred = predict_time(model, processor, item['image_path'])
        gt = item['ground_truth']

        # Check if prediction matches ground truth (exact match)
        is_correct = pred.strip().lower() == gt.strip().lower()
        if is_correct:
            correct += 1

        results.append({
            'ground_truth': gt,
            'prediction': pred,
            'correct': is_correct,
            'image_path': item['image_path']
        })

    accuracy = correct / total if total > 0 else 0

    # Show sample results
    print(f"\n📋 Sample Results for {model_name}:")
    print("-" * 70)
    for i, result in enumerate(results[:10]):
        status = "✅" if result['correct'] else "❌"
        print(f"{i+1:2d}. {status} GT: '{result['ground_truth']}' | Pred: '{result['prediction']}'")

    return accuracy, results


def main():
    # Configuration
    DATASET_PATH = "/path/to/your/dataset"  # Update this path
    MODEL_PATH = "./blip-clock-final"  # Path to your fine-tuned model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load test dataset
    test_data = load_test_dataset(DATASET_PATH)
    if not test_data:
        print("No test data found. Exiting.")
        return

    # Load models and processors
    print("\n🔄 Loading models...")

    # Original model
    print("Loading ORIGINAL BLIP model...")
    original_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    original_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    original_model = original_model.to(device)
    original_model.eval()

    # Fine-tuned model
    print("Loading FINE-TUNED BLIP model...")
    try:
        finetuned_processor = BlipProcessor.from_pretrained(MODEL_PATH)
        finetuned_model = BlipForConditionalGeneration.from_pretrained(MODEL_PATH)
        finetuned_model = finetuned_model.to(device)
        finetuned_model.eval()
        print("✅ Fine-tuned model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading fine-tuned model: {e}")
        return

    # Test with single example first
    print("\n🧪 Testing with single example...")
    if test_data:
        sample = test_data[0]
        orig_pred = predict_time(original_model, original_processor, sample['image_path'])
        ft_pred = predict_time(finetuned_model, finetuned_processor, sample['image_path'])

        print(f"Original model test: GT='{sample['ground_truth']}' | Pred='{orig_pred}'")
        print(f"Fine-tuned model test: GT='{sample['ground_truth']}' | Pred='{ft_pred}'")

    # Run full evaluations
    print("\n" + "=" * 60)
    print("🔍 STARTING MODEL EVALUATION")
    print("=" * 60)

    original_accuracy, original_results = evaluate_model(
        original_model, original_processor, test_data, "ORIGINAL"
    )

    finetuned_accuracy, finetuned_results = evaluate_model(
        finetuned_model, finetuned_processor, test_data, "FINE-TUNED"
    )

    # Calculate improvement
    improvement = finetuned_accuracy - original_accuracy
    improvement_percent = (improvement / original_accuracy * 100) if original_accuracy > 0 else float('inf') if finetuned_accuracy > 0 else 0

    # Summary results
    print("\n" + "=" * 60)
    print("📊 FINAL COMPARISON RESULTS")
    print("=" * 60)
    print(f"🔵 Original Model Accuracy:    {original_accuracy:.4f} ({int(original_accuracy * len(test_data))}/{len(test_data)})")
    print(f"🟢 Fine-tuned Model Accuracy: {finetuned_accuracy:.4f} ({int(finetuned_accuracy * len(test_data))}/{len(test_data)})")
    print(f"🎯 Accuracy Improvement:      {improvement:+.4f} ({improvement_percent:+.2f}%)")

    if improvement > 0:
        print("🎉 Fine-tuning improved the model!")
    elif improvement == 0:
        print("➡️  Same performance - might need different approach")
    else:
        print("⚠️  Fine-tuning decreased performance - check training")

    # Detailed analysis
    original_correct = int(original_accuracy * len(test_data))
    finetuned_correct = int(finetuned_accuracy * len(test_data))
    additional_correct = finetuned_correct - original_correct

    print(f"\n📈 DETAILED ANALYSIS:")
    print(f"Total test samples: {len(test_data)}")
    print(f"Original correct:   {original_correct}")
    print(f"Fine-tuned correct: {finetuned_correct}")
    print(f"Additional correct: {additional_correct}")

    # Save results
    results_summary = {
        "model_comparison": {
            "original_accuracy": float(original_accuracy),
            "finetuned_accuracy": float(finetuned_accuracy),
            "improvement": float(improvement),
            "improvement_percent": float(improvement_percent)
        },
        "dataset_info": {
            "total_samples": len(test_data),
            "original_correct": original_correct,
            "finetuned_correct": finetuned_correct
        },
        "sample_results": {
            "original": original_results[:20],  # Save first 20 results
            "finetuned": finetuned_results[:20]
        }
    }

    # Save to file
    output_file = "./evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")
    print("🎯 Evaluation completed!")


if __name__ == "__main__":
    main()
