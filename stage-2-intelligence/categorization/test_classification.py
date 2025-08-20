"""
🧪 Test Script for Content Classification System
Tests zero-shot classification on real fitness discussion data
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'stage-1-foundation'))

import pandas as pd
import numpy as np
from content_classifier import ContentClassifier
import json
from pathlib import Path

def create_test_samples():
    """Create manually labeled test samples for evaluation"""
    test_samples = [
        {
            "text": "My knee really hurts after squats. Should I see a doctor?",
            "labels": ["injury", "technique"]
        },
        {
            "text": "What's the best protein powder for muscle gain? Looking for recommendations.",
            "labels": ["nutrition", "strength"]
        },
        {
            "text": "Can someone recommend a good home gym setup for beginners?",
            "labels": ["equipment", "beginner"]
        },
        {
            "text": "I keep losing motivation to go to the gym. How do you stay consistent?",
            "labels": ["motivation"]
        },
        {
            "text": "How many sets and reps should I do for deadlifts in my program?",
            "labels": ["programming", "strength", "technique"]
        },
        {
            "text": "I'm trying to lose 20 pounds. What's the best diet approach?",
            "labels": ["weight_management", "nutrition"]
        },
        {
            "text": "Best cardio exercises for fat loss? I hate running.",
            "labels": ["cardio", "weight_management"]
        },
        {
            "text": "How do I improve my flexibility for better squat depth?",
            "labels": ["flexibility", "technique"]
        },
        {
            "text": "Just started lifting. What are some basic beginner mistakes to avoid?",
            "labels": ["beginner", "technique"]
        },
        {
            "text": "Advanced powerlifting techniques for breaking through plateaus?",
            "labels": ["advanced", "strength", "programming"]
        }
    ]
    return test_samples

def main():
    """Run classification evaluation on fitness data"""
    print("🏷️ Testing Content Classification System")
    print("=" * 50)
    
    # Load fitness data
    data_path = "../../data/fitness_comments_clean.jsonl"
    print(f"📂 Loading data from {data_path}")
    
    if not os.path.exists(data_path):
        print("❌ Data file not found. Please ensure you're running from the correct directory.")
        return
    
    df = pd.read_json(data_path, lines=True)
    print(f"📊 Loaded {len(df)} fitness discussions")
    
    # Use a sample for testing (full dataset takes ~2 hours)
    sample_size = 500
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    texts = df_sample['text'].tolist()
    text_ids = [f"text_{i}" for i in range(len(texts))]
    print(f"🎯 Using sample of {len(texts)} discussions for classification")
    
    # Initialize classifier
    print("🔄 Initializing classifier...")
    classifier = ContentClassifier(
        model_name="facebook/bart-large-mnli",
        confidence_threshold=0.3,  # Lower threshold for more categories
        device="cpu"  # Use CPU to avoid memory issues
    )
    
    if not classifier.load_model():
        print("❌ Failed to load classification model")
        return
    
    # Run classification
    print("🔄 Classifying fitness content...")
    cache_path = "classification_results_cache.pkl"
    results = classifier.classify_texts(
        texts=texts,
        text_ids=text_ids,
        batch_size=8,  # Smaller batches for stability
        use_cache=True,
        cache_path=cache_path
    )
    
    # Print results
    print("\n🎉 Classification Results:")
    print("=" * 50)
    
    stats = classifier.get_classification_statistics()
    print(f"📊 Total Texts: {stats['total_texts']}")
    print(f"🏷️ Categories Used: {stats['categories_used']}/{stats['total_categories']}")
    print(f"📈 Category Coverage: {stats['category_coverage']:.1%}")
    print(f"⭐ Average Confidence: {stats['avg_confidence']:.3f}")
    print(f"🔗 Multi-Label Rate: {stats['multi_label_rate']:.1%}")
    print(f"📊 Avg Labels per Text: {stats['avg_labels_per_text']:.2f}")
    
    print("\n🏆 Top Categories:")
    for category, count in stats['top_categories']:
        percentage = (count / stats['total_texts']) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    # Test on manually labeled samples
    print("\n🧪 Evaluating on manually labeled samples...")
    test_samples = create_test_samples()
    test_texts = [sample['text'] for sample in test_samples]
    test_labels = [sample['labels'] for sample in test_samples]
    test_ids = [f"test_{i}" for i in range(len(test_texts))]
    
    try:
        eval_metrics = classifier.evaluate_on_sample(test_texts, test_labels, test_ids)
        print(f"✅ Evaluation Metrics:")
        print(f"   Accuracy: {eval_metrics['accuracy']:.3f}")
        print(f"   Precision (Macro): {eval_metrics['precision_macro']:.3f}")
        print(f"   Recall (Macro): {eval_metrics['recall_macro']:.3f}")
        print(f"   F1 Score (Macro): {eval_metrics['f1_macro']:.3f}")
    except Exception as e:
        print(f"⚠️ Evaluation failed: {e}")
    
    # Show example classifications
    print("\n📝 Example Classifications:")
    for i, text_id in enumerate(text_ids[:5]):
        text = texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i]
        categories = results.predictions[text_id]
        confidences = results.confidence_scores[text_id]
        
        print(f"\nText {i+1}: {text}")
        print(f"Categories: {', '.join(categories)}")
        top_confidences = sorted(confidences.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"Top Scores: {', '.join([f'{cat}({score:.2f})' for cat, score in top_confidences])}")
    
    # Save detailed results
    results_path = "classification_evaluation_results.json"
    classifier.save_results(results_path)
    print(f"\n💾 Detailed results saved to {results_path}")
    
    # Generate visualization
    print("📊 Generating visualization...")
    try:
        fig = classifier.visualize_category_distribution(
            save_path="category_distribution.html",
            show_plot=False
        )
        print("✅ Visualization saved to category_distribution.html")
    except Exception as e:
        print(f"⚠️ Visualization failed: {e}")
    
    print("\n🎯 Classification evaluation complete!")
    print("Check the generated files for detailed results.")

if __name__ == "__main__":
    main()
