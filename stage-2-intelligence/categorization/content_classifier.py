"""
🏷️ Content Classification System
Zero-shot classification for fitness content categorization

This module implements advanced content classification using:
- Zero-shot classification with transformers
- Multi-label fitness category tagging
- Confidence scoring and threshold optimization
- Batch processing for large datasets

Author: ML Engineering Portfolio Project
"""

import numpy as np
import pandas as pd
import json
import pickle
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Core ML libraries
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ClassificationResult:
    """Container for classification results and metadata"""
    predictions: Dict[str, List[str]]  # text_id -> [categories]
    confidence_scores: Dict[str, Dict[str, float]]  # text_id -> {category: score}
    category_stats: Dict[str, int]  # category -> count
    performance_metrics: Dict[str, float]
    threshold: float
    model_info: Dict[str, str]

class ContentClassifier:
    """
    Zero-shot fitness content classification system
    
    This class provides multi-label classification of fitness discussions
    into predefined categories using transformer-based zero-shot learning.
    """
    
    def __init__(self, 
                 model_name: str = "facebook/bart-large-mnli",
                 confidence_threshold: float = 0.5,
                 device: str = "auto"):
        """
        Initialize content classification system
        
        Args:
            model_name: HuggingFace model for zero-shot classification
            confidence_threshold: Minimum confidence for category assignment
            device: Device for model inference ("auto", "cpu", "cuda")
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # Fitness categories with detailed descriptions
        self.fitness_categories = {
            "injury": "Pain, discomfort, injuries, rehabilitation, recovery, aches, soreness, medical issues",
            "nutrition": "Diet, meal planning, supplements, protein, calories, macros, eating habits, food",
            "equipment": "Gym equipment, gear, tools, machines, weights, accessories, home gym setup",
            "motivation": "Mental barriers, consistency, goal setting, discipline, staying motivated, mindset",
            "technique": "Form, exercise execution, proper movement, technique tips, how-to guides",
            "programming": "Workout plans, routines, sets, reps, progression, periodization, training structure",
            "weight_management": "Weight loss, weight gain, body composition, cutting, bulking, fat loss",
            "strength": "Building strength, powerlifting, deadlifts, squats, bench press, PRs",
            "cardio": "Running, cycling, swimming, endurance, HIIT, cardiovascular fitness",
            "flexibility": "Stretching, mobility, yoga, flexibility training, range of motion",
            "beginner": "Starting fitness, newbie questions, basic advice, getting started",
            "advanced": "Advanced techniques, elite training, competition, specialized training"
        }
        
        # Initialize models
        self.classifier = None
        self.is_loaded = False
        
        # Results storage
        self.classification_result = None
    
    def load_model(self) -> bool:
        """Load zero-shot classification model"""
        try:
            logger.info(f"🔄 Loading classification model: {self.model_name}")
            
            # Set device
            device = 0 if torch.cuda.is_available() and self.device == "auto" else -1
            if self.device == "cpu":
                device = -1
            
            # Load zero-shot classifier
            self.classifier = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=device,
                return_all_scores=True
            )
            
            self.is_loaded = True
            logger.info("✅ Classification model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def classify_texts(self, 
                      texts: List[str],
                      text_ids: Optional[List[str]] = None,
                      batch_size: int = 16,
                      use_cache: bool = True,
                      cache_path: str = "classification_cache.pkl") -> ClassificationResult:
        """
        Classify texts into fitness categories
        
        Args:
            texts: List of text documents to classify
            text_ids: Optional list of text identifiers
            batch_size: Number of texts to process at once
            use_cache: Whether to use cached results
            cache_path: Path for caching results
            
        Returns:
            ClassificationResult with comprehensive classification data
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info(f"🏷️ Classifying {len(texts)} texts into fitness categories...")
        
        # Check for cached results
        if use_cache and Path(cache_path).exists():
            logger.info("📂 Loading cached classification results...")
            return self._load_cached_results(cache_path)
        
        # Generate text IDs if not provided
        if text_ids is None:
            text_ids = [f"text_{i}" for i in range(len(texts))]
        
        # Prepare category labels and descriptions
        category_labels = list(self.fitness_categories.keys())
        
        # Process texts in batches
        predictions = {}
        confidence_scores = {}
        
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            batch_ids = text_ids[start_idx:end_idx]
            
            logger.info(f"🔄 Processing batch {batch_idx + 1}/{total_batches}")
            
            for text, text_id in zip(batch_texts, batch_ids):
                try:
                    # Perform zero-shot classification
                    result = self.classifier(text, category_labels)
                    
                    # Extract predictions above threshold
                    text_predictions = []
                    text_confidences = {}
                    
                    for label, score in zip(result['labels'], result['scores']):
                        text_confidences[label] = float(score)
                        if score >= self.confidence_threshold:
                            text_predictions.append(label)
                    
                    # Ensure at least one category if none meet threshold
                    if not text_predictions and text_confidences:
                        best_label = max(text_confidences.keys(), key=lambda x: text_confidences[x])
                        text_predictions.append(best_label)
                    
                    predictions[text_id] = text_predictions
                    confidence_scores[text_id] = text_confidences
                    
                except Exception as e:
                    logger.warning(f"Failed to classify text {text_id}: {e}")
                    predictions[text_id] = ["unknown"]
                    confidence_scores[text_id] = {"unknown": 0.0}
        
        # Calculate statistics
        category_stats = self._calculate_category_statistics(predictions)
        performance_metrics = self._calculate_performance_metrics(predictions, confidence_scores)
        
        # Create result object
        self.classification_result = ClassificationResult(
            predictions=predictions,
            confidence_scores=confidence_scores,
            category_stats=category_stats,
            performance_metrics=performance_metrics,
            threshold=self.confidence_threshold,
            model_info={
                "model_name": self.model_name,
                "device": str(self.classifier.device) if hasattr(self.classifier, 'device') else "unknown",
                "categories": list(self.fitness_categories.keys())
            }
        )
        
        # Cache results
        if use_cache:
            self._save_cached_results(cache_path)
        
        logger.info("🎉 Classification complete!")
        return self.classification_result
    
    def _calculate_category_statistics(self, predictions: Dict[str, List[str]]) -> Dict[str, int]:
        """Calculate category frequency statistics"""
        category_counts = Counter()
        
        for text_predictions in predictions.values():
            for category in text_predictions:
                category_counts[category] += 1
        
        return dict(category_counts)
    
    def _calculate_performance_metrics(self, 
                                     predictions: Dict[str, List[str]], 
                                     confidence_scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate classification performance metrics"""
        
        # Average confidence scores
        all_confidences = []
        for text_confidences in confidence_scores.values():
            all_confidences.extend(text_confidences.values())
        
        # Category coverage
        all_categories = set()
        for text_predictions in predictions.values():
            all_categories.update(text_predictions)
        
        # Multi-label statistics
        multi_label_count = sum(1 for preds in predictions.values() if len(preds) > 1)
        single_label_count = len(predictions) - multi_label_count
        
        return {
            "avg_confidence": np.mean(all_confidences) if all_confidences else 0.0,
            "min_confidence": min(all_confidences) if all_confidences else 0.0,
            "max_confidence": max(all_confidences) if all_confidences else 0.0,
            "categories_covered": len(all_categories),
            "total_categories": len(self.fitness_categories),
            "coverage_rate": len(all_categories) / len(self.fitness_categories),
            "multi_label_rate": multi_label_count / len(predictions) if predictions else 0.0,
            "single_label_rate": single_label_count / len(predictions) if predictions else 0.0,
            "avg_labels_per_text": np.mean([len(preds) for preds in predictions.values()]) if predictions else 0.0
        }
    
    def evaluate_on_sample(self, 
                          texts: List[str], 
                          true_labels: List[List[str]],
                          text_ids: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate classification performance on labeled sample
        
        Args:
            texts: List of text documents
            true_labels: List of ground truth category lists
            text_ids: Optional text identifiers
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info(f"🧪 Evaluating classification on {len(texts)} labeled samples...")
        
        # Classify texts
        results = self.classify_texts(texts, text_ids, use_cache=False)
        
        # Convert to binary label format for evaluation
        all_categories = list(self.fitness_categories.keys())
        y_true = []
        y_pred = []
        
        for i, text_id in enumerate(text_ids or [f"text_{i}" for i in range(len(texts))]):
            # True labels
            true_binary = [1 if cat in true_labels[i] else 0 for cat in all_categories]
            y_true.append(true_binary)
            
            # Predicted labels
            pred_binary = [1 if cat in results.predictions[text_id] else 0 for cat in all_categories]
            y_pred.append(pred_binary)
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate metrics
        metrics = {}
        
        # Overall metrics
        metrics['accuracy'] = np.mean(y_true == y_pred)
        metrics['precision_macro'] = self._calculate_precision(y_true, y_pred, average='macro')
        metrics['recall_macro'] = self._calculate_recall(y_true, y_pred, average='macro')
        metrics['f1_macro'] = self._calculate_f1(y_true, y_pred, average='macro')
        
        # Per-category metrics
        for i, category in enumerate(all_categories):
            cat_true = y_true[:, i]
            cat_pred = y_pred[:, i]
            
            if cat_true.sum() > 0:  # Only calculate if category exists in true labels
                metrics[f'{category}_precision'] = self._calculate_precision(cat_true, cat_pred)
                metrics[f'{category}_recall'] = self._calculate_recall(cat_true, cat_pred)
                metrics[f'{category}_f1'] = self._calculate_f1(cat_true, cat_pred)
        
        logger.info(f"✅ Evaluation complete. Macro F1: {metrics['f1_macro']:.3f}")
        return metrics
    
    def _calculate_precision(self, y_true, y_pred, average=None):
        """Calculate precision score"""
        if average == 'macro':
            precisions = []
            for i in range(y_true.shape[1]):
                tp = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
                fp = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1))
                if tp + fp > 0:
                    precisions.append(tp / (tp + fp))
            return np.mean(precisions) if precisions else 0.0
        else:
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            return tp / (tp + fp) if tp + fp > 0 else 0.0
    
    def _calculate_recall(self, y_true, y_pred, average=None):
        """Calculate recall score"""
        if average == 'macro':
            recalls = []
            for i in range(y_true.shape[1]):
                tp = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
                fn = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0))
                if tp + fn > 0:
                    recalls.append(tp / (tp + fn))
            return np.mean(recalls) if recalls else 0.0
        else:
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            return tp / (tp + fn) if tp + fn > 0 else 0.0
    
    def _calculate_f1(self, y_true, y_pred, average=None):
        """Calculate F1 score"""
        precision = self._calculate_precision(y_true, y_pred, average)
        recall = self._calculate_recall(y_true, y_pred, average)
        
        if precision + recall > 0:
            return 2 * (precision * recall) / (precision + recall)
        return 0.0
    
    def visualize_category_distribution(self, 
                                      save_path: Optional[str] = None,
                                      show_plot: bool = True) -> go.Figure:
        """
        Create visualization of category distribution
        
        Args:
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Plotly figure object
        """
        if not self.classification_result:
            raise ValueError("No classification results available. Run classify_texts() first.")
        
        # Prepare data
        categories = list(self.classification_result.category_stats.keys())
        counts = list(self.classification_result.category_stats.values())
        
        # Create bar plot
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=counts,
                marker_color='lightblue',
                text=counts,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Fitness Category Distribution",
            xaxis_title="Categories",
            yaxis_title="Number of Discussions",
            template="plotly_white",
            height=500
        )
        
        # Rotate x-axis labels for better readability
        fig.update_xaxes(tickangle=45)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"📊 Visualization saved to {save_path}")
        
        if show_plot:
            fig.show()
        
        return fig
    
    def get_classification_statistics(self) -> Dict:
        """Get comprehensive statistics about classification results"""
        if not self.classification_result:
            raise ValueError("No classification results available. Run classify_texts() first.")
        
        result = self.classification_result
        
        stats = {
            "total_texts": len(result.predictions),
            "total_categories": len(self.fitness_categories),
            "categories_used": len(result.category_stats),
            "category_coverage": result.performance_metrics["coverage_rate"],
            "avg_confidence": result.performance_metrics["avg_confidence"],
            "multi_label_rate": result.performance_metrics["multi_label_rate"],
            "avg_labels_per_text": result.performance_metrics["avg_labels_per_text"],
            "threshold": result.threshold,
            "model_info": result.model_info,
            "top_categories": sorted(result.category_stats.items(), key=lambda x: x[1], reverse=True)[:5],
            "category_distribution": result.category_stats
        }
        
        return stats
    
    def save_results(self, save_path: str):
        """Save classification results to JSON file"""
        if not self.classification_result:
            raise ValueError("No classification results available. Run classify_texts() first.")
        
        # Convert results to JSON-serializable format
        results_dict = {
            "predictions": self.classification_result.predictions,
            "confidence_scores": self.classification_result.confidence_scores,
            "category_stats": self.classification_result.category_stats,
            "performance_metrics": self.classification_result.performance_metrics,
            "threshold": self.classification_result.threshold,
            "model_info": self.classification_result.model_info,
            "fitness_categories": self.fitness_categories,
            "statistics": self.get_classification_statistics()
        }
        
        with open(save_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"💾 Classification results saved to {save_path}")
    
    def _save_cached_results(self, cache_path: str):
        """Save classification state for fast loading"""
        cache_data = {
            'classification_result': self.classification_result,
            'fitness_categories': self.fitness_categories,
            'model_name': self.model_name,
            'confidence_threshold': self.confidence_threshold
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"💾 Cached classification state to {cache_path}")
    
    def _load_cached_results(self, cache_path: str) -> ClassificationResult:
        """Load cached classification results"""
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        self.classification_result = cache_data['classification_result']
        self.fitness_categories = cache_data['fitness_categories']
        
        logger.info("📂 Loaded cached classification results")
        return self.classification_result

# Example usage and testing
if __name__ == "__main__":
    print("🏷️ Content Classification System Initialized")
    print("Ready for fitness content categorization!")
