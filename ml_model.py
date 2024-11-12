from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import joblib
import logging
from typing import List, Union, Optional, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessClassifier:
    """Text classification model using TF-IDF and Random Forest."""
    
    def __init__(self, max_features: int = 5000, n_estimators: int = 100):
        """Initialize the classifier with customizable parameters.
        
        Args:
            max_features: Maximum number of features for TF-IDF
            n_estimators: Number of trees in Random Forest
        """
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2)
            )),
            ('clf', RandomForestClassifier(
                n_estimators=n_estimators,
                n_jobs=-1,
                random_state=42
            ))
        ])
        self.is_trained = False
        self.model_metrics: Dict[str, Any] = {}
        
    def train_model(self, texts: List[str], labels: List[str], test_size: float = 0.2) -> Dict[str, Any]:
        """Train the model and evaluate performance.
        
        Args:
            texts: List of text documents
            labels: List of corresponding labels
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary containing model metrics
        """
        try:
            # Perform train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=test_size, random_state=42, stratify=labels
            )
            
            # Train model
            logger.info("Training model...")
            self.pipeline.fit(X_train, y_train)
            self.is_trained = True
            
            # Evaluate model
            train_score = self.pipeline.score(X_train, y_train)
            test_score = self.pipeline.score(X_test, y_test)
            cv_scores = cross_val_score(self.pipeline, texts, labels, cv=5)
            
            # Get predictions for detailed metrics
            y_pred = self.pipeline.predict(X_test)
            
            # Store metrics
            self.model_metrics = {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'cv_scores_mean': cv_scores.mean(),
                'cv_scores_std': cv_scores.std(),
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            logger.info(f"Model trained successfully. Test accuracy: {test_score:.2f}")
            return self.model_metrics
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
            
    def predict_complexity(self, text: Union[str, List[str]]) -> np.ndarray:
        """Predict complexity for single text or multiple texts.
        
        Args:
            text: Input text or list of texts
            
        Returns:
            Predicted label(s)
        """
        if not self.is_trained:
            raise RuntimeError("Model needs to be trained before making predictions")
            
        try:
            if isinstance(text, str):
                text = [text]
            return self.pipeline.predict(text)
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
            
    def save_model(self, filepath: Union[str, Path]) -> None:
        """Save trained model to disk.
        
        Args:
            filepath: Path to save model file
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
            
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.pipeline, filepath)
            logger.info(f"Model saved successfully to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
            
    def load_model(self, filepath: Union[str, Path]) -> None:
        """Load trained model from disk.
        
        Args:
            filepath: Path to model file
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                raise FileNotFoundError(f"Model file not found: {filepath}")
                
            self.pipeline = joblib.load(filepath)
            self.is_trained = True
            logger.info(f"Model loaded successfully from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """Get most important features from the model.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary of feature names and importance scores
        """
        if not self.is_trained:
            raise RuntimeError("Model needs to be trained first")
            
        try:
            feature_names = self.pipeline.named_steps['tfidf'].get_feature_names_out()
            importances = self.pipeline.named_steps['clf'].feature_importances_
            
            # Sort features by importance
            indices = np.argsort(importances)[::-1][:top_n]
            return {feature_names[i]: importances[i] for i in indices}
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            raise