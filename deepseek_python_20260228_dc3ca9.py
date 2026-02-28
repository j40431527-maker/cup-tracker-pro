# src/core/adaptive_learner.py

import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict
import pickle
from pathlib import Path
import asyncio
from dataclasses import dataclass
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)

@dataclass
class GamePattern:
    """Represents a learned game pattern"""
    pattern_id: str
    cup_count: int
    trajectory: List[List[Tuple[int, int]]]
    duration: float
    complexity: float
    outcome: str
    features: np.ndarray

class AdaptiveLearner:
    """Self-learning system that adapts to game patterns"""
    
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Pattern database
        self.patterns: Dict[str, List[GamePattern]] = defaultdict(list)
        self.current_pattern = []
        
        # Machine learning models
        self.pattern_classifier = RandomForestClassifier(n_estimators=100)
        self.scaler = StandardScaler()
        self.models_trained = False
        
        # Performance metrics
        self.accuracy_history = []
        self.adaptation_rate = 0.1
        
        # Load existing models
        self._load_models()
        
    def start_pattern_recording(self):
        """Start recording a new pattern"""
        self.current_pattern = []
        logger.debug("Started pattern recording")
        
    def record_frame(self, cup_positions: List[Tuple[int, int]], timestamp: float):
        """Record cup positions for current pattern"""
        self.current_pattern.append({
            'positions': cup_positions.copy(),
            'timestamp': timestamp,
            'features': self._extract_features(cup_positions)
        })
        
    def end_pattern_recording(self, outcome: str) -> Optional[GamePattern]:
        """End recording and learn from the pattern"""
        if len(self.current_pattern) < 2:
            return None
            
        # Extract pattern features
        pattern = GamePattern(
            pattern_id=self._generate_pattern_id(),
            cup_count=len(self.current_pattern[0]['positions']),
            trajectory=[frame['positions'] for frame in self.current_pattern],
            duration=self.current_pattern[-1]['timestamp'] - self.current_pattern[0]['timestamp'],
            complexity=self._calculate_complexity(),
            outcome=outcome,
            features=self._aggregate_features()
        )
        
        # Store pattern
        self.patterns[outcome].append(pattern)
        
        # Trigger model update
        asyncio.create_task(self._update_models())
        
        return pattern
    
    def predict_pattern(self, initial_positions: List[Tuple[int, int]]) -> Dict[str, float]:
        """Predict outcome based on initial positions"""
        if not self.models_trained or len(self.patterns) < 10:
            return {'unknown': 1.0}
            
        # Extract features
        features = self._extract_features(initial_positions)
        features_scaled = self.scaler.transform([features])
        
        # Predict
        probabilities = self.pattern_classifier.predict_proba(features_scaled)[0]
        classes = self.pattern_classifier.classes_
        
        return dict(zip(classes, probabilities))
    
    def adapt_tracker_parameters(self, performance_metrics: Dict[str, float]):
        """Adapt tracking parameters based on performance"""
        self.accuracy_history.append(performance_metrics.get('accuracy', 0))
        
        if len(self.accuracy_history) > 10:
            trend = np.polyfit(range(10), self.accuracy_history[-10:], 1)[0]
            
            # Adjust adaptation rate based on trend
            if trend < -0.05:  # Performance degrading
                self.adaptation_rate *= 1.2
            elif trend > 0.05:  # Performance improving
                self.adaptation_rate *= 0.9
            
            self.adaptation_rate = np.clip(self.adaptation_rate, 0.01, 0.5)
            
    async def _update_models(self):
        """Update ML models with new patterns"""
        if len(self.patterns) < 10:
            return
            
        # Prepare training data
        X = []
        y = []
        
        for outcome, patterns in self.patterns.items():
            for pattern in patterns[-50:]:  # Use last 50 patterns per outcome
                X.append(pattern.features)
                y.append(outcome)
        
        if len(X) < 10:
            return
            
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train classifier
        self.pattern_classifier.fit(X_scaled, y)
        self.models_trained = True
        
        # Save models
        self._save_models()
        
        logger.info(f"Models updated with {len(X)} patterns")
    
    def _extract_features(self, positions: List[Tuple[int, int]]) -> np.ndarray:
        """Extract features from cup positions"""
        features = []
        
        if not positions:
            return np.zeros(10)
        
        # Convert to numpy for calculations
        pos_array = np.array(positions)
        
        # Basic features
        features.append(len(positions))  # Number of cups
        features.append(np.mean(pos_array[:, 0]))  # Mean X
        features.append(np.mean(pos_array[:, 1]))  # Mean Y
        features.append(np.std(pos_array[:, 0]))  # Std X
        features.append(np.std(pos_array[:, 1]))  # Std Y
        
        # Distances between cups
        if len(positions) > 1:
            distances = []
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    dist = np.sqrt((positions[i][0] - positions[j][0])**2 +
                                  (positions[i][1] - positions[j][1])**2)
                    distances.append(dist)
            features.append(np.mean(distances))
            features.append(np.std(distances))
        else:
            features.extend([0, 0])
        
        # Spatial arrangement (rough estimate)
        angles = []
        if len(positions) > 2:
            center = np.mean(pos_array, axis=0)
            for pos in positions:
                angle = np.arctan2(pos[1] - center[1], pos[0] - center[0])
                angles.append(angle)
            features.append(np.std(angles))
        else:
            features.append(0)
        
        return np.array(features)
    
    def _aggregate_features(self) -> np.ndarray:
        """Aggregate features from recorded pattern"""
        if not self.current_pattern:
            return np.zeros(10)
        
        # Average features across frames
        all_features = [frame['features'] for frame in self.current_pattern]
        return np.mean(all_features, axis=0)
    
    def _calculate_complexity(self) -> float:
        """Calculate pattern complexity score"""
        if len(self.current_pattern) < 2:
            return 0.0
        
        # Calculate total movement
        total_movement = 0
        positions = [frame['positions'] for frame in self.current_pattern]
        
        for i in range(1, len(positions)):
            for j in range(len(positions[i])):
                if j < len(positions[i-1]):
                    dx = positions[i][j][0] - positions[i-1][j][0]
                    dy = positions[i][j][1] - positions[i-1][j][1]
                    total_movement += np.sqrt(dx*dx + dy*dy)
        
        # Normalize by duration
        duration = self.current_pattern[-1]['timestamp'] - self.current_pattern[0]['timestamp']
        if duration > 0:
            return total_movement / duration
        return 0.0
    
    def _generate_pattern_id(self) -> str:
        """Generate unique pattern ID"""
        import hashlib
        import time
        hash_input = f"{time.time_ns()}{np.random.random()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def _save_models(self):
        """Save trained models to disk"""
        model_path = self.model_dir / 'pattern_classifier.joblib'
        scaler_path = self.model_dir / 'scaler.joblib'
        
        joblib.dump(self.pattern_classifier, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"Models saved to {self.model_dir}")
    
    def _load_models(self):
        """Load trained models from disk"""
        model_path = self.model_dir / 'pattern_classifier.joblib'
        scaler_path = self.model_dir / 'scaler.joblib'
        
        if model_path.exists() and scaler_path.exists():
            self.pattern_classifier = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.models_trained = True
            logger.info(f"Models loaded from {self.model_dir}")