"""
Tabular Clustering Module
Supports: K-Means, DBSCAN
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import joblib


class TabularClusterer:
    """AutoML for tabular clustering tasks"""
    
    def __init__(self):
        self.trained_models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = float('-inf')
        self.scaler = StandardScaler()
        self.results = []
    
    def train(
        self,
        X: pd.DataFrame,
        n_clusters_range: tuple = (2, 10),
        include_dbscan: bool = True
    ) -> Dict[str, Any]:
        """Train clustering models and find optimal clusters"""
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        self.results = []
        
        # Try different numbers of clusters with K-Means
        for n_clusters in range(n_clusters_range[0], n_clusters_range[1] + 1):
            try:
                result = self._train_kmeans(X_scaled, n_clusters)
                self.results.append(result)
                
                if result['silhouette_score'] > self.best_score:
                    self.best_score = result['silhouette_score']
                    self.best_model = self.trained_models[f'kmeans_{n_clusters}']
                    self.best_model_name = f'kmeans_{n_clusters}'
                    
            except Exception as e:
                self.results.append({
                    'model_name': f'kmeans_{n_clusters}',
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Try DBSCAN
        if include_dbscan:
            try:
                result = self._train_dbscan(X_scaled)
                self.results.append(result)
                
                if result.get('silhouette_score', -1) > self.best_score:
                    self.best_score = result['silhouette_score']
                    self.best_model = self.trained_models['dbscan']
                    self.best_model_name = 'dbscan'
                    
            except Exception as e:
                self.results.append({
                    'model_name': 'dbscan',
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Find the n_clusters for the best model to return
        best_n_clusters = None
        for res in self.results:
            if res.get('model_name') == self.best_model_name:
                best_n_clusters = res.get('n_clusters')
                break
                
        return {
            'results': self.results,
            'best_model': self.best_model_name,
            'best_score': self.best_score,
            'n_clusters': best_n_clusters
        }
    
    def _train_kmeans(self, X_scaled: np.ndarray, n_clusters: int) -> Dict[str, Any]:
        """Train K-Means with specific number of clusters"""
        
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        
        model_name = f'kmeans_{n_clusters}'
        self.trained_models[model_name] = model
        
        # Compute metrics
        metrics = self._compute_clustering_metrics(X_scaled, labels)
        
        return {
            'model_name': model_name,
            'status': 'completed',
            'n_clusters': n_clusters,
            'inertia': model.inertia_,
            **metrics
        }
    
    def _train_dbscan(self, X_scaled: np.ndarray) -> Dict[str, Any]:
        """Train DBSCAN clustering"""
        
        model = DBSCAN(eps=0.5, min_samples=5)
        labels = model.fit_predict(X_scaled)
        
        self.trained_models['dbscan'] = model
        
        # Get number of clusters (excluding noise points labeled as -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # Compute metrics (only if we have meaningful clusters)
        if n_clusters >= 2:
            metrics = self._compute_clustering_metrics(X_scaled, labels)
        else:
            metrics = {'silhouette_score': -1, 'calinski_harabasz': 0, 'davies_bouldin': float('inf')}
        
        return {
            'model_name': 'dbscan',
            'status': 'completed',
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            **metrics
        }
    
    def _compute_clustering_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute clustering quality metrics"""
        
        # Filter out noise points for metric calculation
        mask = labels != -1
        if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
            return {
                'silhouette_score': -1,
                'calinski_harabasz': 0,
                'davies_bouldin': float('inf')
            }
        
        return {
            'silhouette_score': silhouette_score(X[mask], labels[mask]),
            'calinski_harabasz': calinski_harabasz_score(X[mask], labels[mask]),
            'davies_bouldin': davies_bouldin_score(X[mask], labels[mask])
        }
    
    def predict(self, X: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """Predict cluster labels for new data"""
        model = self.trained_models.get(model_name) if model_name else self.best_model
        if model is None:
            raise ValueError("No model trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        return model.predict(X_scaled) if hasattr(model, 'predict') else model.fit_predict(X_scaled)
    
    def save_model(self, path: str, model_name: Optional[str] = None):
        """Save model and scaler to file"""
        model = self.trained_models.get(model_name) if model_name else self.best_model
        if model is None:
            raise ValueError("No model to save. Call train() first.")
        
        package = {
            'model': model,
            'scaler': self.scaler
        }
        joblib.dump(package, path)
    
    def load_model(self, path: str):
        """Load model and scaler from file"""
        package = joblib.load(path)
        self.best_model = package['model']
        self.scaler = package['scaler']
        return self.best_model
