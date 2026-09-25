import os
import io
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc

class ModelVisualizer:
    """Generates evaluation visualization graphs for trained models"""
    
    @staticmethod
    def generate_classification_graphs(model, X_test, y_test, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        paths = []
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
        # 1. Confusion Matrix
        try:
            from sklearn.metrics import ConfusionMatrixDisplay
            fig, ax = plt.subplots(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax, cmap='Blues', values_format='d')
            plt.title('Confusion Matrix')
            path = os.path.join(output_dir, 'confusion_matrix.png')
            plt.savefig(path, bbox_inches='tight', dpi=100)
            plt.close(fig)
            paths.append(path)
        except Exception as e:
            print("Error generating Confusion Matrix:", e)
            
        # 2. Feature Importance
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[-10:] # Top 10 features
                
                features = X_test.columns if hasattr(X_test, 'columns') else [f'Feature {i}' for i in range(len(importances))]
                top_features = [features[i] for i in indices]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(range(len(indices)), importances[indices], align='center', color='skyblue')
                ax.set_yticks(range(len(indices)))
                ax.set_yticklabels(top_features)
                ax.set_title('Top 10 Feature Importances')
                path = os.path.join(output_dir, 'feature_importance.png')
                plt.savefig(path, bbox_inches='tight', dpi=100)
                plt.close(fig)
                paths.append(path)
        except Exception as e:
            print("Error generating Feature Importance:", e)
            
        # 3. ROC Curve
        try:
            if y_proba is not None and len(np.unique(y_test)) == 2:
                # Binary classification
                fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic (ROC)')
                ax.legend(loc="lower right")
                
                path = os.path.join(output_dir, 'roc_curve.png')
                plt.savefig(path, bbox_inches='tight', dpi=100)
                plt.close(fig)
                paths.append(path)
        except Exception as e:
            print("Error generating ROC Curve:", e)
            
        return paths

    @staticmethod
    def generate_regression_graphs(model, X_test, y_test, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        paths = []
        
        y_pred = model.predict(X_test)
        
        # 1. Actual vs Predicted
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test, y_pred, alpha=0.5, color='royalblue')
            
            # Perfect prediction line
            min_val = min(min(y_test), min(y_pred))
            max_val = max(max(y_test), max(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title('Actual vs Predicted')
            ax.legend()
            
            path = os.path.join(output_dir, 'actual_vs_predicted.png')
            plt.savefig(path, bbox_inches='tight', dpi=100)
            plt.close(fig)
            paths.append(path)
        except Exception as e:
            print("Error generating Actual vs Predicted:", e)
            
        # 2. Residuals Plot
        try:
            residuals = y_test - y_pred
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_pred, residuals, alpha=0.5, color='seagreen')
            ax.axhline(y=0, color='r', linestyle='--', lw=2)
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Residuals')
            ax.set_title('Residual Plot (Error Distribution)')
            
            path = os.path.join(output_dir, 'residual_plot.png')
            plt.savefig(path, bbox_inches='tight', dpi=100)
            plt.close(fig)
            paths.append(path)
        except Exception as e:
            print("Error generating Residuals Plot:", e)
            
        # 3. Feature Importance
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[-10:]
                
                features = X_test.columns if hasattr(X_test, 'columns') else [f'Feature {i}' for i in range(len(importances))]
                top_features = [features[i] for i in indices]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(range(len(indices)), importances[indices], align='center', color='skyblue')
                ax.set_yticks(range(len(indices)))
                ax.set_yticklabels(top_features)
                ax.set_title('Top 10 Feature Importances')
                path = os.path.join(output_dir, 'feature_importance.png')
                plt.savefig(path, bbox_inches='tight', dpi=100)
                plt.close(fig)
                paths.append(path)
        except Exception as e:
            pass
            
        return paths
