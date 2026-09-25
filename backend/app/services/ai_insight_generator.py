"""
AI Insight Generator Service
Uses Gemini AI to generate dataset-specific, contextual insights during training
"""
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from .gemini_service import get_ai_service


class AIInsightGenerator:
    """Generates dynamic, dataset-specific insights using AIService"""
    
    def __init__(self):
        self.ai_service = get_ai_service()
    
    def _call_ai(self, prompt: str, max_tokens: int = 200) -> str:
        """Central AI call using the hardened AIService"""
        try:
            # We use a generic system prompt for insights
            system_prompt = "You are a helpful data science assistant that provides concise insights."
            return self.ai_service.call_llm(system_prompt, prompt)
        except Exception as e:
            print(f"⚠️ AIInsightGenerator error: {e}", flush=True)
            return None
    
    def analyze_dataset(self, df: pd.DataFrame, target_column: str, goal_description: str = "") -> List[str]:
        """
        Analyze dataset and generate specific insights about patterns and relationships
        
        Returns list of 3-4 concise, dataset-specific insights
        """
        try:
            # Calculate key statistics
            stats = self._calculate_statistics(df, target_column)
            
            # Build prompt for Gemini
            prompt = f"""You are analyzing a machine learning dataset for a non-technical user.

Dataset has {len(df):,} rows and {len(df.columns)} columns.
Target column: "{target_column}"
User's goal: "{goal_description or 'predict the target column'}"

Column names: {', '.join(df.columns.tolist())}

Key findings:
{stats}

Generate 3-4 SHORT, SPECIFIC insights about this data that help the user understand:
1. What patterns exist in their data
2. Which factors seem most important  
3. Any interesting relationships discovered

RULES:
- Use actual column names from the data
- Be specific with numbers/thresholds when relevant
- Max 15 words per insight
- One insight per line
- No bullet points or numbering
- Focus on actionable patterns

Example good insights:
"Temperature above 25°C strongly correlates with summer crop varieties"
"Nitrogen levels between 80-100 are most common in wheat samples"
"Rainfall shows clear seasonal pattern with peaks in monsoon months"
"""
            
            response = self._call_ai(prompt, max_tokens=250)
            
            if response:
                # Parse insights (one per line)
                insights = [line.strip() for line in response.split('\n') if line.strip() and len(line.strip()) > 10]
                return insights[:4]  # Max 4 insights
            else:
                # Fallback to basic insights
                return self._generate_fallback_insights(df, target_column, stats)
                
        except Exception as e:
            print(f"⚠️ Error generating insights: {e}", flush=True)
            return self._generate_fallback_insights(df, target_column, {})
    
    def _calculate_statistics(self, df: pd.DataFrame, target_column: str) -> str:
        """Calculate key statistics for the dataset"""
        stats_lines = []
        
        try:
            # Target column analysis
            if target_column in df.columns:
                if df[target_column].dtype == 'object' or df[target_column].nunique() < 20:
                    value_counts = df[target_column].value_counts().head(5)
                    stats_lines.append(f"Target '{target_column}' distribution: {dict(value_counts)}")
                else:
                    stats_lines.append(f"Target '{target_column}' range: {df[target_column].min():.2f} to {df[target_column].max():.2f}")
            
            # Numeric columns correlations
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in numeric_cols and len(numeric_cols) > 1:
                correlations = df[numeric_cols].corr()[target_column].abs().sort_values(ascending=False)
                top_corr = correlations.head(4).to_dict()
                top_corr.pop(target_column, None)  # Remove self-correlation
                if top_corr:
                    stats_lines.append(f"Top correlations with target: {top_corr}")
            
            # Missing values
            missing = df.isnull().sum()
            if missing.sum() > 0:
                stats_lines.append(f"Missing values: {missing[missing > 0].to_dict()}")
            
        except Exception as e:
            print(f"⚠️ Error calculating statistics: {e}", flush=True)
        
        return '\n'.join(stats_lines) if stats_lines else "Basic dataset statistics calculated"
    
    def _generate_fallback_insights(self, df: pd.DataFrame, target_column: str, stats: Dict) -> List[str]:
        """Generate basic insights when AI is unavailable"""
        insights = []
        
        insights.append(f"Dataset contains {len(df):,} examples across {len(df.columns)} features")
        
        if target_column in df.columns:
            if df[target_column].dtype == 'object' or df[target_column].nunique() < 20:
                insights.append(f"Target '{target_column}' has {df[target_column].nunique()} unique categories")
            else:
                insights.append(f"Target '{target_column}' is continuous with values from {df[target_column].min():.1f} to {df[target_column].max():.1f}")
        
        # Find numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            insights.append(f"Found {len(numeric_cols)} numeric features for analysis")
        
        return insights[:3]
    
    def analyze_model_learning(self, model_name: str, feature_importances: Dict[str, float], score: float, target_column: str) -> str:
        """
        Generate insight about what a specific model learned
        
        Returns a single concise insight about the model's learning
        """
        try:
            # Get top 3 features
            sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_features[:3]
            
            prompt = f"""A {model_name} model just finished training.

Target: {target_column}
Accuracy: {score*100:.1f}%

Top important features:
{chr(10).join([f"- {feat}: {imp:.3f}" for feat, imp in top_features])}

Generate ONE SHORT insight (max 15 words) about what this model learned.
Use actual feature names. Be specific.

Example: "Model found temperature (importance 0.42) as the most decisive factor"
"""
            
            response = self._call_ai(prompt, max_tokens=50)
            
            if response:
                # Clean up response
                insight = response.split('\n')[0].strip()
                return insight if len(insight) < 100 else insight[:97] + "..."
            else:
                # Fallback
                top_feat, top_imp = top_features[0]
                return f"Model identified {top_feat} as most important (importance: {top_imp:.2f})"
                
        except Exception as e:
            print(f"⚠️ Error generating model insight: {e}", flush=True)
            return f"{model_name} completed training with {score*100:.1f}% accuracy"
    
    def explain_model_comparison(self, all_results: List[Dict], best_model_name: str, target_column: str) -> str:
        """
        Generate insight about why one model performed better
        
        Returns a single insight explaining the model selection
        """
        try:
            # Format results
            results_text = '\n'.join([
                f"- {r['model']}: {r['score']*100:.1f}%"
                for r in sorted(all_results, key=lambda x: x['score'], reverse=True)[:5]
            ])
            
            prompt = f"""Comparing ML models for predicting {target_column}:

{results_text}

Best model: {best_model_name}

Generate ONE SHORT insight (max 15 words) explaining why this model likely performed best.

Example: "Random Forest excelled by capturing complex non-linear relationships in the data"
"""
            
            response = self._call_ai(prompt, max_tokens=50)
            
            if response:
                insight = response.split('\n')[0].strip()
                return insight if len(insight) < 100 else insight[:97] + "..."
            else:
                # Fallback
                best_score = next(r['score'] for r in all_results if r['model'] == best_model_name)
                return f"{best_model_name} achieved highest accuracy at {best_score*100:.1f}%"
                
        except Exception as e:
            print(f"⚠️ Error generating comparison insight: {e}", flush=True)
            return f"{best_model_name} performed best among all tested models"


# Singleton instance
ai_insight_generator = AIInsightGenerator()
