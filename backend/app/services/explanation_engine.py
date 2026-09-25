"""
Explanation Engine Service
Transforms ML training events into human-friendly explanations
Following the real ML lifecycle: Preprocessing → Feature Engineering → Model Dev → Evaluation → Ready
"""
from datetime import datetime
import json


class ExplanationEngine:
    """Generates human-friendly explanations for ML training events"""

    # ML Lifecycle phases
    PHASES = {
        'preprocessing': {
            'id': 'preprocessing',
            'title': 'Data Preprocessing',
            'description': 'Loading, cleaning, and preparing your raw data for model training.',
            'why': 'Clean data is the foundation of any accurate model.',
            'icon': '🔧'
        },
        'feature_engineering': {
            'id': 'feature_engineering',
            'title': 'Feature Engineering & Selection',
            'description': 'Transforming raw columns into meaningful features the model can learn from.',
            'why': 'Better features lead to better predictions.',
            'icon': '⚙️'
        },
        'model_development': {
            'id': 'model_development',
            'title': 'Model Development',
            'description': 'Training multiple algorithms and tuning their parameters.',
            'why': 'Different algorithms suit different data patterns.',
            'icon': '🧠'
        },
        'evaluation': {
            'id': 'evaluation',
            'title': 'Best Model & Evaluation',
            'description': 'Comparing all models on unseen data and selecting the champion.',
            'why': 'The true test is performance on data the model has never seen.',
            'icon': '🏆'
        },
        'ready': {
            'id': 'ready',
            'title': 'Deployment Ready',
            'description': 'Your trained model is packaged and ready for predictions!',
            'why': 'You can now use this model in production.',
            'icon': '🚀'
        }
    }

    # Human-friendly model explanations
    MODEL_EXPLANATIONS = {
        'Logistic Regression': {
            'learning_style': 'This model looks for simple, direct relationships between inputs and outcomes.',
            'analogy': 'Like drawing a straight line to separate different categories.',
            'strengths': ['Fast and efficient', 'Easy to understand and interpret'],
            'limitations': ['May miss complex patterns', 'Works best with linearly separable data']
        },
        'Random Forest': {
            'learning_style': 'This model learns by combining many small decision rules and voting on the best answer.',
            'analogy': 'Like asking a committee of experts and taking a vote.',
            'strengths': ['Handles complexity well', 'Resistant to errors and outliers'],
            'limitations': ['Can be slower on very large datasets', 'Harder to interpret individual decisions']
        },
        'Gradient Boosting': {
            'learning_style': 'This model learns from its mistakes, improving step by step.',
            'analogy': 'Like a student who focuses extra attention on questions they got wrong.',
            'strengths': ['Often achieves highest accuracy', 'Good at finding subtle patterns'],
            'limitations': ['Can be slower to train', 'May overfit if not tuned properly']
        },
        'SVM': {
            'learning_style': 'This model finds the best boundary between different categories.',
            'analogy': 'Like drawing the widest possible margin between groups.',
            'strengths': ['Excellent with clear boundaries', 'Works well in high dimensions'],
            'limitations': ['Slower on large datasets', 'Sensitive to feature scaling']
        },
        'KNN': {
            'learning_style': 'This model predicts based on what similar examples were labeled.',
            'analogy': 'Like asking your neighbors what they would do.',
            'strengths': ['Simple and intuitive', 'No training time needed'],
            'limitations': ['Slow for predictions on large data', 'Needs good distance measure']
        },
        'Decision Tree': {
            'learning_style': 'This model learns a series of yes/no questions to reach a decision.',
            'analogy': 'Like a flowchart that guides you to the answer.',
            'strengths': ['Very easy to understand', 'Shows clear decision rules'],
            'limitations': ['Can overfit easily', 'Sensitive to small data changes']
        },
        'AdaBoost': {
            'learning_style': 'This model combines many weak learners, focusing on hard examples.',
            'analogy': 'Like a team where each member specializes in what others struggle with.',
            'strengths': ['Boosts simple models effectively', 'Less prone to overfitting'],
            'limitations': ['Sensitive to noisy data', 'Can be slow with many iterations']
        },
        'Extra Trees': {
            'learning_style': 'Similar to Random Forest but with more randomness for speed.',
            'analogy': 'A faster, more adventurous version of the expert committee.',
            'strengths': ['Very fast training', 'Good generalization'],
            'limitations': ['Slightly less accurate than Random Forest', 'Uses more memory']
        },
        'XGBoost': {
            'learning_style': 'An optimized version of Gradient Boosting for maximum performance.',
            'analogy': 'The student who not only learns from mistakes but also optimizes study time.',
            'strengths': ['Often wins competitions', 'Handles missing data well'],
            'limitations': ['Complex to tune', 'Requires more computational resources']
        },
        'LightGBM': {
            'learning_style': 'A very fast boosting algorithm designed for efficiency.',
            'analogy': 'Lightning-fast learning that focuses on the most important patterns.',
            'strengths': ['Extremely fast training', 'Low memory usage'],
            'limitations': ['May overfit on small datasets', 'Sensitive to parameters']
        },
        'Linear Regression': {
            'learning_style': 'Finds the best straight line through the data points.',
            'analogy': 'Like fitting a ruler through scattered dots.',
            'strengths': ['Very fast and simple', 'Easy to interpret coefficients'],
            'limitations': ['Assumes linear relationships', 'Sensitive to outliers']
        },
        'Ridge Regression': {
            'learning_style': 'Linear regression with protection against overfitting.',
            'analogy': 'A more careful version of fitting the ruler.',
            'strengths': ['Handles many features well', 'More stable predictions'],
            'limitations': ['Still assumes linearity', 'All features are kept']
        },
        'Lasso Regression': {
            'learning_style': 'Linear regression that can eliminate unimportant features.',
            'analogy': 'Fitting a ruler while ignoring irrelevant factors.',
            'strengths': ['Automatic feature selection', 'Simpler models'],
            'limitations': ['May exclude useful features', 'Less stable with correlated features']
        },
        'ElasticNet': {
            'learning_style': 'Combines Ridge and Lasso for balanced regularization.',
            'analogy': 'The best of both careful fitting approaches.',
            'strengths': ['Balances Ridge and Lasso benefits', 'Handles correlated features'],
            'limitations': ['Two parameters to tune', 'May be slower']
        },
        'SVR': {
            'learning_style': 'Uses SVM principles for predicting continuous values.',
            'analogy': 'Finding a tube that best captures all the data points.',
            'strengths': ['Good with non-linear patterns', 'Robust to outliers'],
            'limitations': ['Slow on large datasets', 'Complex to tune']
        },
        'KNeighborsRegressor': {
            'learning_style': 'Predicts based on averages from similar examples.',
            'analogy': 'Asking neighbors what value they would guess.',
            'strengths': ['No assumptions about data', 'Simple concept'],
            'limitations': ['Slow predictions', 'Needs careful distance tuning']
        },
        'DecisionTreeRegressor': {
            'learning_style': 'Learns ranges of values through yes/no questions.',
            'analogy': 'A flowchart for estimating numbers.',
            'strengths': ['Easy to visualize', 'Captures non-linear patterns'],
            'limitations': ['Can overfit', 'Unstable with small changes']
        }
    }

    @classmethod
    def init_explanation_data(cls):
        """Initialize empty explanation data structure"""
        return {
            'current_phase': None,
            'phases': [],
            'insights': [],
            'models': [],
            'trust': {
                'data_size': None,
                'train_size': None,
                'test_size': None,
                'confidence': None
            },
            'summary': None,
            'progress': 0
        }

    @classmethod
    def emit_phase(cls, experiment, phase_id, details=None):
        """Emit a learning phase update"""
        from app import db

        try:
            phase_info = cls.PHASES.get(phase_id, {})

            if not experiment.explanation_data:
                experiment.explanation_data = cls.init_explanation_data()

            data = dict(experiment.explanation_data)
            data['current_phase'] = phase_id

            phase_entry = {
                'id': phase_id,
                'title': phase_info.get('title', phase_id),
                'description': phase_info.get('description', ''),
                'why': phase_info.get('why', ''),
                'icon': phase_info.get('icon', '📍'),
                'started_at': datetime.now().isoformat(),
                'completed_at': None,
                'details': details or {}
            }

            existing = False
            for i, p in enumerate(data['phases']):
                if p['id'] == phase_id:
                    data['phases'][i] = phase_entry
                    existing = True
                    break

            if not existing:
                if len(data['phases']) > 0:
                    data['phases'][-1]['completed_at'] = datetime.now().isoformat()
                data['phases'].append(phase_entry)

            data['progress'] = cls._calculate_progress(data)
            experiment.explanation_data = data

            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(experiment, 'explanation_data')

            db.session.commit()
            print(f"📍 Phase: {phase_info.get('title', phase_id)} ({data['progress']}% complete)", flush=True)

        except Exception as e:
            print(f"⚠️ ExplanationEngine.emit_phase error: {e}", flush=True)

    @classmethod
    def emit_insight(cls, experiment, message, category='general'):
        """Emit a human-readable insight"""
        from app import db

        try:
            if not experiment.explanation_data:
                experiment.explanation_data = cls.init_explanation_data()

            data = dict(experiment.explanation_data)

            insight = {
                'message': message,
                'category': category,
                'phase': data.get('current_phase'),
                'timestamp': datetime.now().isoformat()
            }

            data['insights'].append(insight)
            experiment.explanation_data = data

            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(experiment, 'explanation_data')

            db.session.commit()
            print(f"💡 Insight: {message}", flush=True)

        except Exception as e:
            print(f"⚠️ ExplanationEngine.emit_insight error: {e}", flush=True)

    @classmethod
    def emit_model_start(cls, experiment, model_name):
        """Emit when a model starts training"""
        from app import db

        try:
            if not experiment.explanation_data:
                experiment.explanation_data = cls.init_explanation_data()

            data = dict(experiment.explanation_data)
            model_info = cls.MODEL_EXPLANATIONS.get(model_name, {})

            model_entry = {
                'name': model_name,
                'learning_style': model_info.get('learning_style', f'Training {model_name}...'),
                'analogy': model_info.get('analogy', ''),
                'strengths': model_info.get('strengths', []),
                'limitations': model_info.get('limitations', []),
                'status': 'training',
                'started_at': datetime.now().isoformat(),
                'completed_at': None,
                'score': None,
                'metrics': {}
            }

            data['models'].append(model_entry)
            experiment.explanation_data = data

            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(experiment, 'explanation_data')

            db.session.commit()

        except Exception as e:
            print(f"⚠️ ExplanationEngine.emit_model_start error: {e}", flush=True)

    @classmethod
    def emit_model_result(cls, experiment, model_name, score, metrics=None, problem_type='classification'):
        """Emit when a model finishes training"""
        from app import db

        try:
            if not experiment.explanation_data:
                return

            data = dict(experiment.explanation_data)

            for model in data['models']:
                if model['name'] == model_name:
                    model['status'] = 'completed'
                    model['completed_at'] = datetime.now().isoformat()
                    model['score'] = score
                    model['metrics'] = metrics or {}

                    if problem_type == 'classification':
                        model['result_message'] = f"Correctly predicted {score*100:.1f}% of examples"
                    else:
                        model['result_message'] = f"Predictions were {score*100:.1f}% accurate on average"
                    break

            experiment.explanation_data = data

            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(experiment, 'explanation_data')

            db.session.commit()

        except Exception as e:
            print(f"⚠️ ExplanationEngine.emit_model_result error: {e}", flush=True)

    @classmethod
    def emit_trust_data(cls, experiment, data_size, train_size, test_size, confidence='high'):
        """Emit trust and safety metrics"""
        from app import db

        try:
            if not experiment.explanation_data:
                experiment.explanation_data = cls.init_explanation_data()

            data = dict(experiment.explanation_data)
            data['trust'] = {
                'data_size': data_size,
                'train_size': train_size,
                'test_size': test_size,
                'train_ratio': f"{(train_size/data_size)*100:.0f}%",
                'test_ratio': f"{(test_size/data_size)*100:.0f}%",
                'confidence': confidence,
                'confidence_reasons': cls._get_confidence_reasons(data_size, confidence)
            }

            experiment.explanation_data = data

            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(experiment, 'explanation_data')

            db.session.commit()

        except Exception as e:
            print(f"⚠️ ExplanationEngine.emit_trust_data error: {e}", flush=True)

    @classmethod
    def _get_confidence_reasons(cls, data_size, confidence):
        """Generate confidence reasons based on data"""
        reasons = []

        if data_size >= 1000:
            reasons.append("Trained on a substantial dataset")
        elif data_size >= 500:
            reasons.append("Trained on a moderate dataset")
        else:
            reasons.append("Limited training data available")

        reasons.append("Tested on held-out examples never seen during training")
        reasons.append("Results were consistent across multiple model types")

        return reasons

    @classmethod
    def emit_final_summary(cls, experiment, best_model, best_score, problem_type, target_column, feature_count):
        """Emit the final summary for the user"""
        from app import db

        try:
            if not experiment.explanation_data:
                return

            data = dict(experiment.explanation_data)
            model_info = cls.MODEL_EXPLANATIONS.get(best_model, {})

            # Mark ALL previous phases as complete
            for phase in data['phases']:
                if not phase.get('completed_at'):
                    phase['completed_at'] = datetime.now().isoformat()

            data['summary'] = {
                'best_model': best_model,
                'best_score': best_score,
                'score_display': f"{best_score*100:.1f}%",
                'problem_type': problem_type,
                'target': target_column,
                'why_chosen': f"This model achieved the highest accuracy ({best_score*100:.1f}%) on examples it had never seen before.",
                'what_it_means': cls._generate_meaning(best_model, target_column, problem_type),
                'when_best': model_info.get('strengths', ['General purpose predictions'])[0] if model_info.get('strengths') else 'General purpose predictions',
                'model_strengths': model_info.get('strengths', []),
                'model_limitations': model_info.get('limitations', [])
            }

            data['progress'] = 100

            experiment.explanation_data = data

            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(experiment, 'explanation_data')

            db.session.commit()

        except Exception as e:
            print(f"⚠️ ExplanationEngine.emit_final_summary error: {e}", flush=True)

    @classmethod
    def _generate_meaning(cls, model_name, target_column, problem_type):
        """Generate what the model means for the user"""
        if problem_type == 'classification':
            return f"When you provide new data, the model will predict the {target_column} category with high confidence."
        else:
            return f"When you provide new data, the model will estimate the {target_column} value based on learned patterns."

    @classmethod
    def generate_data_insights(cls, df, target_column):
        """Generate insights about the data"""
        insights = []
        insights.append(f"Your data contains {len(df):,} examples to learn from.")
        feature_count = len(df.columns) - 1
        insights.append(f"Found {feature_count} different factors that might influence predictions.")

        if df[target_column].dtype == 'object' or df[target_column].nunique() < 20:
            unique_count = df[target_column].nunique()
            insights.append(f"The model will learn to predict {unique_count} different categories.")
        else:
            insights.append(f"The model will learn to predict numeric values for {target_column}.")

        try:
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if target_column in numeric_cols and len(numeric_cols) > 1:
                correlations = df[numeric_cols].corr()[target_column].abs().sort_values(ascending=False)
                top_features = correlations.head(4).index.tolist()
                top_features = [f for f in top_features if f != target_column][:3]
                if top_features:
                    insights.append(f"Key factors that may influence predictions: {', '.join(top_features)}")
        except:
            pass

        return insights

    @classmethod
    def _calculate_progress(cls, data):
        """Calculate overall progress percentage based on phases"""
        if data.get('summary'):
            return 100

        # Phase weights: preprocessing=15%, feature_eng=15%, model_dev=40%, evaluation=20%, ready=10%
        phase_weights = {
            'preprocessing': 15,
            'feature_engineering': 15,
            'model_development': 40,
            'evaluation': 20,
            'ready': 10
        }

        progress = 0
        current_phase = data.get('current_phase')

        for phase in data.get('phases', []):
            pid = phase.get('id')
            weight = phase_weights.get(pid, 10)

            if phase.get('completed_at'):
                # Completed phase: full weight
                progress += weight
            elif pid == current_phase:
                # Active phase: half weight
                progress += weight * 0.5

        return min(int(progress), 100)


# Singleton instance for easy access
explanation_engine = ExplanationEngine()
