"""
Image Classification Module
Supports: MobileNet, EfficientNet transfer learning
GPU: Automatically uses GPU if available, falls back to CPU.
"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import os
import tempfile

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, EfficientNetB1
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# ── GPU auto-configuration ──────────────────────────────────────
try:
    from ml_engine.utils.gpu_utils import configure_tensorflow_gpu, log_gpu_info
    if TENSORFLOW_AVAILABLE:
        configure_tensorflow_gpu()
except ImportError:
    pass  # utils not available (e.g. standalone use)


class ImageClassifier:
    """
    Image classification using transfer learning
    Supports MobileNetV2 and EfficientNet
    """
    
    MODELS = {
        'mobilenet': {
            'class': MobileNetV2 if TENSORFLOW_AVAILABLE else None,
            'input_size': (224, 224),
            'preprocess': 'tf'
        },
        'efficientnet_b0': {
            'class': EfficientNetB0 if TENSORFLOW_AVAILABLE else None,
            'input_size': (224, 224),
            'preprocess': 'tf'
        },
        'efficientnet_b1': {
            'class': EfficientNetB1 if TENSORFLOW_AVAILABLE else None,
            'input_size': (240, 240),
            'preprocess': 'tf'
        }
    }
    
    def __init__(
        self,
        model_name: str = 'mobilenet',
        num_classes: int = None,
        freeze_base: bool = True,
        dropout: float = 0.5,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32
    ):
        """
        Initialize image classifier
        
        Args:
            model_name: Base model ('mobilenet', 'efficientnet_b0', 'efficientnet_b1')
            num_classes: Number of output classes
            freeze_base: Freeze base model weights
            dropout: Dropout rate before output layer
            learning_rate: Learning rate
            epochs: Training epochs
            batch_size: Batch size
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required. Install with: pip install tensorflow")
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.freeze_base = freeze_base
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Fitted attributes
        self.model_ = None
        self.classes_ = []
        self.history_ = None
        self.input_size_ = self.MODELS[model_name]['input_size']
    
    def build_model(self, num_classes: int) -> 'Model':
        """
        Build transfer learning model
        
        Args:
            num_classes: Number of output classes
            
        Returns:
            Keras Model
        """
        model_config = self.MODELS[self.model_name]
        base_model_class = model_config['class']
        input_size = model_config['input_size']
        
        # Load base model without top layers
        base_model = base_model_class(
            weights='imagenet',
            include_top=False,
            input_shape=(*input_size, 3)
        )
        
        # Freeze base model if specified
        base_model.trainable = not self.freeze_base
        
        # Build top layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(self.dropout)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(self.dropout / 2)(x)
        
        # Output layer
        if num_classes == 2:
            output = Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
        else:
            output = Dense(num_classes, activation='softmax')(x)
            loss = 'sparse_categorical_crossentropy'
        
        # Create model
        model = Model(inputs=base_model.input, outputs=output)
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=loss,
            metrics=['accuracy']
        )
        
        return model
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2,
        class_names: List[str] = None,
        augment: bool = True,
        verbose: int = 1
    ) -> 'ImageClassifier':
        """
        Train the classifier
        
        Args:
            X: Image array (N, H, W, 3)
            y: Labels array
            validation_split: Validation fraction
            class_names: List of class names
            augment: Apply data augmentation
            verbose: Verbosity level
            
        Returns:
            self
        """
        # Log GPU status at training start
        try:
            log_gpu_info()
        except Exception:
            pass

        # Determine number of classes
        if self.num_classes is None:
            self.num_classes = len(np.unique(y))
        
        self.classes_ = class_names or [str(i) for i in range(self.num_classes)]
        
        # Build model
        self.model_ = self.build_model(self.num_classes)
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Data augmentation
        if augment:
            datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2
            )
            datagen.fit(X_train)
            train_generator = datagen.flow(X_train, y_train, batch_size=self.batch_size)
        else:
            train_generator = None
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train
        if augment:
            self.history_ = self.model_.fit(
                train_generator,
                steps_per_epoch=len(X_train) // self.batch_size,
                epochs=self.epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=verbose
            )
        else:
            self.history_ = self.model_.fit(
                X_train, y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=verbose
            )
        
        return self
    
    def fit_from_directory(
        self,
        train_dir: str,
        validation_dir: str = None,
        validation_split: float = 0.2,
        augment: bool = True,
        verbose: int = 1
    ) -> 'ImageClassifier':
        """
        Train from directory structure
        
        Args:
            train_dir: Training directory (with class subfolders)
            validation_dir: Validation directory (optional)
            validation_split: Validation fraction if no val dir
            augment: Apply augmentation
            verbose: Verbosity level
            
        Returns:
            self
        """
        # Log GPU status at training start
        try:
            log_gpu_info()
        except Exception:
            pass

        # Get classes from directory
        self.classes_ = sorted(os.listdir(train_dir))
        self.num_classes = len(self.classes_)
        
        # Build model
        self.model_ = self.build_model(self.num_classes)
        
        # Data generators
        if augment:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                validation_split=validation_split if validation_dir is None else 0
            )
        else:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=validation_split if validation_dir is None else 0
            )
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_size_,
            batch_size=self.batch_size,
            class_mode='sparse',
            subset='training' if validation_dir is None else None
        )
        
        if validation_dir:
            val_datagen = ImageDataGenerator(rescale=1./255)
            val_generator = val_datagen.flow_from_directory(
                validation_dir,
                target_size=self.input_size_,
                batch_size=self.batch_size,
                class_mode='sparse'
            )
        else:
            val_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=self.input_size_,
                batch_size=self.batch_size,
                class_mode='sparse',
                subset='validation'
            )
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5)
        ]
        
        # Train
        self.history_ = self.model_.fit(
            train_generator,
            epochs=self.epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels
        
        Args:
            X: Image array
            
        Returns:
            Predicted labels
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        probs = self.model_.predict(X, verbose=0)
        
        if self.num_classes == 2:
            return (probs > 0.5).astype(int).flatten()
        else:
            return np.argmax(probs, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            X: Image array
            
        Returns:
            Probability array
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        probs = self.model_.predict(X, verbose=0)
        
        if self.num_classes == 2:
            return np.column_stack([1 - probs, probs])
        else:
            return probs
    
    def predict_with_label(self, X: np.ndarray) -> List[Dict[str, Any]]:
        """
        Predict with class names and confidence
        
        Args:
            X: Image array
            
        Returns:
            List of prediction dictionaries
        """
        probs = self.predict_proba(X)
        predictions = []
        
        for prob in probs:
            idx = np.argmax(prob)
            predictions.append({
                'class': self.classes_[idx],
                'confidence': float(prob[idx]),
                'all_probabilities': {
                    cls: float(p) for cls, p in zip(self.classes_, prob)
                }
            })
        
        return predictions
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model
        
        Args:
            X: Test images
            y: True labels
            
        Returns:
            Dictionary of metrics
        """
        loss, accuracy = self.model_.evaluate(X, y, verbose=0)
        
        # Additional metrics
        y_pred = self.predict(X)
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'precision': float(precision_score(y, y_pred, average='weighted')),
            'recall': float(recall_score(y, y_pred, average='weighted')),
            'f1': float(f1_score(y, y_pred, average='weighted'))
        }
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history"""
        if self.history_ is None:
            return {}
        
        return {k: [float(v) for v in vals] for k, vals in self.history_.history.items()}
    
    def fine_tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        unfreeze_layers: int = 20,
        epochs: int = 10,
        learning_rate: float = 0.0001
    ) -> 'ImageClassifier':
        """
        Fine-tune by unfreezing top layers
        
        Args:
            X: Training images
            y: Labels
            unfreeze_layers: Number of layers to unfreeze
            epochs: Fine-tuning epochs
            learning_rate: Lower learning rate
            
        Returns:
            self
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Unfreeze top layers
        for layer in self.model_.layers[-unfreeze_layers:]:
            layer.trainable = True
        
        # Recompile with lower learning rate
        self.model_.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=self.model_.loss,
            metrics=['accuracy']
        )
        
        # Continue training
        self.model_.fit(
            X, y,
            batch_size=self.batch_size,
            epochs=epochs,
            verbose=1
        )
        
        return self
    
    def save(self, path: str):
        """Save model"""
        print(f"DEBUG: ImageClassifier.save called with path={path}", flush=True)
        print(f"DEBUG: self.model_ type: {type(self.model_)}", flush=True)
        
        if self.model_ is not None:
            try:
                self.model_.save(path)
                print("DEBUG: self.model_.save() successful", flush=True)
            except AttributeError:
                print("DEBUG: .save() method missing, trying keras.models.save_model", flush=True)
                try:
                    from tensorflow.keras.models import save_model
                    save_model(self.model_, path)
                    print("DEBUG: keras.models.save_model() successful", flush=True)
                except Exception as e:
                    print(f"DEBUG: Fallback save failed: {e}", flush=True)
                    raise e
            except Exception as e:
                print(f"DEBUG: self.model_.save() failed: {e}", flush=True)
                raise e
            
            # Save metadata
            import json
            if path.endswith('.h5'):
                meta_path = path.replace('.h5', '_meta.json')
            elif path.endswith('.keras'):
                meta_path = path.replace('.keras', '_meta.json')
            else:
                meta_path = path + '_meta.json'
            
            print(f"DEBUG: Saving metadata to {meta_path}", flush=True)
            with open(meta_path, 'w') as f:
                json.dump({
                    'classes': self.classes_,
                    'num_classes': self.num_classes,
                    'model_name': self.model_name,
                    'input_size': self.input_size_
                }, f)
    
    @classmethod
    def load(cls, path: str) -> 'ImageClassifier':
        """Load model"""
        import json
        
        classifier = cls()
        classifier.model_ = keras.models.load_model(path)
        
        # Load metadata
        meta_path = path.replace('.h5', '_meta.json').replace('.keras', '_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            classifier.classes_ = meta['classes']
            classifier.num_classes = meta['num_classes']
            classifier.model_name = meta.get('model_name', 'mobilenet')
            classifier.input_size_ = tuple(meta.get('input_size', (224, 224)))
        
        return classifier
