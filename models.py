# models.py
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2, EfficientNetB4, DenseNet201
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficient_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as dense_preprocess
import logging
from config import MODEL_CONFIG

class ModelManager:
    def __init__(self):
        self.models = {}
        self.preprocess_functions = {}
        self._initialize_models()

    def _initialize_models(self):
        """Initialize all required models"""
        model_mapping = {
            'ResNet50V2': ResNet50V2,
            'EfficientNetB4': EfficientNetB4,
            'DenseNet201': DenseNet201
        }
        
        preprocess_mapping = {
            'ResNet50V2': resnet_preprocess,
            'EfficientNetB4': efficient_preprocess,
            'DenseNet201': dense_preprocess
        }

        try:
            for model_key, config in MODEL_CONFIG.items():
                model_class = model_mapping[config['name']]
                self.models[model_key] = {
                    'model': model_class(weights='imagenet'),
                    'size': config['size']
                }
                self.preprocess_functions[model_key] = preprocess_mapping[config['name']]
            
            logging.info("Successfully loaded all models")
        except Exception as e:
            logging.error(f"Failed to load models: {str(e)}")
            raise

    def get_model(self, model_key):
        """Get model and its preprocessing function"""
        if model_key not in self.models:
            raise KeyError(f"Model {model_key} not found")
        
        return (
            self.models[model_key]['model'],
            self.models[model_key]['size'],
            self.preprocess_functions[model_key]
        )

    def get_all_models(self):
        """Get all models and their configurations"""
        return self.models