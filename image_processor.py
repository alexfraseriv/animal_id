# image_processor.py
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import decode_predictions
import numpy as np
from PIL import Image
import piexif
import logging
from datetime import datetime
import shutil
from pathlib import Path
import os

from config import (
    SUPPORTED_EXTENSIONS, 
    ANIMAL_CATEGORIES, 
    LANDSCAPE_CATEGORIES, 
    CONFIDENCE_THRESHOLDS
)
from models import ModelManager

class SafariImageProcessor:
    def __init__(self, source_dir, backup_dir=None):
        """Initialize the image processor"""
        self.source_dir = Path(source_dir)
        self.backup_dir = Path(backup_dir) if backup_dir else Path(source_dir) / 'backup'
        self.processed_count = 0
        
        # Create backup directory
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model manager
        self.model_manager = ModelManager()
        
        # Load categories
        self.animal_categories = ANIMAL_CATEGORIES
        self.landscape_categories = LANDSCAPE_CATEGORIES
        
    def _predict_animal(self, img_path):
        """Predict animal using ensemble of models"""
        try:
            def preprocess_image(img_path, size):
                img = image.load_img(img_path, target_size=size)
                x = image.img_to_array(img)
                
                # Create augmented versions
                augmented = []
                augmented.append(x)  # Original
                augmented.append(tf.image.flip_left_right(x).numpy())  # Flipped
                augmented.append(tf.image.adjust_brightness(x, delta=0.2).numpy())  # Brightened
                augmented.append(tf.image.adjust_contrast(x, contrast_factor=1.2).numpy())  # Contrasted
                
                return np.stack(augmented)

            predictions_combined = []
            
            # Process with each model
            for model_name in ['resnet', 'efficient', 'dense']:
                model, size, preprocess = self.model_manager.get_model(model_name)
                
                images = preprocess_image(img_path, size)
                x = preprocess(images)
                
                preds = model.predict(x, verbose=0)
                for pred in preds:
                    predictions = decode_predictions(np.expand_dims(pred, 0), top=10)[0]
                    predictions_combined.extend(predictions)

            # Aggregate predictions
            pred_scores = {}
            for _, label, conf in predictions_combined:
                for category, similar_labels in self.animal_categories.items():
                    if any(similar in label.lower() for similar in similar_labels):
                        if category not in pred_scores:
                            pred_scores[category] = []
                        pred_scores[category].append(conf)

            # Calculate final scores
            final_predictions = {}
            for category, scores in pred_scores.items():
                top_scores = sorted(scores, reverse=True)[:3]  # Top 3 scores
                if top_scores:
                    final_predictions[category] = np.mean(top_scores)

            # Get best prediction
            if final_predictions:
                best_category = max(final_predictions.items(), key=lambda x: x[1])
                return best_category[0], float(best_category[1])

            return None, 0.0

        except Exception as e:
            logging.error(f"Error predicting animal in {img_path}: {str(e)}")
            return None, 0.0

    def _predict_landscape(self, img_path):
        """Predict landscape features"""
        try:
            model, size, preprocess = self.model_manager.get_model('landscape')
            
            # Load and preprocess image
            img = image.load_img(img_path, target_size=size)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess(x)
            
            # Get predictions
            preds = model.predict(x, verbose=0)
            predictions = decode_predictions(preds, top=15)[0]  # Get more predictions for landscape
            
            # Initialize landscape features
            detected_features = []
            
            # Check predictions against landscape categories
            for pred_class, label, confidence in predictions:
                label = label.lower()
                for category, keywords in self.landscape_categories.items():
                    if any(keyword in label for keyword in keywords):
                        if (confidence > CONFIDENCE_THRESHOLDS['landscape'] and 
                            category not in detected_features):
                            detected_features.append(category)
            
            return detected_features

        except Exception as e:
            logging.error(f"Error predicting landscape for {img_path}: {str(e)}")
            return []

    def _generate_new_filename(self, original_path, animal_name, landscape_features):
        """Generate new filename with animal and landscape information"""
        self.processed_count += 1
        timestamp = datetime.now().strftime("%Y%m%d")
        sequence = f"{self.processed_count:04d}"
        extension = original_path.suffix.lower()
        
        # Format landscape features string
        landscape_str = "_".join(landscape_features) if landscape_features else "general"
        
        # Combine elements
        new_name = f"{animal_name}_{landscape_str}_{timestamp}_{sequence}{extension}"
        return original_path.parent / new_name

    def _add_metadata_tags(self, img_path, animal_name, confidence, landscape_features):
        """Add metadata including landscape information"""
        try:
            # Load existing EXIF data
            exif_dict = piexif.load(str(img_path))
            
            # Format landscape information
            landscape_str = ", ".join(landscape_features) if landscape_features else "general"
            
            # Create detailed comment
            comment = (
                f"Animal: {animal_name}\n"
                f"Confidence: {confidence:.2f}\n"
                f"Landscape Features: {landscape_str}\n"
                f"Processed: {datetime.now().isoformat()}"
            )
            user_comment = piexif.helper.UserComment.create(comment)
            
            # Update EXIF
            if "Exif" not in exif_dict:
                exif_dict["Exif"] = {}
            exif_dict["Exif"][piexif.ExifIFD.UserComment] = user_comment
            
            # Save EXIF data
            exif_bytes = piexif.dump(exif_dict)
            piexif.insert(exif_bytes, str(img_path))
            
            logging.info(f"Added metadata to {img_path}")
            
        except Exception as e:
            logging.warning(f"Failed to add metadata to {img_path}: {str(e)}")

    def _backup_image(self, img_path):
        """Create image backup"""
        try:
            backup_path = self.backup_dir / img_path.name
            shutil.copy2(img_path, backup_path)
            logging.info(f"Created backup of {img_path.name}")
            return True
        except Exception as e:
            logging.error(f"Failed to backup {img_path}: {str(e)}")
            return False

    def process_single_image(self, img_path):
        """Process a single image"""
        img_path = Path(img_path)
        results = {
            'original_name': img_path.name,
            'success': False,
            'new_name': None,
            'animal': None,
            'confidence': None,
            'landscape_features': None,
            'error': None
        }
        
        try:
            # Backup image
            if not self._backup_image(img_path):
                results['error'] = "Backup failed"
                return results
            
            # Predict animal and landscape
            animal_name, confidence = self._predict_animal(img_path)
            landscape_features = self._predict_landscape(img_path)
            
            results.update({
                'animal': animal_name,
                'confidence': confidence,
                'landscape_features': landscape_features
            })
            
            # Check confidence threshold
            threshold = CONFIDENCE_THRESHOLDS.get(animal_name, 
                                                CONFIDENCE_THRESHOLDS['default'])
            
            if animal_name and confidence > threshold:
                # Generate new filename and rename
                new_path = self._generate_new_filename(img_path, animal_name, 
                                                     landscape_features)
                self._add_metadata_tags(img_path, animal_name, confidence, 
                                      landscape_features)
                img_path.rename(new_path)
                
                results.update({
                    'new_name': new_path.name,
                    'success': True
                })
            else:
                results['error'] = "Low confidence prediction"
                
        except Exception as e:
            results['error'] = str(e)
            
        return results

    def process_images(self):
        """Process all images in source directory"""
        image_files = [
            f for f in self.source_dir.glob("*")
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        
        results = []
        for img_path in image_files:
            result = self.process_single_image(img_path)
            results.append(result)
            
        return results