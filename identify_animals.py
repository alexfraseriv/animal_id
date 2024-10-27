import os
import sys
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import piexif
import logging
from datetime import datetime
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('safari_processing.log'),
        logging.StreamHandler()
    ]
)

class SafariImageProcessor:
    def __init__(self, source_dir, backup_dir=None):
        """
        Initialize the image processor with source directory and optional backup directory
        """
        self.source_dir = Path(source_dir)
        self.backup_dir = Path(backup_dir) if backup_dir else Path(source_dir) / 'backup'
        self.model = None
        self.processed_count = 0
        
        # Create backup directory if it doesn't exist
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the ML model
        self._load_model()
        
        # Supported image extensions
        self.supported_extensions = {'.jpg', '.jpeg', '.png'}

    def _load_model(self):
        """Load the ResNet50 model for image classification"""
        try:
            self.model = ResNet50(weights='imagenet')
            logging.info("Successfully loaded ResNet50 model")
        except Exception as e:
            logging.error(f"Failed to load ML model: {str(e)}")
            sys.exit(1)

    def _predict_animal(self, img_path):
        """
        Predict the animal in the image using ResNet50
        Returns tuple of (animal_name, confidence)
        """
        try:
            # Load and preprocess the image
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # Make prediction
            preds = self.model.predict(x)
            predictions = decode_predictions(preds, top=3)[0]
            
            # Filter for animal predictions
            animal_preds = [pred for pred in predictions if pred[1] in self._get_animal_categories()]
            
            if animal_preds:
                return animal_preds[0][1], float(animal_preds[0][2])
            return None, 0.0
            
        except Exception as e:
            logging.error(f"Error predicting image {img_path}: {str(e)}")
            return None, 0.0

    def _get_animal_categories(self):
        """
        Return set of ImageNet categories that represent animals we're interested in
        """
        return {
            'lion', 'tiger', 'elephant', 'zebra', 'giraffe', 'cheetah', 'leopard',
            'rhinoceros', 'hippopotamus', 'gazelle', 'antelope', 'buffalo',
            'wildebeest', 'monkey', 'baboon', 'crocodile', 'ostrich', 'vulture',
            'eagle', 'hyena'
        }

    def _add_metadata_tags(self, img_path, animal_name, confidence):
        """
        Add metadata tags to the image using piexif
        """
        try:
            # Load existing EXIF data
            exif_dict = piexif.load(str(img_path))
            
            # Create new EXIF UserComment
            comment = f"Animal: {animal_name}; Confidence: {confidence:.2f}"
            
            # Convert comment to bytes
            user_comment = piexif.helper.UserComment.create(comment)
            
            # Add to EXIF
            if "Exif" not in exif_dict:
                exif_dict["Exif"] = {}
            exif_dict["Exif"][piexif.ExifIFD.UserComment] = user_comment
            
            # Save EXIF data back to image
            exif_bytes = piexif.dump(exif_dict)
            piexif.insert(exif_bytes, str(img_path))
            
            logging.info(f"Added metadata tags to {img_path}")
            
        except Exception as e:
            logging.warning(f"Failed to add metadata to {img_path}: {str(e)}")

    def _backup_image(self, img_path):
        """
        Create a backup of the original image
        """
        try:
            backup_path = self.backup_dir / img_path.name
            shutil.copy2(img_path, backup_path)
            logging.info(f"Created backup of {img_path.name}")
            return True
        except Exception as e:
            logging.error(f"Failed to create backup of {img_path}: {str(e)}")
            return False

    def _generate_new_filename(self, original_path, animal_name):
        """
        Generate new filename with animal name and sequence number
        """
        self.processed_count += 1
        timestamp = datetime.now().strftime("%Y%m%d")
        sequence = f"{self.processed_count:04d}"
        extension = original_path.suffix.lower()
        
        new_name = f"{animal_name}_{timestamp}_{sequence}{extension}"
        return original_path.parent / new_name

    def process_images(self):
        """
        Main method to process all images in the source directory
        """
        logging.info(f"Starting image processing in {self.source_dir}")
        
        # Get list of all image files
        image_files = [
            f for f in self.source_dir.glob("*")
            if f.suffix.lower() in self.supported_extensions
        ]
        
        total_files = len(image_files)
        logging.info(f"Found {total_files} images to process")
        
        for img_path in image_files:
            try:
                logging.info(f"Processing {img_path.name}")
                
                # Backup original image
                if not self._backup_image(img_path):
                    continue
                
                # Predict animal in image
                animal_name, confidence = self._predict_animal(img_path)
                
                if animal_name and confidence > 0.5:  # Only process if confident enough
                    # Generate new filename
                    new_path = self._generate_new_filename(img_path, animal_name)
                    
                    # Add metadata tags
                    self._add_metadata_tags(img_path, animal_name, confidence)
                    
                    # Rename file
                    img_path.rename(new_path)
                    logging.info(f"Renamed {img_path.name} to {new_path.name}")
                else:
                    logging.warning(f"Could not confidently identify animal in {img_path.name}")
                
            except Exception as e:
                logging.error(f"Error processing {img_path.name}: {str(e)}")
                continue
        
        logging.info(f"Completed processing {self.processed_count} images")

def main():
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python safari_processor.py <source_directory> [backup_directory]")
        sys.exit(1)
    
    source_dir = sys.argv[1]
    backup_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Create and run processor
    processor = SafariImageProcessor(source_dir, backup_dir)
    processor.process_images()

if __name__ == "__main__":
    main()