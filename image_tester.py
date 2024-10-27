# image_tester.py
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from PIL import Image
from .image_processor import SafariImageProcessor

class SafariImageTester:
    def __init__(self, test_dir):
        """Initialize the tester with test directory"""
        self.test_dir = Path(test_dir)
        self.processor = SafariImageProcessor(self.test_dir)
        
    def preview_image_prediction(self, img_path):
        """Preview image with predictions"""
        img = Image.open(img_path)
        animal_name, confidence = self.processor._predict_animal(img_path)
        landscape_features = self.processor._predict_landscape(img_path)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Display image
        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title(f'Original Image: {Path(img_path).name}')
        
        # Display prediction info
        ax2.axis('off')
        landscape_str = ', '.join(landscape_features) if landscape_features else 'None detected'
        prediction_text = f"""
        Predicted Animal: {animal_name}
        Confidence: {confidence:.3f}
        
        Landscape Features: {landscape_str}
        
        Status: {'✓ Accepted' if confidence > 0.35 else '✗ Rejected'}
        """
        ax2.text(0.1, 0.5, prediction_text, fontsize=12, va='center')
        
        plt.show()
        
        return animal_name, confidence, landscape_features
        
    def test_single(self, image_name=None):
        """Test a single image"""
        if image_name:
            image_path = self.test_dir / image_name
        else:
            images = list(self.test_dir.glob("*.[jJ][pP][gG]"))
            if not images:
                print("No images found in test directory")
                return
            image_path = images[0]
            
        print(f"\nTesting image: {image_path.name}")
        
        animal, confidence, landscape = self.preview_image_prediction(image_path)
        
        print("\nResults:")
        print(f"Animal: {animal}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Landscape Features: {', '.join(landscape) if landscape else 'None detected'}")
        print(f"Status: {'✓ Accepted' if confidence > 0.35 else '✗ Rejected'}")
        
        return animal, confidence, landscape
    
    def test_all(self, preview=True):
        """Test all images in directory"""
        images = list(self.test_dir.glob("*.[jJ][pP][gG]"))
        if not images:
            print("No images found in test directory")
            return []
        
        results = []
        print(f"\nTesting {len(images)} images:")
        
        for i, img_path in enumerate(images, 1):
            print(f"\nImage {i}/{len(images)}: {img_path.name}")
            
            if preview:
                animal, confidence, landscape = self.preview_image_prediction(img_path)
            else:
                animal, confidence = self.processor._predict_animal(img_path)
                landscape = self.processor._predict_landscape(img_path)
            
            results.append({
                'image': img_path.name,
                'animal': animal,
                'confidence': confidence,
                'landscape': landscape,
                'accepted': confidence > 0.35
            })
            
            print(f"Predicted: {animal} (Confidence: {confidence:.3f})")
            print(f"Landscape: {', '.join(landscape) if landscape else 'None detected'}")
        
        # Print summary
        print("\nSummary:")
        accepted = sum(1 for r in results if r['accepted'])
        print(f"Total images: {len(results)}")
        print(f"Accepted predictions: {accepted}")
        print(f"Rejected predictions: {len(results) - accepted}")
        
        return results
    
    def process_all(self):
        """Process all images"""
        return self.processor.process_images()