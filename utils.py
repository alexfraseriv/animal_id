# utils.py
import logging
from pathlib import Path
import json
from datetime import datetime
import os
from PIL import Image
import piexif
from config import LOG_CONFIG

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(**LOG_CONFIG)

def validate_image(img_path):
    """Validate image file"""
    try:
        with Image.open(img_path) as img:
            img.verify()
        return True
    except Exception as e:
        logging.error(f"Invalid image file {img_path}: {str(e)}")
        return False

def create_directory_structure(base_dir):
    """Create necessary directory structure"""
    dirs = {
        'backup': base_dir / 'backup',
        'processed': base_dir / 'processed',
        'rejected': base_dir / 'rejected',
        'logs': base_dir / 'logs'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs

def save_processing_report(results, output_dir):
    """Save processing results to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"processing_report_{timestamp}.json"
    
    report_data = {
        'timestamp': timestamp,
        'total_images': len(results),
        'successful': sum(1 for r in results if r['success']),
        'failed': sum(1 for r in results if not r['success']),
        'results': results
    }
    
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=4)
    
    return report_file

def extract_image_metadata(img_path):
    """Extract and format image metadata"""
    metadata = {}
    try:
        img = Image.open(img_path)
        metadata['format'] = img.format
        metadata['size'] = img.size
        metadata['mode'] = img.mode
        
        if 'exif' in img.info:
            exif_dict = piexif.load(img.info['exif'])
            
            # Extract basic EXIF data
            if '0th' in exif_dict:
                for tag, value in exif_dict['0th'].items():
                    tag_name = piexif.TAGS['0th'].get(tag, {}).get('name', str(tag))
                    if isinstance(value, bytes):
                        try:
                            value = value.decode()
                        except:
                            value = str(value)
                    metadata[tag_name] = value
            
            # Extract GPS data if available
            if 'GPS' in exif_dict:
                gps_data = {}
                for tag, value in exif_dict['GPS'].items():
                    tag_name = piexif.TAGS['GPS'].get(tag, {}).get('name', str(tag))
                    gps_data[tag_name] = value
                metadata['GPS'] = gps_data
    
    except Exception as e:
        logging.warning(f"Error extracting metadata from {img_path}: {str(e)}")
    
    return metadata

def generate_html_report(results, output_dir):
    """Generate HTML report with processing results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"processing_report_{timestamp}.html"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Safari Image Processing Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .summary {{ background-color: #f0f0f0; padding: 15px; margin-bottom: 20px; }}
            .results {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }}
            .result-card {{ border: 1px solid #ddd; padding: 15px; }}
            .success {{ color: green; }}
            .error {{ color: red; }}
        </style>
    </head>
    <body>
        <h1>Safari Image Processing Report</h1>
        <div class="summary">
            <h2>Summary</h2>
            <p>Processed at: {timestamp}</p>
            <p>Total Images: {len(results)}</p>
            <p>Successful: {sum(1 for r in results if r['success'])}</p>
            <p>Failed: {sum(1 for r in results if not r['success'])}</p>
        </div>
        
        <div class="results">
    """
    
    for result in results:
        status_class = "success" if result['success'] else "error"
        html_content += f"""
            <div class="result-card">
                <h3>{result['original_name']}</h3>
                <p class="{status_class}">Status: {"Success" if result['success'] else "Failed"}</p>
                <p>Animal: {result['animal'] or 'Not detected'}</p>
                <p>Confidence: {result['confidence']:.3f if result['confidence'] else 'N/A'}</p>
                <p>Landscape Features: {', '.join(result['landscape_features']) if result['landscape_features'] else 'None detected'}</p>
                {f'<p>New Name: {result["new_name"]}</p>' if result['new_name'] else ''}
                {f'<p class="error">Error: {result["error"]}</p>' if result['error'] else ''}
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(report_file, 'w') as f:
        f.write(html_content)
    
    return report_file

def analyze_results(results):
    """Analyze processing results and return statistics"""
    stats = {
        'total': len(results),
        'successful': sum(1 for r in results if r['success']),
        'failed': sum(1 for r in results if not r['success']),
        'animals': {},
        'landscapes': {},
        'avg_confidence': 0.0,
        'error_types': {}
    }
    
    total_confidence = 0
    confidence_count = 0
    
    for result in results:
        # Animal statistics
        if result['animal']:
            stats['animals'][result['animal']] = stats['animals'].get(result['animal'], 0) + 1
        
        # Landscape statistics
        if result['landscape_features']:
            for feature in result['landscape_features']:
                stats['landscapes'][feature] = stats['landscapes'].get(feature, 0) + 1
        
        # Confidence statistics
        if result['confidence']:
            total_confidence += result['confidence']
            confidence_count += 1
        
        # Error statistics
        if result['error']:
            stats['error_types'][result['error']] = stats['error_types'].get(result['error'], 0) + 1
    
    # Calculate average confidence
    if confidence_count > 0:
        stats['avg_confidence'] = total_confidence / confidence_count
    
    return stats