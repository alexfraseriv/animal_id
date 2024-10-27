# main.py
import argparse
from pathlib import Path
import logging
from datetime import datetime

from config import TEST_DIR, BACKUP_DIR
from image_processor import SafariImageProcessor
from image_tester import SafariImageTester
from utils import (
    setup_logging,
    create_directory_structure,
    save_processing_report,
    generate_html_report,
    analyze_results
)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Safari Image Processing Tool')
    parser.add_argument('--input', type=str, default=TEST_DIR,
                       help='Input directory containing images')
    parser.add_argument('--backup', type=str, default=BACKUP_DIR,
                       help='Backup directory for original images')
    parser.add_argument('--preview', action='store_true',
                       help='Preview mode - show predictions without processing')
    parser.add_argument('--test', type=str,
                       help='Test specific image file')
    parser.add_argument('--report', choices=['json', 'html', 'both'],
                       default='both', help='Report format')
    return parser.parse_args()

def main():
    """Main execution function"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging()
    
    # Create directory structure
    input_dir = Path(args.input)
    dirs = create_directory_structure(input_dir)
    
    try:
        if args.preview:
            # Preview mode
            tester = SafariImageTester(input_dir)
            if args.test:
                results = [tester.test_single(args.test)]
            else:
                results = tester.test_all(preview=True)
            
            # Generate preview report
            if args.report in ['json', 'both']:
                report_file = save_processing_report(results, dirs['logs'])
                logging.info(f"Preview report saved to: {report_file}")
            
            if args.report in ['html', 'both']:
                html_file = generate_html_report(results, dirs['logs'])
                logging.info(f"HTML report saved to: {html_file}")
        
        else:
            # Processing mode
            processor = SafariImageProcessor(input_dir, dirs['backup'])
            
            if args.test:
                # Process single image
                result = processor.process_single_image(input_dir / args.test)
                results = [result]
            else:
                # Process all images
                results = processor.process_images()
            
            # Analyze results
            stats = analyze_results(results)
            
            # Log summary
            logging.info("\nProcessing Summary:")
            logging.info(f"Total images processed: {stats['total']}")
            logging.info(f"Successfully processed: {stats['successful']}")
            logging.info(f"Failed: {stats['failed']}")
            logging.info(f"Average confidence: {stats['avg_confidence']:.3f}")
            
            # Generate reports
            if args.report in ['json', 'both']:
                report_file = save_processing_report(results, dirs['logs'])
                logging.info(f"Processing report saved to: {report_file}")
            
            if args.report in ['html', 'both']:
                html_file = generate_html_report(results, dirs['logs'])
                logging.info(f"HTML report saved to: {html_file}")
    
    except Exception as e:
        logging.error(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()