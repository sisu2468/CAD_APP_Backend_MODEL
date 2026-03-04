"""
Main Entry Point
Command-line interface for the CAD extraction system.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

from src.pipeline import CADExtractionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Extract CAD drawings from smartphone images'
    )
    
    parser.add_argument(
        'images',
        nargs='+',
        help='Input image paths (multiple images for multi-view reconstruction)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='output',
        help='Output directory (default: output)'
    )
    
    parser.add_argument(
        '-n', '--name',
        default='cad_drawing',
        help='Output file name (default: cad_drawing)'
    )
    
    parser.add_argument(
        '--views',
        nargs='+',
        help='View types (e.g., front side top). Auto-detected if not provided.'
    )
    
    parser.add_argument(
        '--no-pdf',
        action='store_true',
        help='Skip PDF generation'
    )
    
    parser.add_argument(
        '--no-dimensions',
        action='store_true',
        help='Skip dimension extraction'
    )
    
    args = parser.parse_args()
    
    # Validate input images
    for image_path in args.images:
        if not Path(image_path).exists():
            logger.error(f"Image not found: {image_path}")
            sys.exit(1)
    
    # Create pipeline
    pipeline = CADExtractionPipeline(
        output_dir=args.output,
        enable_dimensions=not args.no_dimensions,
        enable_pdf=not args.no_pdf
    )
    
    # Process images
    # Single image: use the accurate outline-extraction path directly.
    # Multi-view (2+ images): use the reconstructor (experimental).
    try:
        if len(args.images) == 1:
            results = pipeline.process_single_image(
                image_path=args.images[0],
                output_name=args.name,
            )
        else:
            results = pipeline.process_images(
                image_paths=args.images,
                view_types=args.views,
                output_name=args.name,
            )
        
        logger.info("=" * 60)
        logger.info("Processing completed successfully!")
        logger.info("=" * 60)
        logger.info(f"DXF file: {results['dxf']}")
        if results['pdf']:
            logger.info(f"PDF file: {results['pdf']}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error processing images: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
