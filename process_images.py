"""
Quick script to process IMG_*.jpg files and generate CAD drawings
"""

import os
from src.pipeline import CADExtractionPipeline
import logging

logging.basicConfig(level=logging.INFO)

# Create pipeline
pipeline = CADExtractionPipeline(
    output_dir="output",
    enable_dimensions=True,
    enable_pdf=True
)

# Available images
images = [
    "IMG_5047.jpg",
    "IMG_5048.jpg", 
    "IMG_5402.jpg",
    "IMG_5545.JPG",
    "IMG_5546.JPG",
    "IMG_5547.JPG"
]

# Check which images exist
existing_images = [img for img in images if os.path.exists(img)]
print(f"Found {len(existing_images)} images: {existing_images}")

# Process each image individually
for image in existing_images:
    try:
        print(f"\nProcessing {image}...")
        output_name = os.path.splitext(image)[0]  # Use filename without extension
        
        results = pipeline.process_single_image(
            image_path=image,
            output_name=output_name
        )
        
        print(f"  ✓ Generated: {results['dxf']}")
        if results.get('pdf'):
            print(f"  ✓ Generated: {results['pdf']}")
            
    except Exception as e:
        print(f"  ✗ Error processing {image}: {e}")

print("\n" + "="*60)
print("Processing complete! Check the 'output' folder for results.")
print("="*60)
