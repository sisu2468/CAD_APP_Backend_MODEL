"""
Example Usage Script
Demonstrates how to use the CAD extraction system.
"""

from src.pipeline import CADExtractionPipeline
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def example_single_image():
    """Example: Process a single image."""
    print("=" * 60)
    print("Example 1: Processing single image")
    print("=" * 60)
    
    pipeline = CADExtractionPipeline(
        output_dir="output",
        enable_dimensions=True,
        enable_pdf=True
    )
    
    # Process single image
    results = pipeline.process_single_image(
        image_path="IMG_8840.jpg",
        output_name="single_view_example"
    )
    
    print(f"Generated DXF: {results['dxf']}")
    if results['pdf']:
        print(f"Generated PDF: {results['pdf']}")


def example_multi_view():
    """Example: Process multiple views."""
    print("=" * 60)
    print("Example 2: Processing multiple views")
    print("=" * 60)
    
    pipeline = CADExtractionPipeline(
        output_dir="output",
        enable_dimensions=True,
        enable_pdf=True
    )
    
    # Process multiple images with specified view types
    results = pipeline.process_images(
        image_paths=["IMG_8843.jpg", "IMG_8844.jpg"],
        view_types=["front", "side"],
        output_name="multi_view_example"
    )
    
    print(f"Generated DXF: {results['dxf']}")
    if results['pdf']:
        print(f"Generated PDF: {results['pdf']}")


def example_auto_detect_views():
    """Example: Auto-detect view types."""
    print("=" * 60)
    print("Example 3: Auto-detecting view types")
    print("=" * 60)
    
    pipeline = CADExtractionPipeline(
        output_dir="output",
        enable_dimensions=True,
        enable_pdf=True
    )
    
    # Process without specifying view types (auto-detection)
    results = pipeline.process_images(
        image_paths=["IMG_8840.jpg", "IMG_8841.jpg"],
        output_name="auto_detect_example"
    )
    
    print(f"Generated DXF: {results['dxf']}")
    if results['pdf']:
        print(f"Generated PDF: {results['pdf']}")


if __name__ == '__main__':
    # Run examples
    try:
        example_single_image()
        print("\n")
        
        # Uncomment to run other examples
        # example_multi_view()
        # print("\n")
        # example_auto_detect_views()
        
    except FileNotFoundError as e:
        print(f"Error: Image file not found. {e}")
        print("Please ensure the image files exist in the current directory.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
