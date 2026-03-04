"""
Test script to verify all imports work correctly.
"""

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        print("  - Testing src.image_processor...")
        from src.image_processor import ImageProcessor, ImageFeatures
        print("    ✓ ImageProcessor imported successfully")
        
        print("  - Testing src.multi_view_reconstructor...")
        from src.multi_view_reconstructor import MultiViewReconstructor
        print("    ✓ MultiViewReconstructor imported successfully")
        
        print("  - Testing src.cad_generator...")
        from src.cad_generator import CADGenerator
        print("    ✓ CADGenerator imported successfully")
        
        print("  - Testing src.pdf_exporter...")
        from src.pdf_exporter import PDFExporter
        print("    ✓ PDFExporter imported successfully")
        
        print("  - Testing src.dimension_measurement...")
        from src.dimension_measurement import DimensionMeasurer, Dimension
        print("    ✓ DimensionMeasurer imported successfully")
        
        print("  - Testing src.pipeline...")
        from src.pipeline import CADExtractionPipeline
        print("    ✓ CADExtractionPipeline imported successfully")
        
        print("\n[SUCCESS] All imports successful!")
        return True
    
    except ImportError as e:
        print(f"\n[ERROR] Import error: {e}")
        print("\nPlease install dependencies: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_imports()
    exit(0 if success else 1)
