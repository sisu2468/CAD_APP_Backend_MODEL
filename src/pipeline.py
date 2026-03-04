"""
Main Pipeline Module
Orchestrates the complete CAD extraction pipeline from images to PDF output.
Single-image path: uses trained outline model if available, else rule-based extractor with params.json.
"""

import os
from typing import List, Optional, Dict
import logging
from pathlib import Path

from src.image_processor import ImageProcessor
from src.multi_view_reconstructor import MultiViewReconstructor
from src.cad_generator import CADGenerator
from src.pdf_exporter import PDFExporter
from src.outline_extractor import create_extractor_from_params, load_params

logger = logging.getLogger(__name__)

TRAINING_PARAMS_PATH = Path(__file__).resolve().parent.parent / "training_data" / "params.json"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
TRAINING_MODEL_PATH = Path(__file__).resolve().parent.parent / "training_data" / "outline_model.pt"


def _get_outline_result(image_path: str):
    """Get outline result: use trained model if available, else rule-based extractor.

    Returns either:
    - (contours, holes)        — classic tuple for add_outline()
    - GeometricOutline object  — for add_geometric_outline() (higher accuracy)
    """
    try:
        from src.models.predictor import OutlineModelPredictor, get_default_model_path
        model_path = get_default_model_path()
        if model_path is not None:
            predictor = OutlineModelPredictor(str(model_path))
            geometric = predictor.extract_geometric(image_path)
            logger.info("Using outline model (geometric): %s", model_path)
            return geometric  # GeometricOutline
    except Exception as e:
        logger.debug("Model geometric inference skipped: %s", e)
    # Fallback: rule-based extractor
    params = None
    if TRAINING_PARAMS_PATH.exists():
        try:
            params = load_params(str(TRAINING_PARAMS_PATH))
        except Exception:
            pass
    extractor = create_extractor_from_params(params)
    result = extractor.extract(image_path)
    return extractor.to_dxf_friendly(result, scale=1.0, origin=(0, 0))


class CADExtractionPipeline:
    """Main pipeline for extracting CAD drawings from smartphone images."""
    
    def __init__(self,
                 output_dir: str = "output",
                 enable_dimensions: bool = True,
                 enable_pdf: bool = True):
        """
        Initialize the CAD extraction pipeline.
        
        Args:
            output_dir: Directory for output files
            enable_dimensions: Whether to extract and add dimensions
            enable_pdf: Whether to generate PDF output
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.image_processor = ImageProcessor()
        self.reconstructor = MultiViewReconstructor(self.image_processor)
        self.cad_generator = CADGenerator()
        self.pdf_exporter = PDFExporter() if enable_pdf else None
        
        self.enable_dimensions = enable_dimensions
    
    def process_images(self,
                      image_paths: List[str],
                      view_types: Optional[List[str]] = None,
                      output_name: str = "output") -> Dict[str, str]:
        """
        Process multiple images and generate CAD drawing.
        
        Args:
            image_paths: List of paths to input images
            view_types: Optional list of view types (auto-detected if None)
            output_name: Base name for output files
            
        Returns:
            Dictionary with paths to generated files
        """
        logger.info(f"Processing {len(image_paths)} images")
        
        # Add views to reconstructor
        for i, image_path in enumerate(image_paths):
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            view_type = view_types[i] if view_types and i < len(view_types) else None
            if view_type is None:
                view_type = self.reconstructor.infer_view_type(image_path, i)
            
            self.reconstructor.add_view(image_path, view_type)
        
        # Reconstruct geometry
        geometry = self.reconstructor.reconstruct_geometry()
        logger.info("Geometry reconstructed successfully")
        
        # Generate CAD drawing
        self.cad_generator.create_drawing()
        self.cad_generator.add_orthographic_views(geometry)
        
        if self.enable_dimensions and geometry.dimensions:
            self.cad_generator.add_dimensions(geometry.dimensions)
        
        # Save DXF
        dxf_path = str(self.output_dir / f"{output_name}.dxf")
        self.cad_generator.save(dxf_path)
        logger.info(f"DXF saved to {dxf_path}")
        
        # Generate PDF if enabled
        pdf_path = None
        if self.pdf_exporter:
            pdf_path = str(self.output_dir / f"{output_name}.pdf")
            self.pdf_exporter.export_from_dxf(dxf_path, pdf_path, 
                                             title=f"CAD Drawing: {output_name}")
        
        return {
            'dxf': dxf_path,
            'pdf': pdf_path
        }
    
    def process_single_image(self,
                            image_path: str,
                            output_name: str = "output") -> Dict[str, str]:
        """
        Process a single image: extract clean outline + holes, export DXF/PDF.
        Uses trained outline model if present (models/outline_model.pt or training_data/outline_model.pt),
        else rule-based extractor with training_data/params.json.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        outline_result = _get_outline_result(image_path)
        self.cad_generator.create_drawing()
        # If we got a GeometricOutline (from the model), use it for higher accuracy.
        # Otherwise fall back to the classic (contours, holes) tuple.
        from src.outline_extractor import GeometricOutline
        if isinstance(outline_result, GeometricOutline):
            self.cad_generator.add_geometric_outline(outline_result, offset=(0, 0))
        else:
            contours, holes = outline_result
            self.cad_generator.add_outline(contours, holes, offset=(0, 0))
        dxf_path = str(self.output_dir / f"{output_name}.dxf")
        self.cad_generator.save(dxf_path)
        logger.info("DXF saved to %s", dxf_path)
        pdf_path = None
        if self.pdf_exporter:
            pdf_path = str(self.output_dir / f"{output_name}.pdf")
            self.pdf_exporter.export_from_dxf(dxf_path, pdf_path, title=f"CAD Drawing: {output_name}")
        return {"dxf": dxf_path, "pdf": pdf_path}
    
    def add_custom_features(self, features: Dict) -> None:
        """
        Add custom features to the CAD drawing.
        
        Args:
            features: Dictionary of custom features
        """
        # This can be extended for custom feature addition
        pass
