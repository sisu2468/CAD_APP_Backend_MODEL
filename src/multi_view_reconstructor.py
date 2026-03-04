"""
Multi-View Reconstruction Module
Combines information from multiple images taken from different angles to reconstruct 3D geometry.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from src.image_processor import ImageFeatures, ImageProcessor

logger = logging.getLogger(__name__)


@dataclass
class ViewInfo:
    """Information about a single view."""
    image_path: str
    features: ImageFeatures
    view_type: str  # 'front', 'side', 'top', 'isometric', etc.
    camera_angle: Optional[Tuple[float, float, float]] = None  # (pitch, yaw, roll)


@dataclass
class ReconstructedGeometry:
    """Reconstructed 3D geometry from multiple views."""
    vertices: np.ndarray  # Nx3 array of 3D points
    edges: List[Tuple[int, int]]  # List of edge connections
    faces: List[List[int]]  # List of face vertex indices
    dimensions: Dict[str, float]  # Extracted dimensions
    features: Dict[str, List]  # Geometric features (holes, chamfers, etc.)


class MultiViewReconstructor:
    """Reconstructs 3D geometry from multiple orthographic views."""
    
    def __init__(self, image_processor: Optional[ImageProcessor] = None):
        """
        Initialize the multi-view reconstructor.
        
        Args:
            image_processor: Image processor instance (creates new one if None)
        """
        self.image_processor = image_processor or ImageProcessor()
        self.views: List[ViewInfo] = []
    
    def add_view(self, image_path: str, view_type: str = 'unknown') -> ViewInfo:
        """
        Add a view to the reconstruction.
        
        Args:
            image_path: Path to the image
            view_type: Type of view ('front', 'side', 'top', etc.)
            
        Returns:
            ViewInfo object
        """
        features = self.image_processor.extract_features(image_path)
        view_info = ViewInfo(
            image_path=image_path,
            features=features,
            view_type=view_type
        )
        self.views.append(view_info)
        logger.info(f"Added {view_type} view: {image_path}")
        return view_info
    
    def infer_view_type(self, image_path: str, index: int) -> str:
        """
        Infer view type from image index or content.
        
        Args:
            image_path: Path to image
            index: Index of image in sequence
            
        Returns:
            Inferred view type
        """
        # Simple heuristic: assume first image is front, second is side, etc.
        view_types = ['front', 'side', 'top']
        if index < len(view_types):
            return view_types[index]
        return 'unknown'
    
    def reconstruct_geometry(self) -> ReconstructedGeometry:
        """
        Reconstruct 3D geometry from all views.
        
        Returns:
            ReconstructedGeometry object
        """
        if len(self.views) < 1:
            raise ValueError("At least one view is required for reconstruction")
        
        logger.info(f"Reconstructing geometry from {len(self.views)} views")
        
        # Combine features from all views
        combined_features = self._combine_features()
        
        # Extract 3D geometry
        vertices, edges, faces = self._extract_3d_geometry()
        
        # Extract dimensions
        dimensions = self._extract_combined_dimensions()
        
        # Extract features (holes, chamfers, etc.)
        features = self._extract_combined_features()
        
        return ReconstructedGeometry(
            vertices=vertices,
            edges=edges,
            faces=faces,
            dimensions=dimensions,
            features=features
        )
    
    def _combine_features(self) -> Dict:
        """Combine features from all views."""
        combined = {
            'contours': [],
            'circles': [],
            'rectangles': [],
            'lines': [],
            'polygons': []
        }
        
        for view in self.views:
            combined['contours'].extend(view.features.contours)
            combined['circles'].extend(view.features.circles)
            combined['rectangles'].extend(view.features.rectangles)
            combined['lines'].extend(view.features.lines)
            combined['polygons'].extend(view.features.polygons)
        
        return combined
    
    def _extract_3d_geometry(self) -> Tuple[np.ndarray, List[Tuple[int, int]], List[List[int]]]:
        """
        Extract 3D geometry from 2D views.
        
        Returns:
            Tuple of (vertices, edges, faces)
        """
        # For now, create a simplified 3D representation
        # In a full implementation, this would use proper multi-view geometry
        
        vertices = []
        edges = []
        faces = []
        
        # Extract bounding boxes from each view
        for view in self.views:
            if view.features.contours:
                largest_contour = max(
                    view.features.contours, 
                    key=lambda c: cv2.contourArea(c) if hasattr(cv2, 'contourArea') else 0
                )
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Create 3D vertices based on view type
                if view.view_type == 'front':
                    # Front view: X-Y plane at Z=0
                    vertices.extend([
                        [x, y, 0],
                        [x+w, y, 0],
                        [x+w, y+h, 0],
                        [x, y+h, 0]
                    ])
                elif view.view_type == 'side':
                    # Side view: Y-Z plane
                    vertices.extend([
                        [0, y, x],
                        [0, y+h, x],
                        [0, y+h, x+w],
                        [0, y, x+w]
                    ])
                elif view.view_type == 'top':
                    # Top view: X-Z plane
                    vertices.extend([
                        [x, 0, y],
                        [x+w, 0, y],
                        [x+w, 0, y+h],
                        [x, 0, y+h]
                    ])
        
        if not vertices:
            # Fallback: create a simple box
            vertices = np.array([
                [0, 0, 0],
                [100, 0, 0],
                [100, 100, 0],
                [0, 100, 0],
                [0, 0, 10],
                [100, 0, 10],
                [100, 100, 10],
                [0, 100, 10]
            ])
        else:
            vertices = np.array(vertices)
        
        # Create edges (simplified)
        if len(vertices) >= 4:
            # Create edges for a box
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
                (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
                (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
            ]
        
        return vertices, edges, faces
    
    def _extract_combined_dimensions(self) -> Dict[str, float]:
        """Extract dimensions by combining information from all views."""
        dimensions = {}
        
        # Combine dimensions from all views
        for view in self.views:
            for key, value in view.features.dimensions.items():
                if key not in dimensions:
                    dimensions[key] = value
                else:
                    # Use average or maximum depending on the dimension type
                    dimensions[key] = max(dimensions[key], value)
        
        return dimensions
    
    def _extract_combined_features(self) -> Dict[str, List]:
        """Extract combined features (holes, chamfers, etc.) from all views."""
        features = {
            'holes': [],
            'chamfers': [],
            'fillets': []
        }
        
        # Extract holes from circles
        for view in self.views:
            for circle in view.features.circles:
                features['holes'].append({
                    'center': (circle[0], circle[1]),
                    'radius': circle[2],
                    'view': view.view_type
                })
        
        return features
