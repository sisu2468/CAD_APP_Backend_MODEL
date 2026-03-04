"""
CAD Drawing Generator Module
Generates professional CAD drawings with orthographic projections from extracted features.
"""

import ezdxf
from ezdxf import colors
from ezdxf.math import Vec3
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from src.multi_view_reconstructor import ReconstructedGeometry

logger = logging.getLogger(__name__)


class CADGenerator:
    """Generates CAD drawings in DXF format with orthographic projections."""
    
    def __init__(self, dxf_version: str = 'R2010'):
        """
        Initialize CAD generator.
        
        Args:
            dxf_version: DXF version to use
        """
        self.dxf_version = dxf_version
        self.doc = None
        self.msp = None
    
    def create_drawing(self) -> None:
        """Create a new DXF document."""
        self.doc = ezdxf.new(self.dxf_version)
        self.msp = self.doc.modelspace()
        logger.info(f"Created new DXF document (version {self.dxf_version})")
    
    def add_orthographic_views(self, geometry: ReconstructedGeometry,
                               view_spacing: float = 200.0) -> None:
        """
        Add orthographic views (front, side, top) to the drawing.
        
        Args:
            geometry: Reconstructed geometry
            view_spacing: Spacing between views
        """
        if self.doc is None:
            self.create_drawing()
        
        # Front view (X-Y plane)
        front_offset = Vec3(0, 0, 0)
        self._add_view(geometry, front_offset, 'front')
        
        # Side view (Y-Z plane, projected to right)
        side_offset = Vec3(view_spacing, 0, 0)
        self._add_view(geometry, side_offset, 'side')
        
        # Top view (X-Z plane, projected upward)
        top_offset = Vec3(0, view_spacing, 0)
        self._add_view(geometry, top_offset, 'top')
    
    def _add_view(self, geometry: ReconstructedGeometry, 
                 offset: Vec3, view_type: str) -> None:
        """
        Add a single orthographic view.
        
        Args:
            geometry: Reconstructed geometry
            offset: Offset for this view
            view_type: Type of view ('front', 'side', 'top')
        """
        # Project vertices based on view type
        if view_type == 'front':
            # X-Y projection (Z is depth)
            projected_vertices = [(v[0] + offset.x, v[1] + offset.y) 
                                 for v in geometry.vertices]
        elif view_type == 'side':
            # Y-Z projection (X is depth)
            projected_vertices = [(v[1] + offset.x, v[2] + offset.y) 
                                 for v in geometry.vertices]
        elif view_type == 'top':
            # X-Z projection (Y is depth)
            projected_vertices = [(v[0] + offset.x, v[2] + offset.y) 
                                 for v in geometry.vertices]
        else:
            projected_vertices = [(v[0] + offset.x, v[1] + offset.y) 
                                 for v in geometry.vertices]
        
        # Draw edges
        for edge in geometry.edges:
            if edge[0] < len(projected_vertices) and edge[1] < len(projected_vertices):
                start = projected_vertices[edge[0]]
                end = projected_vertices[edge[1]]
                self.msp.add_line(
                    Vec3(start[0], start[1], 0),
                    Vec3(end[0], end[1], 0),
                    dxfattribs={'color': colors.BLACK}
                )
        
        # Draw features (holes, etc.)
        self._add_features(geometry.features, offset, view_type)
    
    def _add_features(self, features: Dict[str, List], 
                     offset: Vec3, view_type: str) -> None:
        """
        Add geometric features (holes, chamfers, etc.) to the view.
        
        Args:
            features: Dictionary of features
            offset: Offset for this view
            view_type: Type of view
        """
        # Add holes
        for hole in features.get('holes', []):
            center = hole['center']
            radius = hole['radius']
            
            # Only draw if it's visible in this view
            if hole.get('view') == view_type or view_type == 'front':
                circle_center = Vec3(
                    center[0] + offset.x,
                    center[1] + offset.y,
                    0
                )
                self.msp.add_circle(
                    circle_center,
                    radius,
                    dxfattribs={'color': colors.BLACK}
                )
    
    def add_dimensions(self, dimensions: Dict[str, float],
                      position: Tuple[float, float] = (0, -50)) -> None:
        """
        Add dimension annotations to the drawing.
        
        Args:
            dimensions: Dictionary of dimension values
            position: Position for dimension text
        """
        if self.doc is None:
            self.create_drawing()
        
        # Add dimension text
        y_offset = 0
        for key, value in dimensions.items():
            text_y = position[1] - y_offset
            self.msp.add_text(
                f"{key}: {value:.2f}",
                dxfattribs={
                    'height': 10,
                    'color': colors.BLACK
                }
            ).set_placement(Vec3(position[0], text_y, 0))
            y_offset += 15
    
    def add_contours(self, contours: List[np.ndarray],
                    offset: Tuple[float, float] = (0, 0)) -> None:
        """
        Add contours directly to the drawing.
        
        Args:
            contours: List of contour arrays (each Nx2 or Nx1x2)
            offset: Offset for positioning
        """
        if self.doc is None:
            self.create_drawing()
        
        ox, oy = offset
        for contour in contours:
            if len(contour) == 0:
                continue
            c = np.asarray(contour)
            if c.ndim == 3:
                pts = c.reshape(-1, 2)
            else:
                pts = c.reshape(-1, 2)
            if len(pts) < 2:
                continue
            self.msp.add_lwpolyline(
                [(float(pts[i, 0]) + ox, float(pts[i, 1]) + oy) for i in range(len(pts))],
                close=True,
                dxfattribs={'color': colors.BLACK}
            )
    
    def add_outline(
        self,
        contours: List[np.ndarray],
        holes: List[Tuple[Tuple[float, float], float]],
        offset: Tuple[float, float] = (0, 0),
    ) -> None:
        """
        Add clean outline (main contour polylines + circles for holes).
        contours: list of (N, 2) point arrays; holes: list of ((cx, cy), radius).
        """
        if self.doc is None:
            self.create_drawing()
        ox, oy = offset
        for contour in contours:
            if len(contour) < 2:
                continue
            pts = np.asarray(contour).reshape(-1, 2)
            self.msp.add_lwpolyline(
                [(float(pts[i, 0]) + ox, float(pts[i, 1]) + oy) for i in range(len(pts))],
                close=True,
                dxfattribs={'color': colors.BLACK}
            )
        for (cx, cy), r in holes:
            self.msp.add_circle(
                Vec3(cx + ox, cy + oy, 0),
                float(r),
                dxfattribs={'color': colors.BLACK}
            )
    
    def add_geometric_outline(
        self,
        geometric,
        offset: Tuple[float, float] = (0, 0),
        line_layer: str = "OUTLINE_LINES",
        poly_layer: str = "OUTLINE_POLY",
        hole_layer: str = "HOLES",
    ) -> None:
        """Add a GeometricOutline to the DXF drawing.

        Uses LSD-detected line segments as DXF LINE entities (straight edges),
        the raw contour as a fallback LWPOLYLINE, and algebraically-fitted hole
        circles as DXF CIRCLE entities.

        Args:
            geometric: GeometricOutline from OutlineExtractor.extract_geometric()
                       or OutlineModelPredictor.extract_geometric().
            offset: (x, y) translation for all geometry.
            line_layer: DXF layer for straight edge LINE entities.
            poly_layer: DXF layer for contour fallback LWPOLYLINE.
            hole_layer: DXF layer for CIRCLE entities.
        """
        if self.doc is None:
            self.create_drawing()

        ox, oy = offset

        # Ensure layers exist
        for layer in (line_layer, poly_layer, hole_layer):
            if layer not in self.doc.layers:
                self.doc.layers.add(layer)

        # LSD-detected straight segments → LINE entities
        for (x1, y1), (x2, y2) in getattr(geometric, "line_segments", []):
            self.msp.add_line(
                Vec3(x1 + ox, y1 + oy, 0),
                Vec3(x2 + ox, y2 + oy, 0),
                dxfattribs={"layer": line_layer, "color": colors.BLACK},
            )

        # Raw contour → LWPOLYLINE (fallback / curved sections)
        outline = getattr(geometric, "outline", None)
        if outline is not None and len(outline) >= 2:
            pts = np.asarray(outline).reshape(-1, 2)
            self.msp.add_lwpolyline(
                [(float(pts[i, 0]) + ox, float(pts[i, 1]) + oy) for i in range(len(pts))],
                close=True,
                dxfattribs={"layer": poly_layer, "color": colors.BLACK},
            )

        # Algebraically-fitted hole circles → CIRCLE entities
        for (cx, cy), r in getattr(geometric, "holes", []):
            self.msp.add_circle(
                Vec3(cx + ox, cy + oy, 0),
                float(r),
                dxfattribs={"layer": hole_layer, "color": colors.BLACK},
            )

    def save(self, filepath: str) -> None:
        """
        Save the DXF drawing to file.

        Args:
            filepath: Output file path
        """
        if self.doc is None:
            raise ValueError("No drawing created. Call create_drawing() first.")

        self.doc.saveas(filepath)
        logger.info(f"Saved CAD drawing to {filepath}")
