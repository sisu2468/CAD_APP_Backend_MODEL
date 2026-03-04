"""
Dimension Measurement Module
Extracts and measures dimensions from images and CAD drawings.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Dimension:
    """Represents a dimension measurement."""
    name: str
    value: float
    unit: str = "mm"
    start_point: Optional[Tuple[float, float]] = None
    end_point: Optional[Tuple[float, float]] = None
    tolerance: Optional[Tuple[float, float]] = None


class DimensionMeasurer:
    """Measures dimensions from images and extracted features."""
    
    def __init__(self, reference_length: Optional[float] = None):
        """
        Initialize dimension measurer.
        
        Args:
            reference_length: Known reference length in mm for calibration
        """
        self.reference_length = reference_length
        self.pixel_to_mm_ratio: Optional[float] = None
    
    def calibrate_from_reference(self, 
                                 reference_pixels: float,
                                 reference_length_mm: float) -> None:
        """
        Calibrate pixel-to-mm conversion using a known reference.
        
        Args:
            reference_pixels: Length in pixels
            reference_length_mm: Known length in mm
        """
        self.pixel_to_mm_ratio = reference_length_mm / reference_pixels
        logger.info(f"Calibrated: {self.pixel_to_mm_ratio:.4f} mm/pixel")
    
    def measure_contour_dimensions(self, 
                                   contour: np.ndarray) -> Dict[str, float]:
        """
        Measure dimensions from a contour.
        
        Args:
            contour: Contour array
            
        Returns:
            Dictionary of dimension measurements
        """
        dimensions = {}
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        if self.pixel_to_mm_ratio:
            dimensions['width'] = w * self.pixel_to_mm_ratio
            dimensions['height'] = h * self.pixel_to_mm_ratio
        else:
            dimensions['width'] = float(w)
            dimensions['height'] = float(h)
        
        # Area
        area = cv2.contourArea(contour)
        if self.pixel_to_mm_ratio:
            dimensions['area'] = area * (self.pixel_to_mm_ratio ** 2)
        else:
            dimensions['area'] = float(area)
        
        # Perimeter
        perimeter = cv2.arcLength(contour, True)
        if self.pixel_to_mm_ratio:
            dimensions['perimeter'] = perimeter * self.pixel_to_mm_ratio
        else:
            dimensions['perimeter'] = float(perimeter)
        
        return dimensions
    
    def measure_circle(self, 
                      center: Tuple[float, float],
                      radius: float) -> Dimension:
        """
        Measure a circle feature.
        
        Args:
            center: Circle center (x, y)
            radius: Circle radius in pixels
            
        Returns:
            Dimension object
        """
        diameter = radius * 2
        
        if self.pixel_to_mm_ratio:
            diameter_mm = diameter * self.pixel_to_mm_ratio
            radius_mm = radius * self.pixel_to_mm_ratio
        else:
            diameter_mm = diameter
            radius_mm = radius
        
        return Dimension(
            name="diameter",
            value=diameter_mm,
            unit="mm",
            start_point=(center[0] - radius, center[1]),
            end_point=(center[0] + radius, center[1])
        )
    
    def measure_hole_pitch(self,
                          holes: List[Tuple[float, float]],
                          reference_hole: int = 0) -> List[Dimension]:
        """
        Measure pitch (distance) between holes.
        
        Args:
            holes: List of (x, y) hole centers
            reference_hole: Index of reference hole
            
        Returns:
            List of dimension measurements
        """
        if len(holes) < 2:
            return []
        
        dimensions = []
        ref_hole = holes[reference_hole]
        
        for i, hole in enumerate(holes):
            if i == reference_hole:
                continue
            
            dx = hole[0] - ref_hole[0]
            dy = hole[1] - ref_hole[1]
            distance = np.sqrt(dx**2 + dy**2)
            
            if self.pixel_to_mm_ratio:
                distance_mm = distance * self.pixel_to_mm_ratio
            else:
                distance_mm = distance
            
            dimensions.append(Dimension(
                name=f"hole_pitch_{reference_hole}_to_{i}",
                value=distance_mm,
                unit="mm",
                start_point=ref_hole,
                end_point=hole
            ))
        
        return dimensions
    
    def measure_line_length(self,
                           start: Tuple[float, float],
                           end: Tuple[float, float]) -> Dimension:
        """
        Measure length of a line segment.
        
        Args:
            start: Start point (x, y)
            end: End point (x, y)
            
        Returns:
            Dimension object
        """
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = np.sqrt(dx**2 + dy**2)
        
        if self.pixel_to_mm_ratio:
            length_mm = length * self.pixel_to_mm_ratio
        else:
            length_mm = length
        
        return Dimension(
            name="length",
            value=length_mm,
            unit="mm",
            start_point=start,
            end_point=end
        )
