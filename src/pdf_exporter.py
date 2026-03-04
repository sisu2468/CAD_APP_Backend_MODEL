"""
PDF Export Module
Converts CAD drawings to PDF format for technical documentation.
"""

import ezdxf
from ezdxf import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib import colors as rl_colors
from reportlab.lib.utils import ImageReader
import io
import numpy as np
from PIL import Image, ImageDraw
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PDFExporter:
    """Exports CAD drawings to PDF format."""
    
    def __init__(self, page_size: str = 'A4', margin: float = 20.0):
        """
        Initialize PDF exporter.
        
        Args:
            page_size: Page size ('A4' or 'letter')
            margin: Margin in mm
        """
        self.page_size = A4 if page_size == 'A4' else letter
        self.margin = margin * mm
        self.width, self.height = self.page_size
    
    def export_from_dxf(self, dxf_path: str, pdf_path: str,
                       title: str = "CAD Drawing") -> None:
        """
        Export DXF file to PDF.
        
        Args:
            dxf_path: Path to input DXF file
            pdf_path: Path to output PDF file
            title: Title for the PDF document
        """
        # Read DXF file
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        
        # Create PDF canvas
        c = canvas.Canvas(pdf_path, pagesize=self.page_size)
        
        # Add title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(self.margin, self.height - self.margin - 20, title)
        
        # Convert DXF entities to PDF
        self._draw_dxf_entities(c, msp)
        
        # Save PDF
        c.save()
        logger.info(f"Exported PDF to {pdf_path}")
    
    def _draw_dxf_entities(self, canvas_obj: canvas.Canvas, 
                          modelspace) -> None:
        """
        Draw DXF entities on PDF canvas.
        
        Args:
            canvas_obj: ReportLab canvas object
            modelspace: DXF modelspace
        """
        # Scale factor to fit drawing on page
        # Get bounding box of all entities
        bbox = self._get_bounding_box(modelspace)
        
        if bbox is None:
            logger.warning("No entities found in DXF file")
            return
        
        # Calculate scale to fit
        bbox_width = bbox[1][0] - bbox[0][0]
        bbox_height = bbox[1][1] - bbox[0][1]
        
        if bbox_width == 0 or bbox_height == 0:
            scale = 1.0
        else:
            scale_x = (self.width - 2 * self.margin) / bbox_width
            scale_y = (self.height - 2 * self.margin - 40) / bbox_height
            scale = min(scale_x, scale_y) * 0.9  # 90% to add some margin
        
        # Offset to center
        offset_x = self.margin - bbox[0][0] * scale
        offset_y = self.margin - bbox[0][1] * scale
        
        # Draw entities
        for entity in modelspace:
            if entity.dxftype() == 'LINE':
                self._draw_line(canvas_obj, entity, scale, offset_x, offset_y)
            elif entity.dxftype() == 'CIRCLE':
                self._draw_circle(canvas_obj, entity, scale, offset_x, offset_y)
            elif entity.dxftype() == 'LWPOLYLINE':
                self._draw_polyline(canvas_obj, entity, scale, offset_x, offset_y)
            elif entity.dxftype() == 'ARC':
                self._draw_arc(canvas_obj, entity, scale, offset_x, offset_y)
            elif entity.dxftype() == 'TEXT':
                self._draw_text(canvas_obj, entity, scale, offset_x, offset_y)
    
    def _get_bounding_box(self, modelspace) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Get bounding box of all entities."""
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        
        has_entities = False
        
        for entity in modelspace:
            if entity.dxftype() == 'LINE':
                start = entity.dxf.start
                end = entity.dxf.end
                min_x = min(min_x, start.x, end.x)
                min_y = min(min_y, start.y, end.y)
                max_x = max(max_x, start.x, end.x)
                max_y = max(max_y, start.y, end.y)
                has_entities = True
            elif entity.dxftype() == 'CIRCLE':
                center = entity.dxf.center
                radius = entity.dxf.radius
                min_x = min(min_x, center.x - radius)
                min_y = min(min_y, center.y - radius)
                max_x = max(max_x, center.x + radius)
                max_y = max(max_y, center.y + radius)
                has_entities = True
            elif entity.dxftype() == 'LWPOLYLINE':
                points = entity.get_points()
                for point in points:
                    min_x = min(min_x, point[0])
                    min_y = min(min_y, point[1])
                    max_x = max(max_x, point[0])
                    max_y = max(max_y, point[1])
                has_entities = True
        
        if not has_entities:
            return None
        
        return ((min_x, min_y), (max_x, max_y))
    
    def _draw_line(self, canvas_obj: canvas.Canvas, entity, 
                  scale: float, offset_x: float, offset_y: float) -> None:
        """Draw a line entity."""
        start = entity.dxf.start
        end = entity.dxf.end
        
        x1 = start.x * scale + offset_x
        y1 = start.y * scale + offset_y
        x2 = end.x * scale + offset_x
        y2 = end.y * scale + offset_y
        
        # Flip Y coordinate (DXF uses bottom-left origin, PDF uses top-left)
        y1 = self.height - y1
        y2 = self.height - y2
        
        canvas_obj.line(x1, y1, x2, y2)
    
    def _draw_circle(self, canvas_obj: canvas.Canvas, entity,
                    scale: float, offset_x: float, offset_y: float) -> None:
        """Draw a circle entity."""
        center = entity.dxf.center
        radius = entity.dxf.radius
        
        x = center.x * scale + offset_x
        y = center.y * scale + offset_y
        r = radius * scale
        
        # Flip Y coordinate
        y = self.height - y
        
        canvas_obj.circle(x, y, r)
    
    def _draw_polyline(self, canvas_obj: canvas.Canvas, entity,
                      scale: float, offset_x: float, offset_y: float) -> None:
        """Draw a polyline entity."""
        points = entity.get_points()
        
        if len(points) < 2:
            return
        
        path = canvas_obj.beginPath()
        
        first_point = points[0]
        x = first_point[0] * scale + offset_x
        y = first_point[1] * scale + offset_y
        y = self.height - y
        path.moveTo(x, y)
        
        for point in points[1:]:
            x = point[0] * scale + offset_x
            y = point[1] * scale + offset_y
            y = self.height - y
            path.lineTo(x, y)
        
        if entity.is_closed:
            path.close()
        
        canvas_obj.drawPath(path)
    
    def _draw_arc(self, canvas_obj: canvas.Canvas, entity,
                 scale: float, offset_x: float, offset_y: float) -> None:
        """Draw an arc entity."""
        center = entity.dxf.center
        radius = entity.dxf.radius
        start_angle = entity.dxf.start_angle
        end_angle = entity.dxf.end_angle
        
        x = center.x * scale + offset_x
        y = center.y * scale + offset_y
        r = radius * scale
        
        # Flip Y coordinate
        y = self.height - y
        
        # Convert angles (DXF uses degrees, ReportLab uses degrees)
        # ReportLab arcs go counterclockwise from start_angle
        canvas_obj.arc(x - r, y - r, x + r, y + r, 
                       start_angle, end_angle - start_angle)
    
    def _draw_text(self, canvas_obj: canvas.Canvas, entity,
                  scale: float, offset_x: float, offset_y: float) -> None:
        """Draw a text entity."""
        text = entity.dxf.text
        position = entity.dxf.insert
        
        x = position.x * scale + offset_x
        y = position.y * scale + offset_y
        y = self.height - y
        
        height = entity.dxf.height * scale if hasattr(entity.dxf, 'height') else 10
        
        canvas_obj.setFont("Helvetica", height)
        canvas_obj.drawString(x, y, text)
