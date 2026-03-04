"""
Image Processing Module
Handles preprocessing, edge detection, and feature extraction from input images.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ImageFeatures:
    """Container for extracted image features."""
    edges: np.ndarray
    contours: List[np.ndarray]
    circles: List[Tuple[float, float, float]]  # (center_x, center_y, radius)
    rectangles: List[np.ndarray]  # 4 corner points
    lines: List[Tuple[float, float, float, float]]  # (x1, y1, x2, y2)
    polygons: List[np.ndarray]
    dimensions: Dict[str, float]


class ImageProcessor:
    """Advanced image processor for extracting geometric features from machine part images."""
    
    def __init__(self, 
                 canny_low: int = 50,
                 canny_high: int = 150,
                 gaussian_kernel: Tuple[int, int] = (5, 5),
                 min_contour_area: int = 100):
        """
        Initialize the image processor.
        
        Args:
            canny_low: Lower threshold for Canny edge detection
            canny_high: Upper threshold for Canny edge detection
            gaussian_kernel: Kernel size for Gaussian blur
            min_contour_area: Minimum contour area to consider
        """
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.gaussian_kernel = gaussian_kernel
        self.min_contour_area = min_contour_area
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess an image for feature extraction.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Preprocessed grayscale image
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, self.gaussian_kernel, 0)
        
        # Apply adaptive thresholding for better edge detection
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        return adaptive_thresh
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Detect edges in the image using Canny edge detection.
        
        Args:
            image: Preprocessed grayscale image
            
        Returns:
            Binary image with detected edges
        """
        edges = cv2.Canny(image, self.canny_low, self.canny_high)
        
        # Apply morphological operations to close gaps
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return edges
    
    def extract_contours(self, edges: np.ndarray) -> List[np.ndarray]:
        """
        Extract contours from edge-detected image.
        
        Args:
            edges: Binary edge image
            
        Returns:
            List of contours
        """
        contours, hierarchy = cv2.findContours(
            edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours by area
        filtered_contours = [
            cnt for cnt in contours 
            if cv2.contourArea(cnt) > self.min_contour_area
        ]
        
        return filtered_contours
    
    def detect_circles(self, image: np.ndarray, 
                      min_radius: int = 5,
                      max_radius: int = 500) -> List[Tuple[float, float, float]]:
        """
        Detect circular features (holes, rounded edges) using Hough Circle Transform.
        
        Args:
            image: Preprocessed image
            min_radius: Minimum circle radius
            max_radius: Maximum circle radius
            
        Returns:
            List of (center_x, center_y, radius) tuples
        """
        circles = cv2.HoughCircles(
            image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            return [(c[0], c[1], c[2]) for c in circles]
        return []
    
    def detect_lines(self, edges: np.ndarray, 
                    min_line_length: int = 50,
                    max_line_gap: int = 10) -> List[Tuple[float, float, float, float]]:
        """
        Detect straight lines using Probabilistic Hough Line Transform.
        
        Args:
            edges: Binary edge image
            min_line_length: Minimum line length
            max_line_gap: Maximum gap between line segments
            
        Returns:
            List of (x1, y1, x2, y2) tuples
        """
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=100,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )
        
        if lines is not None:
            return [(line[0][0], line[0][1], line[0][2], line[0][3]) 
                   for line in lines]
        return []
    
    def classify_shapes(self, contours: List[np.ndarray]) -> Dict[str, List[np.ndarray]]:
        """
        Classify contours into geometric shapes.
        
        Args:
            contours: List of contours
            
        Returns:
            Dictionary with classified shapes
        """
        shapes = {
            'rectangles': [],
            'circles': [],
            'polygons': []
        }
        
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Classify based on number of vertices
            if len(approx) == 4:
                # Check if it's approximately rectangular
                area = cv2.contourArea(approx)
                x, y, w, h = cv2.boundingRect(approx)
                rect_area = w * h
                if area / rect_area > 0.8:  # 80% of bounding rectangle
                    shapes['rectangles'].append(approx)
                else:
                    shapes['polygons'].append(approx)
            elif len(approx) > 8:
                # Check if it's approximately circular
                area = cv2.contourArea(approx)
                (x, y), radius = cv2.minEnclosingCircle(contour)
                circle_area = np.pi * radius * radius
                if area / circle_area > 0.7:  # 70% of enclosing circle
                    shapes['circles'].append(approx)
                else:
                    shapes['polygons'].append(approx)
            else:
                shapes['polygons'].append(approx)
        
        return shapes
    
    def extract_features(self, image_path: str) -> ImageFeatures:
        """
        Extract all geometric features from an image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            ImageFeatures object containing all extracted features
        """
        logger.info(f"Processing image: {image_path}")
        
        # Preprocess
        preprocessed = self.preprocess_image(image_path)
        
        # Detect edges
        edges = self.detect_edges(preprocessed)
        
        # Extract contours
        contours = self.extract_contours(edges)
        
        # Detect circles
        circles = self.detect_circles(preprocessed)
        
        # Detect lines
        lines = self.detect_lines(edges)
        
        # Classify shapes
        shapes = self.classify_shapes(contours)
        
        # Extract dimensions (basic measurements)
        dimensions = self._extract_dimensions(contours, preprocessed.shape)
        
        return ImageFeatures(
            edges=edges,
            contours=contours,
            circles=circles,
            rectangles=shapes['rectangles'],
            lines=lines,
            polygons=shapes['polygons'],
            dimensions=dimensions
        )
    
    def _extract_dimensions(self, contours: List[np.ndarray], 
                           image_shape: Tuple[int, int]) -> Dict[str, float]:
        """
        Extract basic dimensions from contours.
        
        Args:
            contours: List of contours
            image_shape: Shape of the image (height, width)
            
        Returns:
            Dictionary of dimension measurements
        """
        dimensions = {}
        
        if contours:
            # Find largest contour (likely the main part)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            dimensions['width'] = float(w)
            dimensions['height'] = float(h)
            dimensions['area'] = float(cv2.contourArea(largest_contour))
        
        return dimensions
