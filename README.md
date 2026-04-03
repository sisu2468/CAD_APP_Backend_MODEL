# CAD Drawing Extraction System

A professional Python system for extracting accurate CAD drawings from smartphone images of machine parts. This system processes multiple views of a part and generates technical drawings in DXF and PDF formats.

## Features

- **Multi-View Reconstruction**: Combines information from multiple images taken from different angles
- **Geometric Feature Extraction**: Detects edges, contours, circles, rectangles, and complex shapes
- **CAD Drawing Generation**: Creates professional technical drawings with orthographic projections
- **PDF Export**: Generates PDF documents suitable for technical documentation
- **Dimension Measurement**: Extracts and measures dimensions from images
- **High Accuracy**: Designed for precise extraction of machine part geometry

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd CAD_APP_Backend_Model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Process a single image:
```bash
python main.py IMG_5047.jpg -n output_drawing
```

Process multiple views:
```bash
python main.py IMG_8843.jpg IMG_8844.jpg --views front side -n output_drawing
```

Process with custom output directory:
```bash
python main.py IMG_8840.jpg IMG_8841.jpg -o ./results -n part_001
```

### Python API

```python
from src.pipeline import CADExtractionPipeline

# Create pipeline
pipeline = CADExtractionPipeline(
    output_dir="output",
    enable_dimensions=True,
    enable_pdf=True
)

# Process images
results = pipeline.process_images(
    image_paths=["IMG_8843.jpg", "IMG_8844.jpg"],
    view_types=["front", "side"],
    output_name="my_part"
)

print(f"DXF: {results['dxf']}")
print(f"PDF: {results['pdf']}")
```

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── image_processor.py      # Image preprocessing and feature extraction
│   ├── multi_view_reconstructor.py  # Multi-view geometry reconstruction
│   ├── cad_generator.py        # CAD drawing generation (DXF)
│   ├── pdf_exporter.py         # PDF export functionality
│   ├── dimension_measurement.py # Dimension measurement tools
│   └── pipeline.py             # Main orchestration pipeline
├── main.py                     # Command-line interface
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Modules

### ImageProcessor
Handles image preprocessing, edge detection, and geometric feature extraction:
- Edge detection using Canny algorithm
- Contour extraction
- Circle detection (Hough Transform)
- Line detection
- Shape classification

### MultiViewReconstructor
Combines information from multiple views:
- View registration and alignment
- 3D geometry reconstruction
- Feature combination from multiple angles

### CADGenerator
Generates CAD drawings in DXF format:
- Orthographic projections (front, side, top views)
- Feature representation (holes, chamfers, etc.)
- Dimension annotations

### PDFExporter
Converts DXF drawings to PDF:
- Professional technical drawing layout
- Proper scaling and positioning
- Title and annotation support

## Requirements

- Python 3.8+
- OpenCV 4.8+
- NumPy 1.24+
- ezdxf 1.1+
- ReportLab 4.0+
- Pillow 10.0+

## Future Enhancements

- [ ] Machine learning model for improved accuracy
- [ ] Surface roughness measurement
- [ ] Hole pitch measurement automation
- [ ] 3D model generation (STEP/STL export)
- [ ] Web API interface
- [ ] Mobile app integration

## Contributing

This is a professional system designed for manufacturing environments. Code contributions should maintain high quality standards and include appropriate tests.

## License

[Specify your license here]

## Contact

For questions or support, please contact [your contact information].
