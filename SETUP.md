# Setup Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Installation

Run the test script to verify all imports work:

```bash
python test_imports.py
```

### 3. Test with Example Images

```bash
python example_usage.py
```

Or process your own images:

```bash
python main.py IMG_8840.jpg -n my_output
```

## Project Structure

```
CAD_APP_Backend_Model/
├── src/                          # Source code modules
│   ├── image_processor.py       # Image preprocessing and feature extraction
│   ├── multi_view_reconstructor.py  # Multi-view geometry reconstruction
│   ├── cad_generator.py         # CAD drawing generation
│   ├── pdf_exporter.py          # PDF export
│   ├── dimension_measurement.py # Dimension measurement tools
│   ├── pipeline.py              # Main orchestration pipeline
│   └── training/                # ML training modules (future)
├── main.py                      # Command-line interface
├── app.py                       # Flask API server
├── example_usage.py             # Usage examples
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
└── README.md                    # Documentation
```

## Usage

### Command Line

Process single image:
```bash
python main.py path/to/image.jpg -n output_name
```

Process multiple views:
```bash
python main.py img1.jpg img2.jpg --views front side -n output_name
```

### Python API

```python
from src.pipeline import CADExtractionPipeline

pipeline = CADExtractionPipeline()
results = pipeline.process_images(
    image_paths=["img1.jpg", "img2.jpg"],
    view_types=["front", "side"],
    output_name="my_part"
)
```

### Flask API

Start the server:
```bash
python app.py
```

Then use the API endpoints:
- `POST /process-image` - Process single image
- `POST /process-multi-view` - Process multiple views
- `POST /process-with-pdf` - Process and return PDF
- `GET /health` - Health check

## Troubleshooting

### Import Errors

If you get import errors, make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Image Not Found Errors

Ensure image paths are correct and files exist. Use absolute paths if relative paths don't work.

### Output Directory Issues

The system will create output directories automatically. If you encounter permission errors, ensure you have write access to the output directory.

## Next Steps

1. Process your own images using the examples
2. Adjust configuration in `config.py` for your use case
3. Integrate with your mobile app using the Flask API
4. Train ML models using the training module structure (future enhancement)
