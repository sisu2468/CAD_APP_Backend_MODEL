"""
Flask API for CAD Extraction System
Provides REST API endpoints for processing images and generating CAD drawings.
"""

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os
import tempfile
import logging
from pathlib import Path
from werkzeug.utils import secure_filename

from src.pipeline import CADExtractionPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create directories
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

# Initialize pipeline
pipeline = CADExtractionPipeline(
    output_dir=OUTPUT_FOLDER,
    enable_dimensions=True,
    enable_pdf=True
)


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def index():
    """API information endpoint."""
    return jsonify({
        'name': 'CAD Extraction API',
        'version': '1.0.0',
        'endpoints': {
            'GET /': 'API information',
            'GET /health': 'Health check',
            'POST /process-image': 'Process single image (params: image, output_name)',
            'POST /process-multi-view': 'Process multiple views (params: images, view_types, output_name)',
            'POST /process-with-pdf': 'Process and return PDF (params: image, output_name)'
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'message': 'CAD Extraction API is running'})


@app.route('/process-image', methods=['POST'])
def process_image():
    """
    Process a single image and generate CAD drawing.
    
    Expected form data:
    - image: Image file (required)
    - output_name: Optional output file name
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Get output name from request or use default
        output_name = request.form.get('output_name', 'cad_output')
        
        # Process image
        results = pipeline.process_single_image(
            image_path=filepath,
            output_name=output_name
        )
        
        # Return DXF file
        return send_file(
            results['dxf'],
            as_attachment=True,
            download_name=f"{output_name}.dxf",
            mimetype='application/dxf'
        )
    
    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/process-multi-view', methods=['POST'])
def process_multi_view():
    """
    Process multiple images from different views and generate CAD drawing.
    
    Expected form data:
    - images: Multiple image files (required)
    - view_types: Optional comma-separated list of view types (e.g., "front,side,top")
    - output_name: Optional output file name
    """
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'No image files provided'}), 400
        
        files = request.files.getlist('images')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No files selected'}), 400
        
        # Save uploaded files
        image_paths = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                image_paths.append(filepath)
        
        if not image_paths:
            return jsonify({'error': 'No valid image files provided'}), 400
        
        # Get view types
        view_types_str = request.form.get('view_types', '')
        view_types = [v.strip() for v in view_types_str.split(',')] if view_types_str else None
        
        # Get output name
        output_name = request.form.get('output_name', 'cad_output')
        
        # Process images
        results = pipeline.process_images(
            image_paths=image_paths,
            view_types=view_types,
            output_name=output_name
        )
        
        # Return DXF file
        return send_file(
            results['dxf'],
            as_attachment=True,
            download_name=f"{output_name}.dxf",
            mimetype='application/dxf'
        )
    
    except Exception as e:
        logger.error(f"Error processing multi-view images: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/process-with-pdf', methods=['POST'])
def process_with_pdf():
    """
    Process images and return both DXF and PDF files as a zip archive.
    Note: This is a simplified version - in production, you'd want to zip the files.
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Get output name
        output_name = request.form.get('output_name', 'cad_output')
        
        # Process image
        results = pipeline.process_single_image(
            image_path=filepath,
            output_name=output_name
        )
        
        # Return PDF if available, otherwise DXF
        if results.get('pdf'):
            return send_file(
                results['pdf'],
                as_attachment=True,
                download_name=f"{output_name}.pdf",
                mimetype='application/pdf'
            )
        else:
            return send_file(
                results['dxf'],
                as_attachment=True,
                download_name=f"{output_name}.dxf",
                mimetype='application/dxf'
            )
    
    except Exception as e:
        logger.error(f"Error processing with PDF: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import os
    
    # Fix Windows socket error
    os.environ.pop("WERKZEUG_SERVER_FD", None)
    os.environ.pop("WERKZEUG_RUN_MAIN", None)
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
