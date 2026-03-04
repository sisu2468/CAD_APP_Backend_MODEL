import cv2
import ezdxf

# Function to process the image and export DXF
def process_image_and_export_dxf(image_path, dxf_path='output.dxf'):
    # Load the image
    img = cv2.imread(image_path)

    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply Gaussian blur to remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Step 4: Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Debugging: Visualize the contours (optional)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    cv2.imshow('Contours', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Step 5: Create DXF file and export the contours
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    for contour in contours:
        points = contour[:, 0, :]  # Extract x, y points from contour
        polyline = msp.add_lwpolyline(points, close=True)

    # Save the DXF file
    doc.saveas(dxf_path)
    print(f"DXF file saved: {dxf_path}")
    return dxf_path

# Example usage
if __name__ == '__main__':
    input_image_path = 'IMG_5402.jpg'  # Replace with your image file path
    output_dxf_path = 'output.dxf'
    process_image_and_export_dxf(input_image_path, output_dxf_path)
