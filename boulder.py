# Install required libraries
!pip install scikit-image matplotlib numpy opencv-python rasterio

# --- IMPORTS ---
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import blob_log
from skimage import exposure, img_as_ubyte
from google.colab import files
import pandas as pd
import cv2
try:
    import rasterio
    from rasterio.plot import show
    RASTERIO_AVAILABLE = True
except ImportError:
    print("Warning: rasterio not available. DTM functionality will be limited.")
    RASTERIO_AVAILABLE = False

# --- Upload lunar image (TMC/OHRC) ---
print("Upload a lunar image (PNG/JPG) to detect boulders.")
uploaded = files.upload()
filename = list(uploaded.keys())[0]
print(f"Loaded: {filename}")

# --- Load DTM file from Google Drive (optional) ---
print("\nOptional: Load a DTM file (GeoTIFF/TIF) from Google Drive for enhanced boulder detection.")
print("DTM files provide elevation data that can improve detection accuracy.")
print("\nInstructions:")
print("1. Upload your DTM file to Google Drive first")
print("2. Mount Google Drive when prompted")
print("3. Provide the path to your DTM file in Google Drive")

# Import required modules
import os
from google.colab import drive

dtm_filename = None
dtm_data = None
dtm_transform = None

# Ask user if they want to use DTM from Google Drive
use_dtm = input("\nDo you want to load a DTM file from Google Drive? (y/n): ").lower()

if use_dtm == 'y':
    try:
        # Mount Google Drive
        print("\nMounting Google Drive...")
        drive.mount('/content/drive')
        print("Google Drive mounted successfully!")
        
        # List available files in Google Drive (optional helper)
        print("\nListing files in your Google Drive root directory:")
        try:
            drive_files = os.listdir('/content/drive/MyDrive')
            print("Available files/folders:")
            for i, file in enumerate(drive_files[:10]):  # Show first 10 items
                print(f"  {i+1}. {file}")
            if len(drive_files) > 10:
                print(f"  ... and {len(drive_files) - 10} more files/folders")
        except Exception as e:
            print(f"Could not list files: {e}")
        
        # Get DTM file path from user
        print("\nEnter the path to your DTM file in Google Drive:")
        print("Example: /content/drive/MyDrive/your_dtm_file.tif")
        print("Or just the filename if it's in the root: your_dtm_file.tif")
        
        dtm_path = input("DTM file path: ").strip()
        
        # Handle relative paths
        if not dtm_path.startswith('/content/drive/'):
            dtm_path = f'/content/drive/MyDrive/{dtm_path}'
        
        # Check if file exists
        if os.path.exists(dtm_path):
            dtm_filename = dtm_path
            file_size = os.path.getsize(dtm_path) / (1024*1024)  # Size in MB
            print(f"\nDTM file found: {os.path.basename(dtm_path)} ({file_size:.2f} MB)")
            
            # Read DTM data
            if RASTERIO_AVAILABLE:
                try:
                    print("Reading DTM data...")
                    with rasterio.open(dtm_path) as src:
                        dtm_data = src.read(1)  # Read first band
                        dtm_transform = src.transform
                        dtm_crs = src.crs
                        print(f"DTM shape: {dtm_data.shape}")
                        print(f"DTM data range: {dtm_data.min():.2f} to {dtm_data.max():.2f}")
                        print(f"DTM CRS: {dtm_crs}")
                        print("DTM file successfully loaded from Google Drive!")
                except Exception as e:
                    print(f"Error reading DTM file: {e}")
                    print("Proceeding with image-only detection...")
                    dtm_data = None
            else:
                print("rasterio not available. Cannot process DTM file.")
                print("Proceeding with image-only detection...")
                dtm_data = None
        else:
            print(f"DTM file not found at: {dtm_path}")
            print("Available files in the specified directory:")
            try:
                dir_path = os.path.dirname(dtm_path)
                if os.path.exists(dir_path):
                    files_in_dir = os.listdir(dir_path)
                    for file in files_in_dir:
                        if file.lower().endswith(('.tif', '.tiff')):
                            print(f"  - {file}")
                else:
                    print("  Directory does not exist")
            except Exception as e:
                print(f"  Could not list directory: {e}")
            print("Proceeding with image-only detection...")
            dtm_data = None
            
    except Exception as e:
        print(f"Error accessing Google Drive: {e}")
        print("Proceeding with image-only detection...")
        dtm_data = None
else:
    print("Skipping DTM loading. Proceeding with image-only detection...")

# --- Read and preprocess the image ---
img = imread(filename)

# Convert to grayscale if needed
if len(img.shape) == 3:
    if img.shape[2] == 4:  # RGBA image
        img_rgb = img[:, :, :3]
        img_gray = rgb2gray(img_rgb)
    elif img.shape[2] == 3:  # RGB image
        img_gray = rgb2gray(img)
    else:
        print(f"Unexpected number of channels: {img.shape[2]}")
        img_gray = img
else:
    # Assume already grayscale and normalize if necessary
    if img.max() > 1.001 or img.min() < -0.001:
        img_gray = img / img.max()
    else:
        img_gray = img

# Enhance contrast
img_eq = exposure.equalize_adapthist(img_gray, clip_limit=0.03)

# --- DTM Processing for Enhanced Detection ---
dtm_processed = None
if dtm_data is not None:
    print("Processing DTM data for enhanced boulder detection...")
    
    # Resize DTM to match image dimensions if needed
    if dtm_data.shape != img_gray.shape:
        print(f"Resizing DTM from {dtm_data.shape} to {img_gray.shape}")
        dtm_resized = cv2.resize(dtm_data, (img_gray.shape[1], img_gray.shape[0]), 
                                interpolation=cv2.INTER_LINEAR)
    else:
        dtm_resized = dtm_data.copy()
    
    # Normalize DTM data
    dtm_normalized = (dtm_resized - dtm_resized.min()) / (dtm_resized.max() - dtm_resized.min())
    
    # Calculate DTM gradient (slope) to identify elevated features
    dtm_gradient_x = cv2.Sobel(dtm_normalized, cv2.CV_64F, 1, 0, ksize=3)
    dtm_gradient_y = cv2.Sobel(dtm_normalized, cv2.CV_64F, 0, 1, ksize=3)
    dtm_gradient_magnitude = np.sqrt(dtm_gradient_x*2 + dtm_gradient_y*2)
    
    # Calculate local elevation maxima (potential boulder locations)
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    dtm_local_mean = cv2.filter2D(dtm_normalized, -1, kernel)
    dtm_elevation_difference = dtm_normalized - dtm_local_mean
    
    # Combine DTM information with original image
    # Weight the combination based on elevation difference and gradient
    dtm_weight = 0.3  # Adjust this weight as needed
    img_combined = img_eq * (1 - dtm_weight) + dtm_elevation_difference * dtm_weight
    
    # Apply additional enhancement based on gradient
    gradient_threshold = np.percentile(dtm_gradient_magnitude, 75)  # Top 25% of gradients
    gradient_mask = dtm_gradient_magnitude > gradient_threshold
    img_combined[gradient_mask] = np.minimum(img_combined[gradient_mask] * 1.2, 1.0)
    
    dtm_processed = {
        'elevation': dtm_normalized,
        'gradient': dtm_gradient_magnitude,
        'elevation_diff': dtm_elevation_difference,
        'combined_image': img_combined
    }
    
    print("DTM processing complete. Using enhanced detection.")
else:
    print("No DTM data available. Using standard detection.")
    img_combined = img_eq

# --- Parameters for LoG blob detection ---
# Adjust parameters based on whether DTM is available
if dtm_data is not None:
    # More sensitive detection with DTM
    min_sigma = 1.5
    max_sigma = 12
    num_sigma = 15
    threshold = 0.03  # Lower threshold for more sensitive detection
else:
    # Standard parameters
    min_sigma = 3
    max_sigma = 8
    num_sigma = 8
    threshold = 0.07

print(f"Detection parameters: min_sigma={min_sigma}, max_sigma={max_sigma}, threshold={threshold}")

# --- Detect Boulders using Laplacian of Gaussian (LoG) ---
blobs_log = blob_log(img_combined, min_sigma=min_sigma, max_sigma=max_sigma, 
                    num_sigma=num_sigma, threshold=threshold)

# Compute radii in pixels from the sigma values
blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)

# --- Filter blobs based on DTM data (if available) ---
if dtm_processed is not None:
    filtered_blobs = []
    elevation_threshold = np.percentile(dtm_processed['elevation_diff'], 60)  # Top 40% elevation difference
    
    for blob in blobs_log:
        y, x, r = blob
        y_int, x_int = int(y), int(x)
        
        # Check if the blob is at a locally elevated position
        if (0 <= y_int < dtm_processed['elevation_diff'].shape[0] and 
            0 <= x_int < dtm_processed['elevation_diff'].shape[1]):
            
            elevation_value = dtm_processed['elevation_diff'][y_int, x_int]
            gradient_value = dtm_processed['gradient'][y_int, x_int]
            
            # Keep blobs that are elevated or have significant gradient
            if elevation_value > elevation_threshold or gradient_value > np.percentile(dtm_processed['gradient'], 70):
                filtered_blobs.append(blob)
    
    blobs_log = np.array(filtered_blobs) if filtered_blobs else np.array([]).reshape(0, 3)
    print(f"Filtered {len(blobs_log)} potential boulders using DTM data")

# --- Analyze shape and extract detailed info ---
boulder_details = []
img_eq_uint8 = img_as_ubyte(img_combined)

for blob in blobs_log:
    y, x, r = blob
    y_int, x_int, r_int = int(y), int(x), int(r)

    # Define the bounding box around the blob
    padding = int(r * 0.5)
    y_min = max(0, y_int - r_int - padding)
    y_max = min(img_eq_uint8.shape[0], y_int + r_int + padding)
    x_min = max(0, x_int - r_int - padding)
    x_max = min(img_eq_uint8.shape[1], x_int + r_int + padding)

    # Extract the image patch
    patch = img_eq_uint8[y_min:y_max, x_min:x_max]

    # Initialize DTM-based measurements
    elevation_info = {}
    if dtm_processed is not None and 0 <= y_int < dtm_processed['elevation'].shape[0] and 0 <= x_int < dtm_processed['elevation'].shape[1]:
        elevation_info = {
            'elevation_value': dtm_processed['elevation'][y_int, x_int],
            'elevation_difference': dtm_processed['elevation_diff'][y_int, x_int],
            'gradient_magnitude': dtm_processed['gradient'][y_int, x_int]
        }

    if patch.size > 0:
        # Create a binary mask for the blob
        mask = np.zeros_like(patch, dtype=np.uint8)
        patch_center_x = x_int - x_min
        patch_center_y = y_int - y_min
        cv2.circle(mask, (patch_center_x, patch_center_y), r_int, 255, -1)

        if np.sum(mask) > 0:
            moments = cv2.moments(mask, True)

            if moments['m00'] != 0:
                # Calculate covariance matrix for shape analysis
                mu20 = moments['mu20'] / moments['m00']
                mu02 = moments['mu02'] / moments['m00']
                mu11 = moments['mu11'] / moments['m00']

                cov_matrix = np.array([[mu20, mu11], [mu11, mu02]])
                eigenvalues = np.linalg.eigvalsh(cov_matrix)
                eigenvalues = np.sort(eigenvalues)

                # Estimate dimensions
                approx_length = 4 * np.sqrt(eigenvalues[1]) if eigenvalues[1] > 0 else 0
                approx_diameter = 4 * np.sqrt(eigenvalues[0]) if eigenvalues[0] > 0 else 0

                if approx_diameter > approx_length:
                    approx_length, approx_diameter = approx_diameter, approx_length

                boulder_detail = {
                    'Y_pixel': y,
                    'X_pixel': x,
                    'Radius_pixel': r,
                    'Estimated_Length_pixel': approx_length,
                    'Estimated_Diameter_pixel': approx_diameter
                }
                boulder_detail.update(elevation_info)
                boulder_details.append(boulder_detail)
            else:
                boulder_detail = {
                    'Y_pixel': y,
                    'X_pixel': x,
                    'Radius_pixel': r,
                    'Estimated_Length_pixel': r * 2,
                    'Estimated_Diameter_pixel': r * 2
                }
                boulder_detail.update(elevation_info)
                boulder_details.append(boulder_detail)
        else:
            boulder_detail = {
                'Y_pixel': y,
                'X_pixel': x,
                'Radius_pixel': r,
                'Estimated_Length_pixel': r * 2,
                'Estimated_Diameter_pixel': r * 2
            }
            boulder_detail.update(elevation_info)
            boulder_details.append(boulder_detail)
    else:
        boulder_detail = {
            'Y_pixel': y,
            'X_pixel': x,
            'Radius_pixel': r,
            'Estimated_Length_pixel': r * 2,
            'Estimated_Diameter_pixel': r * 2
        }
        boulder_detail.update(elevation_info)
        boulder_details.append(boulder_detail)

# Create DataFrame
df_boulders = pd.DataFrame(boulder_details)

# --- Visualization ---
if dtm_processed is not None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # Original image with detections
    axes[0, 0].imshow(img_eq, cmap='gray')
    for index, row in df_boulders.iterrows():
        y, x, r = row['Y_pixel'], row['X_pixel'], row['Radius_pixel']
        c = plt.Circle((x, y), r, color='red', linewidth=1.5, fill=False)
        axes[0, 0].add_patch(c)
    axes[0, 0].set_title(f'Original Image with Detections: {len(df_boulders)}', fontsize=12)
    axes[0, 0].axis('on')
    
    # DTM elevation
    im1 = axes[0, 1].imshow(dtm_processed['elevation'], cmap='terrain')
    axes[0, 1].set_title('DTM Elevation', fontsize=12)
    plt.colorbar(im1, ax=axes[0, 1])
    
    # DTM gradient
    im2 = axes[1, 0].imshow(dtm_processed['gradient'], cmap='hot')
    axes[1, 0].set_title('DTM Gradient Magnitude', fontsize=12)
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Combined enhanced image
    axes[1, 1].imshow(img_combined, cmap='gray')
    for index, row in df_boulders.iterrows():
        y, x, r = row['Y_pixel'], row['X_pixel'], row['Radius_pixel']
        c = plt.Circle((x, y), r, color='cyan', linewidth=1.5, fill=False)
        axes[1, 1].add_patch(c)
    axes[1, 1].set_title('Enhanced Detection (Image + DTM)', fontsize=12)
    axes[1, 1].axis('on')
    
    plt.tight_layout()
    plt.show()
else:
    # Standard visualization
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img_eq, cmap='gray')
    for index, row in df_boulders.iterrows():
        y, x, r = row['Y_pixel'], row['X_pixel'], row['Radius_pixel']
        c = plt.Circle((x, y), r, color='red', linewidth=1.5, fill=False)
        ax.add_patch(c)
    ax.set_title(f'Detected Boulders (LoG): {len(df_boulders)}', fontsize=15)
    plt.axis('on')
    plt.show()

# --- Export results ---
csv_filename = "detected_boulders_enhanced.csv"
df_boulders.to_csv(csv_filename, index=False)
files.download(csv_filename)

print(f"\nDetected {len(df_boulders)} boulders.")
print(f"Enhanced boulder data saved to {csv_filename} and is available for download.")

# Print summary statistics
if len(df_boulders) > 0:
    print(f"\nBoulder Detection Summary:")
    print(f"- Total boulders detected: {len(df_boulders)}")
    print(f"- Average radius: {df_boulders['Radius_pixel'].mean():.2f} pixels")
    print(f"- Radius range: {df_boulders['Radius_pixel'].min():.2f} - {df_boulders['Radius_pixel'].max():.2f} pixels")
    
    if dtm_processed is not None:
        print(f"- Average elevation difference: {df_boulders['elevation_difference'].mean():.4f}")
        print(f"- Average gradient magnitude: {df_boulders['gradient_magnitude'].mean():.4f}")
