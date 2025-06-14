import cv2
import os
import numpy as np
import pickle
import csv
from ultralytics import YOLO
from datetime import datetime
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import time

# ------------------ CONFIGURATION ------------------
KNOWN_FACES_DIR = r"D:\face\non_face" ## path to the folder containing the known faces
ENCODINGS_FILE = "face_arc_encodings.pkl" ## path to the file containing the encodings
UNKNOWN_IMAGE_PATH = r"C:\Users\Hp\Downloads\20250529_151504.jpg" ## path to the image t    o be recognized
YOLO_MODEL_PATH = r"D:\face\best.pt" ## path to the YOLO custom trained model which detect faces 
ATTENDANCE_CSV = "attendance.csv" ## path to the file containing the attendance 
SIMILARITY_THRESHOLD = 0.40  # Lowered threshold for better matching    
DEBUG = True  # Set to True for verbose logging (if you want to see the logs)
# ---------------------------------------------------

def log(message):
    """Custom logging function that only prints when DEBUG is True"""
    if DEBUG:
        print(message)

# Initialize ArcFace embedding model with proper configuration
log("🚀 Initializing ArcFace model...")
face_analyzer = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load YOLO model
log("🚀 Loading YOLO model...")
model = YOLO(YOLO_MODEL_PATH)

# Function to extract name from folder structure
def get_name_from_id(person_id):
    """Extract name from the first image filename in the person's folder"""
    try:
        person_path = os.path.join(KNOWN_FACES_DIR, person_id)
        if os.path.isdir(person_path):
            for filename in os.listdir(person_path):
                if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                    # Extract name without extension and non-alphabetic characters
                    return ''.join(filter(str.isalpha, os.path.splitext(filename)[0]))
    except Exception as e:
        log(f"⚠️ Error getting name for ID {person_id}: {e}")
    
    return "Unknown"  # Default if no name found

# Create ID to name mapping
id_to_name = {}
for person_id in os.listdir(KNOWN_FACES_DIR):
    if os.path.isdir(os.path.join(KNOWN_FACES_DIR, person_id)):
        id_to_name[person_id] = get_name_from_id(person_id)

log(f"📋 Found {len(id_to_name)} IDs with names in the dataset")

# Load or generate known embeddings
if os.path.exists(ENCODINGS_FILE):
    log("🔍 Loading cached ArcFace embeddings...")
    try:
        with open(ENCODINGS_FILE, "rb") as file:
            known_encodings = pickle.load(file)
        log(f"✅ Loaded {len(known_encodings)} known encodings")
    except Exception as e:
        log(f"⚠️ Error loading encodings: {e}")
        known_encodings = {}
else:
    log("🆕 Generating new ArcFace embeddings...")
    known_encodings = {}

    for person_id in os.listdir(KNOWN_FACES_DIR):
        person_path = os.path.join(KNOWN_FACES_DIR, person_id)

        if os.path.isdir(person_path):
            log(f"🔹 Processing ID: {person_id}")
            encodings_list = []
            successful_count = 0

            for filename in os.listdir(person_path):
                if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(person_path, filename)
                    try:
                        image = cv2.imread(img_path)
                        if image is None:
                            log(f"⚠️ Could not read image: {img_path}")
                            continue
                        
                        # Ensure image is in RGB for better face detection
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        # Try to detect faces with ArcFace
                        faces = face_analyzer.get(rgb_image)
                        
                        if faces:
                            # Take the face with highest detection score if multiple are found
                            best_face = max(faces, key=lambda x: x.det_score)
                            embedding = best_face.embedding
                            encodings_list.append(embedding)
                            successful_count += 1
                        else:
                            # If no face detected, try different approach
                            # Resize image to help detection
                            resized = cv2.resize(image, (640, 640))
                            faces = face_analyzer.get(resized)
                            
                            if faces:
                                best_face = max(faces, key=lambda x: x.det_score)
                                embedding = best_face.embedding
                                encodings_list.append(embedding)
                                successful_count += 1
                                log(f"✅ Embedded (after resize): {filename}")
                            else:
                                log(f"⚠️ No face found in: {filename}")
                    except Exception as e:
                        log(f"⚠️ Error processing {img_path}: {e}")

            if encodings_list:
                known_encodings[person_id] = encodings_list
                log(f"✅ ID {person_id}: Successfully embedded {successful_count}/{len(os.listdir(person_path))} images")
            else:
                log(f"⚠️ ID {person_id}: Failed to embed any faces!")

    # Save the encodings to file
    try:
        with open(ENCODINGS_FILE, "wb") as file:
            pickle.dump(known_encodings, file)
        log("✅ Saved encodings to file")
    except Exception as e:
        log(f"⚠️ Error saving encodings: {e}")

# Load the unknown image
log(f"\n🔍 Loading image from: {UNKNOWN_IMAGE_PATH}")
image = cv2.imread(UNKNOWN_IMAGE_PATH)
if image is None:
    print(f"⚠️ Could not read image: {UNKNOWN_IMAGE_PATH}")
    exit(1)

# Make a copy for drawing
display_image = image.copy()

# Detect faces using YOLO
log("\n🔍 Detecting faces using YOLO...")
results = model(image)
detections = results[0].boxes.xyxy.cpu().numpy()

if len(detections) == 0:
    print("⚠️ No faces detected with YOLO!")
else:
    print(f"✅ {len(detections)} face(s) detected with YOLO!")

# CSV Attendance setup
if not os.path.exists(ATTENDANCE_CSV):
    with open(ATTENDANCE_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Roll Number", "Name", "Timestamp"])

# Process the image with ArcFace
log("\n🔍 Processing with ArcFace...")
# Convert to RGB for better face detection
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
all_faces = face_analyzer.get(rgb_image)

if all_faces:
    print(f"✅ ArcFace found {len(all_faces)} faces")
    
    # Create a mapping of face boxes to embeddings and detection scores
    face_data = {}
    for face in all_faces:
        box = face.bbox.astype(int)
        # Store embedding and score with box coordinates as key
        face_data[tuple(box)] = {
            'embedding': face.embedding,
            'score': face.det_score
        }
else:
    print("⚠️ ArcFace couldn't detect any faces!")
    # Try an alternative approach
    resized = cv2.resize(image, (640, 640))
    rgb_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    all_faces = face_analyzer.get(rgb_resized)
    
    if all_faces:
        print(f"✅ ArcFace found {len(all_faces)} faces after resizing")
        face_data = {}
        # Need to rescale bounding boxes back to original image size
        h_ratio = image.shape[0] / 640
        w_ratio = image.shape[1] / 640
        
        for face in all_faces:
            box = face.bbox.astype(int)
            # Rescale box coordinates
            scaled_box = (
                int(box[0] * w_ratio),
                int(box[1] * h_ratio),
                int(box[2] * w_ratio),
                int(box[3] * h_ratio)
            )
            face_data[scaled_box] = {
                'embedding': face.embedding,
                'score': face.det_score
            }
    else:
        face_data = {}
        print("⚠️ ArcFace couldn't detect any faces even after resizing!")

# Process each YOLO detection
recognized_faces = []
for idx, (x1, y1, x2, y2) in enumerate(detections):
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    face_crop = image[y1:y2, x1:x2]
    
    # Skip saving cropped faces as requested
    
    # Try to find the best embedding for this face
    best_match_id = None
    best_similarity = 0
    best_name = "Unknown"
    found_embedding = None
    
    # Approach 1: Try to directly analyze the cropped face
    rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    faces_in_crop = face_analyzer.get(rgb_crop)
    
    if faces_in_crop:
        # We got a valid embedding from the crop
        found_embedding = faces_in_crop[0].embedding
        log(f"✅ Got embedding directly from crop for face {idx + 1}")
    else:
        # Approach 2: Find the closest ArcFace box to this YOLO box
        best_iou = 0
        for arc_box, data in face_data.items():
            # Calculate IoU between YOLO box and ArcFace box
            ax1, ay1, ax2, ay2 = arc_box
            
            # Calculate intersection
            ix1 = max(x1, ax1)
            iy1 = max(y1, ay1)
            ix2 = min(x2, ax2)
            iy2 = min(y2, ay2)
            
            if ix2 > ix1 and iy2 > iy1:
                # Calculate areas
                intersection = (ix2 - ix1) * (iy2 - iy1)
                yolo_area = (x2 - x1) * (y2 - y1)
                arc_area = (ax2 - ax1) * (ay2 - ay1)
                union = yolo_area + arc_area - intersection
                iou = intersection / union
                
                if iou > best_iou and iou > 0.5:  # Threshold for IoU
                    best_iou = iou
                    found_embedding = data['embedding']
        
        if found_embedding is not None:
            print(f"✅ Matched YOLO box to ArcFace box for face {idx + 1} (IoU: {best_iou:.2f})")
        else:
            log(f"⚠️ No matching ArcFace embedding for face {idx + 1}")
            # Last resort: Try resizing the cropped face
            if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                resized_crop = cv2.resize(face_crop, (640, 640))
                rgb_resized_crop = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB)
                faces_resized = face_analyzer.get(rgb_resized_crop)
                
                if faces_resized:
                    found_embedding = faces_resized[0].embedding
                    log(f"✅ Got embedding from resized crop for face {idx + 1}")
    
    # If we found an embedding, try to match it with known faces
    if found_embedding is not None:
        unknown_embedding = found_embedding.reshape(1, -1)
        
        # Calculate similarities to all known face embeddings
        all_similarities = []
        
        for person_id, encodings_list in known_encodings.items():
            similarities = cosine_similarity(unknown_embedding, np.array(encodings_list))[0]
            max_similarity = np.max(similarities)
            all_similarities.append((person_id, max_similarity))
        
        # Sort by similarity (highest first)
        all_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get the best match
        if all_similarities and all_similarities[0][1] >= SIMILARITY_THRESHOLD:
            best_match_id = all_similarities[0][0]
            best_similarity = all_similarities[0][1]
            best_name = id_to_name.get(best_match_id, "Unknown")
            
            print(f"🎯 Face {idx + 1}: Matched to {best_match_id} ({best_name}) with similarity {best_similarity:.4f}")
            
            # Show top 3 matches for debugging
            for i, (pid, sim) in enumerate(all_similarities[:3]):
                log(f"  Match {i+1}: ID {pid} ({id_to_name.get(pid, 'Unknown')}) - {sim:.4f}")
        else:
            if all_similarities:
                log(f"ℹ️ Face {idx + 1}: Best match below threshold - {all_similarities[0][0]} with {all_similarities[0][1]:.4f}")
            else:
                log(f"ℹ️ Face {idx + 1}: No matches found")
    
    # Create label for drawing
    if best_match_id:
        label = f"{best_name}({best_match_id})"
        if DEBUG:
            label += f" {best_similarity:.2f}"
    else:
        label = "Unknown"
    
    # Draw bounding box and label
    # Darker colors for better visibility
    color = (0, 200, 0) if best_match_id else (0, 0, 200)  # Darker green for match, Darker red for unknown
    
    # Display rectangle with thicker lines for better visibility
    thickness = 12  # Increased thickness
    cv2.rectangle(display_image, (x1, y1), (x2, y2), color, thickness)
    
    # Better text display with background
    font_scale = 2.0  # Increased font size
    thickness = 3  # Increased text thickness
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    
    # Make background rectangle slightly wider than text
    padding = 10  # Increased padding
    cv2.rectangle(
        display_image, 
        (x1 - 2, y1 - text_size[1] - padding * 2), 
        (x1 + text_size[0] + padding * 2, y1), 
        color, 
        -1
    )
    
    # Text with white color for better contrast
    cv2.putText(
        display_image, 
        label, 
        (x1 + padding, y1 - padding), 
        font, 
        font_scale, 
        (255, 255, 255), 
        thickness
    )
    
    # No need to rename files since we're not saving cropped faces
    
    # Log attendance
    if best_match_id:
        recognized_faces.append((best_match_id, best_name))
        with open(ATTENDANCE_CSV, mode='a', newline='') as file:
            writer = csv.writer(file)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([best_match_id, best_name, now])

# Add a small delay to ensure file system operations complete
time.sleep(0.5)

# Make sure output path is absolute to avoid save issues
output_path = os.path.abspath("output_with_arcface.jpg")
cv2.imwrite(output_path, display_image)

# Verify that the file was saved successfully
if os.path.exists(output_path):
    print(f"✅ Successfully saved output image to: {output_path}")
    print(f"   Image size: {display_image.shape[1]}x{display_image.shape[0]} pixels")
    
    # Try to show the image if on a system with display capability
    try:
        # For systems with display capability
        cv2.imshow("Recognition Results", display_image)
        cv2.waitKey(1)  # Show for at least 1ms, keep window open
        print("✅ Displaying output image window (close window to continue)")
    except Exception as e:
        print(f"⚠️ Cannot display image window: {e}")
else:
    print(f"⚠️ Failed to save output image at {output_path}")
    # Try alternative save method
    try:
        alternative_path = "recognition_result.jpg"
        cv2.imwrite(alternative_path, display_image)
        if os.path.exists(alternative_path):
            print(f"✅ Saved output image using alternative method to: {alternative_path}")
        else:
            print("⚠️ Failed to save output image using alternative method")
    except Exception as e:
        print(f"⚠️ Error during alternative save: {e}")

# Summary of recognized faces
print("\n📊 Recognition Summary:")
if recognized_faces:
    for idx, (id_val, name) in enumerate(recognized_faces):
        print(f"  Face {idx+1}: {name}({id_val})")
else:
    print("  No faces recognized")

print(f"\n✅ Total faces detected: {len(detections)}")
print(f"✅ Total faces recognized: {len(recognized_faces)}")
