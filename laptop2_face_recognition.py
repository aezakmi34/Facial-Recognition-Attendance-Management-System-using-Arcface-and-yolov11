import cv2
import os
import numpy as np
import pickle
import csv
from datetime import datetime
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ CONFIGURATION ------------------
KNOWN_FACES_DIR = r"D:\face\non_face"
ENCODINGS_FILE = "face_arc_encodings.pkl"
UNKNOWN_IMAGE_PATH = r"C:\Users\Hp\Downloads\20250529_151504.jpg"
ATTENDANCE_CSV = "attendance.csv"
SIMILARITY_THRESHOLD = 0.5
DEBUG = False  # Set to False to reduce logging
# ---------------------------------------------------

def log(message):
    """Custom logging function that only prints when DEBUG is True"""
    if DEBUG:
        print(message)

# Initialize ArcFace embedding model
face_analyzer = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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
                            log(f"✅ Embedded: {filename}")
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

# Convert to RGB for better face detection
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect faces using ArcFace
log("\n🔍 Detecting faces using ArcFace...")
faces = face_analyzer.get(rgb_image)

if not faces:
    # Try resizing if no faces detected
    log("⚠️ No faces detected, trying with resized image...")
    resized = cv2.resize(image, (640, 640))
    rgb_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    faces = face_analyzer.get(rgb_resized)
    
    if faces:
        # Rescale bounding boxes back to original image size
        h_ratio = image.shape[0] / 640
        w_ratio = image.shape[1] / 640
        for face in faces:
            face.bbox = face.bbox * np.array([w_ratio, h_ratio, w_ratio, h_ratio])

if len(faces) == 0:
    print("⚠️ No faces detected!")
    exit(1)
else:
    print(f"✅ {len(faces)} face(s) detected!")

# CSV Attendance setup
if not os.path.exists(ATTENDANCE_CSV):
    with open(ATTENDANCE_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Roll Number", "Name", "Timestamp"])

# Process each detected face
recognized_faces = []
for idx, face in enumerate(faces):
    # Get face bounding box
    x1, y1, x2, y2 = map(int, face.bbox)
    
    # Get face embedding
    unknown_embedding = face.embedding.reshape(1, -1)
    
    # Calculate similarities to all known face embeddings
    all_similarities = []
    
    for person_id, encodings_list in known_encodings.items():
        similarities = cosine_similarity(unknown_embedding, np.array(encodings_list))[0]
        max_similarity = np.max(similarities)
        all_similarities.append((person_id, max_similarity))
    
    # Sort by similarity (highest first)
    all_similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Get the best match
    best_match_id = None
    best_similarity = 0
    best_name = "Unknown"
    
    if all_similarities and all_similarities[0][1] >= SIMILARITY_THRESHOLD:
        best_match_id = all_similarities[0][0]
        best_similarity = all_similarities[0][1]
        best_name = id_to_name.get(best_match_id, "Unknown")
    
    # Create label for drawing
    if best_match_id:
        label = f"{best_name}({best_match_id})"
    else:
        label = "Unknown"
    
    # Draw bounding box and label
    color = (0, 255, 0) if best_match_id else (0, 0, 255)  # Green for match, Red for unknown
    
    # Draw rectangle
    cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
    
    # Better text display with background
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.rectangle(display_image, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
    cv2.putText(display_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Log attendance
    if best_match_id:
        recognized_faces.append((best_match_id, best_name))
        with open(ATTENDANCE_CSV, mode='a', newline='') as file:
            writer = csv.writer(file)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([best_match_id, best_name, now])

# Save final image
output_path = "output_with_arcface.jpg"
cv2.imwrite(output_path, display_image)

# Print only the recognition summary
print("\n📊 Recognition Summary:")
if recognized_faces:
    for idx, (id_val, name) in enumerate(recognized_faces):
        print(f"  Face {idx+1}: {name}({id_val})")
else:
    print("  No faces recognized")
