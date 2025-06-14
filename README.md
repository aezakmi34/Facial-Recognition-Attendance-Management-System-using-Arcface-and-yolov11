# Facial Recognition Attendance Management System

This project uses YOLO (for face detection) and ArcFace (via InsightFace) for face recognition and attendance marking.  
Repository: [GitHub - aezakmi34/Facial-Recognition-Attendance-Management-System-using-Arcface-and-yolov11](https://github.com/aezakmi34/Facial-Recognition-Attendance-Management-System-using-Arcface-and-yolov11.git)

---

## Prerequisites

- **Python 3.11.9** (other 3.11.x versions should work)
- **pip** (Python package manager)
- **Git** (for cloning and version control)
- **YOLOv8** (via the `ultralytics` Python package)
- **InsightFace** (ArcFace, installed via pre-built wheel)
- **OpenCV**
- **Virtual Environment** (recommended)
- **Git LFS** (for large model files like `best.pt`)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/aezakmi34/Facial-Recognition-Attendance-Management-System-using-Arcface-and-yolov11.git
cd Facial-Recognition-Attendance-Management-System-using-Arcface-and-yolov11
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate
```

### 3. Install dependencies

#### a. Install from requirements.txt

```bash
pip install -r requirements.txt
```

#### b. Install InsightFace (ArcFace) using the pre-built wheel

Replace the path with your actual wheel file location if different:

```bash
# Example (adjust path if needed)
& d:/face/.venv/Scripts/python.exe -m pip install "D:\Download\insightface-0.7.3-cp311-cp311-win_amd64.whl"
```

#### c. If you don't have requirements.txt, install manually:

```bash
pip install ultralytics opencv-python numpy scikit-learn
```

---

## File & Folder Setup

- Place your YOLO model file as `best.pt` in the project directory.
- Place your known faces in subfolders under `D:\face\non_face` (or change the path in the script):
  - Each subfolder should be named with a unique ID (e.g., roll number) and contain images of that person.
- Place the image you want to recognize at the path specified by `UNKNOWN_IMAGE_PATH` in the script.

---

## Usage

1. **Edit the configuration in `improved_face_recognition.py`**  
   Set the following variables at the top of the script as needed:
   - `KNOWN_FACES_DIR`
   - `ENCODINGS_FILE`
   - `UNKNOWN_IMAGE_PATH`
   - `YOLO_MODEL_PATH`
   - `ATTENDANCE_CSV`

2. **Run the script**

```bash
python improved_face_recognition.py
```

- The script will:
  - Detect faces in the unknown image using YOLO
  - Recognize faces using ArcFace (InsightFace)
  - Draw bounding boxes and labels on the image
  - Save the output image as `output_with_arcface.jpg`
  - Log recognized faces to `attendance.csv`

---

## Example Directory Structure
in Non_face folder folder name must be id and inside u have the images such as Alisa1.jpg , Alisa2.jpg like and same goes for all folders 
but folder name must be ID (015 , 344, like that)

```
project/
├── best.pt
├── improved_face_recognition.py
├── README.md
├── .gitignore
├── .gitattributes
└── D:/face/non_face/
    ├── 12345/        
    │   ├── img1.jpg
    │   └── img2.jpg
    └── 67890/
        └── img1.jpg
```

---

## Troubleshooting

- **InsightFace install issues:**  
  Use the provided wheel for Python 3.11.9 as shown above.
- **YOLO model issues:**  
  Make sure your `best.pt` is compatible with YOLOv8 (ultralytics).
- **OpenCV errors:**  
  Ensure your Python and OpenCV versions are compatible.
- **Large files on GitHub:**  
  Use Git LFS for model files like `best.pt`.

---
if you face any issue do hit me up on my 
instagram: abdullahumer442

For more, see the [GitHub repository](https://github.com/aezakmi34/Facial-Recognition-Attendance-Management-System-using-Arcface-and-yolov11.git). 
