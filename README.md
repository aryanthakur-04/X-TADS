# X-TADS
**X-ray Threat Assessment & Detection System**

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo & Screenshots](#demo--screenshots)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Dataset & Model Details](#dataset--model-details)
- [Results & Performance](#results--performance)
- [Challenges & Solutions](#challenges--solutions)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [References](#references)

---

## Overview

**X-TADS** is an AI-driven X-ray Threat Assessment & Detection System designed to automate security screening at checkpoints such as airports. It leverages state-of-the-art deep learning (YOLOv5) to detect prohibited and hazardous items in X-ray images of baggage, providing real-time, reliable, and scalable threat detection. The system features a user-friendly web interface (Gardio App) for image upload, threat visualization, and comprehensive reporting.

---

## Features

- **Real-time Detection:** Automated detection and localization of prohibited items (e.g., knives, guns, explosives) in X-ray baggage images.
- **User-Friendly GUI:** Web-based Gradio interface for secure image upload, threat visualization, and result download.
- **Admin Controls:** Upload new detection models, set confidence thresholds, monitor usage stats, and reset inference data.
- **Export Reports:** Download detailed PDF and CSV reports containing detections, bounding box coordinates, confidence scores, and image thumbnails.
- **Authentication:** Secure login/signup system with role-based access (user/admin).
- **Batch Processing:** Supports multiple image uploads for bulk screening.
- **Fast & Accurate:** Achieves high mAP and fast inference thanks to optimized YOLOv5 model.

---

## Demo & Screenshots


### Gardio App Workflow

- **Login/Signup:** Register or log in to access the detection panel.
- **Upload Images:** Drag and drop X-ray `.jpg` or `.png` files.
- **Detect Threats:** Click "Detect Threats" for instant results—bounding boxes highlight detected items.
- **Download Reports:** Export results as PDF/CSV or a combined ZIP.

> *Sample PDF Report:*
> - User info, original and annotated images, detection details (e.g., “Straight_Knife — 87.94% at [611, 508, 690, 612]”)

---

## Installation

### Prerequisites

- Python 3.7+
- pip
- (Optional) CUDA-enabled GPU for training/inference

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/aryanthakur-04/X-TADS.git
   cd X-TADS
   ```

2. **Create and Activate Virtual Environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate              # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Start the Gardio Web App

```bash
python Code.py
```

- Access the GUI in your web browser  (as indicated in the terminal output).

### 2. Authenticate

- **Register** for a new account or **log in** with existing credentials.
- Admins use email-role mapping for elevated privileges.

### 3. Upload X-ray Images

- Upload single or multiple X-ray images (`.jpg` or `.png`).

### 4. Run Detection

- Click “Detect Threats” to process uploaded images.
- Results are displayed with bounding boxes and confidence scores.

### 5. Download Reports

- Export detection results as **PDF**, **CSV**, or a combined **ZIP**.

---

## Configuration

- **Model Upload:** Admins can upload new YOLO `.pt` weights.
- **Confidence Slider:** Adjust minimum detection confidence threshold.
- **System Stats:** View real-time stats (users, files processed, reports).
- **Reset Data:** Admins can clear temporary files and reset outputs.

---

## Dataset & Model Details

- **Datasets Used:** OPIXray, HiXray, HUMS (publicly available security X-ray datasets)
- **Classes Detected:** Utility_Knife, Scissor, Folding_Knife, Straight_Knife, Multi-tool_Knife, Gun, Explosive, Drug, Lighter, KnifeCustom, Mobile_Phone, Portable_Chargers, Laptop, Tablet, Cosmetic, Water
- **Model:** YOLOv5s (pre-trained, fine-tuned for 19 epochs, batch size 16, image size 640x640)
- **Training Split:** 90% train, 10% validation

---

## Results & Performance

- **mAP@0.5:** 0.81
- **Precision:** 0.87 | **Recall:** 0.78
- **Inference Speed:** ~4.7 ms/image

### Example Results

- **Knife Detected:** 88.4% confidence, correctly localized
- **No Threats:** Clean images returned no false positives

---

## Challenges & Solutions

- **Dataset Format Mismatch:** Scripts convert VOC/custom annotations to YOLO format
- **GPU Constraints:** Resumed training in multiple Google Colab sessions
- **Class Imbalance:** Stratified splits and augmentation for minority classes
- **GUI Bugs:** Debugged file paths and threading; separated admin/user logic

---

## Contributing

Contributions are welcome!

1. Fork this repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to your fork (`git push origin feature/your-feature`).
5. Open a Pull Request describing your changes.

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

---

## Acknowledgements

- Thanks to OpenAI’s ChatGPT for support with debugging and documentation.
- Gradio and Ultralytics for their open-source contributions.

---

## References

1. [Ultralytics YOLOv5 GitHub](https://github.com/ultralytics/yolov5)
2. [Gradio Documentation](https://www.gradio.app/docs/)
3. [OPIXray Dataset on Kaggle](https://www.kaggle.com/datasets/dhanushnarayananr/opixray-dataset)
4. [FPDF Python Library](https://pyfpdf.github.io/)
5. [Pandas Documentation](https://pandas.pydata.org/docs/)

---

*Begin your journey to safer automated threat detection with X-TADS!*
