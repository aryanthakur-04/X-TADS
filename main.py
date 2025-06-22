# PHASE 1: Environment Setup 
#  Clone YOLOv5 and install dependencies
!git clone https://github.com/ultralytics/yolov5.git
%cd yolov5
!pip install -r requirements.txt
#  Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# PHASE 2: Dataset Setup 
#  Unzip datasets (OPIXray, HiXray, HUMS)
!unzip "/content/drive/MyDrive/OPIXray.zip" -d /content/yolov5/OPIXray/
!unzip "/content/drive/MyDrive/HiXray.zip" -d /content/yolov5/HiXray/
!unzip "/content/drive/MyDrive/HUMUS.zip" -d /content/yolov5/HUMUS/
#  Verify dataset structure
import os
print("OPIXray Train folder:", os.listdir("/content/yolov5/OPIXray/OPIXray_dataset/train"))
print("HiXray folder:", os.listdir("/content/yolov5/HiXray"))
print("HUMS folder:", os.listdir("/content/yolov5/HUMUS/HUMS-X-ray-Dataset-main/ThreatsRGB"))

# PHASE 3: Preprocessing - Organize & Convert Labels to YOLO Format
import os, shutil, random
from PIL import Image
from pathlib import Path
import xml.etree.ElementTree as ET

opi_image_folder = "/content/yolov5/OPIXray/OPIXray_dataset/train/train_image/"
opi_label_folder = "/content/yolov5/OPIXray/OPIXray_dataset/train/train_annotation/"
hixray_image_folder = "/content/yolov5/HiXray/train/train_image/"
hixray_label_folder = "/content/yolov5/HiXray/train/train_annotation/"
hums_image_folder = "/content/yolov5/HUMUS/HUMS-X-ray-Dataset-main/ThreatsRGB/"
hums_annotation_folder = "/content/yolov5/HUMUS/HUMS-X-ray-Dataset-main/Annotation/"

image_dest = "/content/yolov5/images/train/"
label_dest = "/content/yolov5/labels/train/"
val_image_dir = "/content/yolov5/images/val"
val_label_dir = "/content/yolov5/labels/val"

os.makedirs(image_dest, exist_ok=True)
os.makedirs(label_dest, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# Normalize class names
def normalize_class_name(name):
    mapping = {
        "Multi_tool_Knife": "Multi-tool_Knife",
        "Knife": "KnifeCustom",
        "Pistol": "Gun",
        "Revolver": "Gun",
        "Nonmetallic_Lighter": "Lighter",  # If needed
    }
    return mapping.get(name, name)

class_mapping = {
    "Utility_Knife": "0",
    "Scissor": "1",
    "Folding_Knife": "2",
    "Straight_Knife": "3",
    "Multi-tool_Knife": "4",
    "Gun": "5",
    "Explosive": "6",
    "Drug": "7",
    "Lighter": "8",
    "KnifeCustom": "9",
    "Mobile_Phone": "10",
    "Portable_Charger_1": "11",
    "Portable_Charger_2": "12",
    "Laptop": "13",
    "Tablet": "14",
    "Cosmetic": "15",
    "Water": "16"
}

def convert_bbox_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    x_center = ((xmin + xmax) / 2) / img_w
    y_center = ((ymin + ymax) / 2) / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    return x_center, y_center, width, height

# HUMS (XML to YOLO format)
for xml_file in os.listdir(hums_annotation_folder):
    if not xml_file.endswith(".xml"): continue
    tree = ET.parse(os.path.join(hums_annotation_folder, xml_file))
    root = tree.getroot()
    filename = root.find("filename").text
    image_path = os.path.join(hums_image_folder, filename)
    if not os.path.exists(image_path): continue
    shutil.copy(image_path, os.path.join(image_dest, filename))
    iw = int(root.find("size/width").text)
    ih = int(root.find("size/height").text)
    yolo_lines = []
    for obj in root.findall("object"):
        cname = normalize_class_name(obj.find("name").text)
        class_id = class_mapping.get(cname)
        if class_id is None:
            print(f"⚠️ Unknown class: {cname}")
            continue
        box = obj.find("bndbox")
        xc, yc, w, h = convert_bbox_to_yolo(float(box.find("xmin").text), float(box.find("ymin").text),
                                            float(box.find("xmax").text), float(box.find("ymax").text), iw, ih)
        yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
    with open(os.path.join(label_dest, filename.replace(".jpg", ".txt")), "w") as f:
        f.writelines(yolo_lines)

# TXT Label Conversion (OPIXray/HiXray)
def process_txt_labels(img_dir, lbl_dir):
    for file in os.listdir(lbl_dir):
        full_label_path = os.path.join(lbl_dir, file)
        img_path = os.path.join(img_dir, file.replace(".txt", ".jpg"))
        if not os.path.exists(img_path): continue
        shutil.copy(img_path, os.path.join(image_dest, os.path.basename(img_path)))
        with Image.open(img_path) as img:
            iw, ih = img.size
        new_lines = []
        with open(full_label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6: continue
                _, cname, xmin, ymin, xmax, ymax = parts
                cname = normalize_class_name(cname)
                class_id = class_mapping.get(cname)
                if class_id is None:
                    print(f"⚠️ Unknown class: {cname}")
                    continue
                xc, yc, w, h = convert_bbox_to_yolo(float(xmin), float(ymin), float(xmax), float(ymax), iw, ih)
                new_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        with open(os.path.join(label_dest, file), "w") as f:
            f.writelines(new_lines)

process_txt_labels(opi_image_folder, opi_label_folder)
process_txt_labels(hixray_image_folder, hixray_label_folder)

# Create validation split
image_files = list(Path(image_dest).glob("*.jpg"))
random.shuffle(image_files)
val_count = int(len(image_files) * 0.1)
for img_path in image_files[:val_count]:
    lbl_path = Path(label_dest) / (img_path.stem + ".txt")
    shutil.move(str(img_path), os.path.join(val_image_dir, img_path.name))
    shutil.move(str(lbl_path), os.path.join(val_label_dir, lbl_path.name))
  
#How many instances in each class
from collections import Counter
import glob

counter = Counter()
for label_file in glob.glob('/content/yolov5/labels/train/*.txt'):
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                counter[int(parts[0])] += 1

print("Class distribution in training labels:")
for class_id, count in sorted(counter.items()):
    print(f"Class {class_id}: {count} instances")

# PHASE 4: Create YAML Config
%%writefile /content/yolov5/data.yaml
train: /content/yolov5/images/train
val: /content/yolov5/images/val
nc: 17
names: ["Utility_Knife", "Scissor", "Folding_Knife", "Straight_Knife", "Multi-tool_Knife", "Gun", "Explosive", "Drug", "Lighter", "KnifeCustom", "Mobile_Phone", "Portable_Charger_1", "Portable_Charger_2", "Laptop", "Tablet", "Cosmetic", "Water"]

# Saving the file in Drive
!cp /content/yolov5/data.yaml /content/drive/MyDrive/expanded_data.yaml

# Calling the file in every new session
!cp /content/drive/MyDrive/expanded_data.yaml /content/yolov5/data.yaml

# PHASE 5: Train the Model
!python train.py \
  --img 640 \
  --batch 16 \
  --epochs 4 \
  --data /content/yolov5/data.yaml \
  --weights yolov5s.pt \
  --name extended_threat_model \
  --cache

# PHASE 6: Backup If You Stop Training or Before Timeout
!cp -r /content/yolov5/runs/train/extended_threat_model /content/drive/MyDrive/
#Saving Trained Model to Drive (if not done)
!cp /content/yolov5/runs/train/extended_threat_model/weights/best.pt /content/drive/MyDrive/extended_final_best.pt

# PHASE 7: Resume Training in Future Session
# Mount Drive again (if restarted)
from google.colab import drive
drive.mount('/content/drive')

# Restore training folder
!mkdir -p /content/yolov5/runs/train/ 
!cp -r /content/drive/MyDrive/extended_threat_model /content/yolov5/runs/train/

# Resume training from last checkpoint
!python train.py \
  --img 640 \
  --batch 16 \
  --epochs 5 \
  --data /content/yolov5/data.yaml \
  --weights /content/drive/MyDrive/extended_threat_model/weights/best.pt \
  --name extended_threat_model \
  --exist-ok \
  --cache

# Backup the Resumed Results
!cp -r /content/yolov5/runs/train/extended_threat_model /content/drive/MyDrive/extended_threat_model/

# Save the updated best model separately
!cp /content/yolov5/runs/train/extended_threat_model/weights/best.pt /content/drive/MyDrive/extended_threat_model_best.pt

# PHASE 8: Restoring Trained model from Drive

#To use the trained model:
!cp /content/drive/MyDrive/extended_threat_model_best.pt /content/yolov5/


# Inference on Custom Image (Manual Upload)
#Upload a Custom Image
from google.colab import files
uploaded = files.upload()  # Upload a .jpg or .png image

#Run YOLOv5 Detection on That Image (replace .jpg)
# Run inference
!python detect.py \
  --weights /content/drive/MyDrive/extended_final_best.pt \
  --img 640 \
  --conf 0.25 \
  --source R6175179_img1_SERVICE2_4_v1.PNG \
  --name custom_infer

#View the Output (replace mytest.jpg)
from IPython.display import Image, display
display(Image(filename='runs/detect/custom_infer2/R6175179_imag3_SERVICE2_4_v1.PNG', width=600))

#View Training Results & Run Inference
# Evaluate model performance
!python val.py \
  --weights /content/drive/MyDrive/extended_threat_model_best.pt \
  --data /content/yolov5/data.yaml \
  --img 640

#View Example Detection Output
from IPython.display import Image, display
import glob

result_images = glob.glob('runs/val/exp2/*.jpg')
for img_path in result_images[:3]:  # display first 3
    display(Image(filename=img_path, width=500))

# Images in the Train & Valid
!ls /content/yolov5/images/train | wc -l
!ls /content/yolov5/images/val | wc -l

# View training graphs
from IPython.display import Image
Image(filename='/content/drive/MyDrive/extended_threat_model/extended_threat_model/results.png', width=800) #need to change

!cat /content/drive/MyDrive/extended_threat_model/extended_threat_model/results.csv # need to change

# View the PR_Curve
from IPython.display import Image
Image(filename='/content/drive/MyDrive/extended_threat_model/extended_threat_model/PR_curve.png', width=800) # need to change

#GUI to visualize results (Gardio)
!pip install -q gradio fpdf pandas

import os, zipfile, shutil, uuid
import torch
import gradio as gr
import pandas as pd
from fpdf import FPDF
from PIL import Image

# Load default YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/content/drive/MyDrive/extended_final_best.pt', force_reload=True)
model.conf = 0.25  # initial threshold

# Override class names
class_names = {
    0:"Utility_Knife",
    1:"Scissor",
    2:"Folding_Knife",
    3:"Straight_Knife",
    4:"Multi-tool_Knife",
    5:"Gun",
    6:"Explosive,
    7:"Drug",
    8:"Lighter",
    9:"KnifeCustom",
    10:"Mobile_Phone",
    11:"Portable_Charger_1",
    12:"Portable_Charger_2",
    13:"Laptop",
    14:"Tablet",
    15:"Cosmetic",
    16:"Water"
}
model.names = class_names

# In-memory user store
auth_users = {"thakuraaryan04@gmail.com": {"password": "Aryan2004", "role": "admin"}}
user_sessions = {}

# AUTH FUNCTIONS
def login(email, password):
    if email in auth_users and auth_users[email]["password"] == password:
        session_id = str(uuid.uuid4())
        user_sessions[session_id] = {"email": email, "role": auth_users[email]["role"]}
        return gr.update(visible=True), gr.update(visible=False), "", session_id, auth_users[email]["role"]
    return gr.update(), gr.update(), " Invalid credentials", None, None

def logout():
    return gr.update(visible=False), gr.update(visible=True), "", None, None

def register(email, password):
    if email in auth_users:
        return " Already registered"
    auth_users[email] = {"password": password, "role": "user"}
    return "✅ Registration successful"

# PDF GENERATOR
def generate_combined_pdf(all_data, pdf_path, email):
    pdf = FPDF()
    pdf.set_font("Arial", size=12)
    for entry in all_data:
        original_path, detected_path, detections, filename = entry
        pdf.add_page()
        pdf.cell(200, 10, txt="Gardio Threat Detection Report", ln=True)
        pdf.cell(200, 10, txt=f"User: {email}", ln=True)
        pdf.cell(200, 10, txt=f"Image: {filename}", ln=True)
        pdf.ln(10)
        try: pdf.image(original_path, x=10, w=90)
        except: pdf.cell(200, 10, txt="Original image load failed.", ln=True)
        try: pdf.image(detected_path, x=110, w=90)
        except: pdf.cell(200, 10, txt="Detected image load failed.", ln=True)
        pdf.ln(10)
        if detections.empty:
            pdf.cell(200, 10, txt="No threats detected.", ln=True)
        else:
            for _, row in detections.iterrows():
                s = f"{row['name']} - {row['confidence']*100:.2f}% at [{int(row['xmin'])},{int(row['ymin'])},{int(row['xmax'])},{int(row['ymax'])}]"
                pdf.cell(200, 10, txt=s, ln=True)
    pdf.output(pdf_path)
    return pdf_path

# DETECTION 
def detect_all(images, email):
    base_dir = "/tmp/detect_output"
    os.makedirs(base_dir, exist_ok=True)
    all_detections, all_pdf_data, gallery_paths, logs_per_image = [], [], [], []

    for img in images:
        path = img.name if hasattr(img, 'name') else img
        filename = os.path.basename(path)
        original_path = os.path.join(base_dir, f"original_{filename}")
        detected_path = os.path.join(base_dir, f"detected_{filename}")
        shutil.copy(path, original_path)

        results = model(original_path)
        results.render()
        Image.fromarray(results.ims[0]).save(detected_path)

        detections = results.pandas().xyxy[0]
        detections.insert(0, "Image", filename)
        all_detections.append(detections)
        all_pdf_data.append((original_path, detected_path, detections, filename))
        gallery_paths.append(detected_path)

        # Logs
        log = f"**{filename}**\n"
        if detections.empty:
            log += "- No threats detected\n"
        else:
            for _, row in detections.iterrows():
                log += f"- `{row['name']}` at [{int(row['xmin'])}, {int(row['ymin'])}, {int(row['xmax'])}, {int(row['ymax'])}] — **{row['confidence']*100:.2f}%**\n"
        logs_per_image.append(log)

    csv_path = os.path.join(base_dir, f"{email.replace('@','_')}_combined.csv")
    if all_detections:
        pd.concat(all_detections, ignore_index=True).to_csv(csv_path, index=False)
    else:
        pd.DataFrame().to_csv(csv_path, index=False)

    pdf_path = os.path.join(base_dir, f"{email.replace('@','_')}_combined.pdf")
    generate_combined_pdf(all_pdf_data, pdf_path, email)

    return gallery_paths, [pdf_path], [csv_path], "\n".join(logs_per_image)

# ZIP CREATOR
def zip_all_reports():
    base_dir = "/tmp/detect_output"
    zip_path = os.path.join(base_dir, "gardio_reports.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for fname in os.listdir(base_dir):
            if fname.endswith("combined.pdf") or fname.endswith("combined.csv"):
                zipf.write(os.path.join(base_dir, fname), arcname=fname)
    return zip_path

# ADMIN PANEL FUNCTIONS
def load_new_model(model_path):
    global model
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        return "New model loaded successfully"
    except Exception as e:
        return f" Error loading model: {str(e)}"

def show_stats():
    output_dir = "/tmp/detect_output"
    if not os.path.exists(output_dir):
        return "No detections yet."
    csvs = [f for f in os.listdir(output_dir) if f.endswith(".csv")]
    pdfs = [f for f in os.listdir(output_dir) if f.endswith(".pdf")]
    return f"- Total CSV Reports: {len(csvs)}\n- Total PDF Reports: {len(pdfs)}\n- Users Active: {len(auth_users)}"

def reset_data():
    output_dir = "/tmp/detect_output"
    try:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        return "All detection files deleted"
    except Exception as e:
        return f" Error: {str(e)}"

def set_conf(thresh):
    model.conf = thresh
    return f"Confidence threshold set to {thresh:.2f}"

def toggle_admin(role): return gr.update(visible=(role == "admin"))

# GRADIO UI
with gr.Blocks() as app:
    gr.Markdown("## 🛡️ Gardio X-Ray Threat Detection System")

    with gr.Row(visible=True) as auth_row:
        email = gr.Textbox(label="Email")
        password = gr.Textbox(label="Password", type="password")
        login_btn = gr.Button("🔐 Login")
        signup_btn = gr.Button("📝 Register")
        auth_msg = gr.Markdown("")
        session_state = gr.State()
        user_role = gr.State()

    with gr.Column(visible=False) as main_row:
        logout_btn = gr.Button("🚪 Logout")
        gr.Markdown("### 📤 Upload and detect threats in X-ray images")
        img_input = gr.File(file_types=['image'], file_count='multiple', label="Upload Images")
        detect_btn = gr.Button("🚨 Detect Threats")
        img_display = gr.Gallery(label="Detection Results", show_label=True, height=400)
        report_output = gr.Files(label="📄 Download Combined PDF")
        csv_output = gr.Files(label="📊 Download Combined CSV")
        detection_logs = gr.Markdown("### 🧾 Detection Logs Per Image")
        zip_all_btn = gr.Button("📦 Download ZIP of All")
        zip_file = gr.File(label="ZIP File")

    with gr.Column(visible=False) as admin_panel:
        gr.Markdown("🛠️ **Admin Tools**")

        # Model upload
        gr.Markdown("#### 🔁 Upload or Re-train Model")
        model_file = gr.File(label="Upload YOLO Weights (.pt)", file_types=[".pt"])
        load_model_btn = gr.Button("📦 Load Uploaded Model")
        model_status = gr.Textbox(label="Model Load Status", interactive=False)
        load_model_btn.click(fn=load_new_model, inputs=[model_file], outputs=model_status)

        # System stats
        gr.Markdown("#### 📈 System Logs / Usage Stats")
        view_stats_btn = gr.Button("📊 Show Stats")
        stats_output = gr.Markdown()
        view_stats_btn.click(fn=show_stats, outputs=stats_output)

        # Reset detection data
        gr.Markdown("#### 🧹 Reset Detection Data")
        reset_btn = gr.Button("🗑️ Delete All Detection Files")
        reset_status = gr.Textbox(label="Reset Status", interactive=False)
        reset_btn.click(fn=reset_data, outputs=reset_status)

        # Threshold control
        gr.Markdown("#### 🎚️ Detection Confidence Threshold")
        conf_slider = gr.Slider(minimum=0.1, maximum=0.9, step=0.05, value=0.25, label="Set Threshold")
        conf_status = gr.Textbox(label="Threshold Status", interactive=False)
        conf_slider.change(fn=set_conf, inputs=conf_slider, outputs=conf_status)

    # Button wiring
    login_btn.click(fn=login, inputs=[email, password], outputs=[main_row, auth_row, auth_msg, session_state, user_role])
    logout_btn.click(fn=logout, outputs=[main_row, auth_row, auth_msg, session_state, user_role])
    signup_btn.click(fn=register, inputs=[email, password], outputs=[auth_msg])
    detect_btn.click(fn=detect_all, inputs=[img_input, email], outputs=[img_display, report_output, csv_output, detection_logs])
    zip_all_btn.click(fn=zip_all_reports, inputs=[], outputs=[zip_file])
    user_role.change(fn=toggle_admin, inputs=user_role, outputs=admin_panel)

app.launch()

