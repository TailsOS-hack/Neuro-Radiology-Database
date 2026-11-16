import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from dataclasses import dataclass
from typing import Callable, Optional
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, mobilenet_v3_large
from torchvision import transforms
from src.gatekeeper_model import GatekeeperClassifier
from src.grounded_report import build_grounded_report, render_report_html, render_report_markdown
from src.grad_cam import GradCAM, get_target_layer, overlay_cam_on_image
from src import scan_view
from src.mri_volume import VolumeError
try:
    from src.experiment_pipeline import build_model as build_pipeline_model
    from src.experiment_pipeline import build_transforms as build_pipeline_transforms
except Exception:
    build_pipeline_model = None
    build_pipeline_transforms = None
import ollama
import threading
import queue
import time
import webbrowser
from pathlib import Path
import io
import base64
from datetime import datetime
from PIL import Image, ImageTk
from tkhtmlview import HTMLLabel
from tkcalendar import DateEntry
from weasyprint import HTML as WeasyHTML
from pypdf import PdfReader, PdfWriter

# WORKAROUND: Map 'gatekeeper_model' to 'src.gatekeeper_model' so torch.load finds it
import src.gatekeeper_model
sys.modules['gatekeeper_model'] = src.gatekeeper_model

APP_TITLE = "Radiology Report Generator"
BINARY_ROUTER_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "binary_router.pt")
GATEKEEPER_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "gatekeeper_classifier.pt")
TUMOR_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "brain_tumor_classifier.pt")
ALZHEIMERS_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "alzheimers_classifier.pt")

# Constants for Alzheimer's
ALZ_CLASSES_3 = ['MildDemented', 'ModerateDemented', 'VeryMildDemented']
ALZ_CLASSES_4 = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
ALZ_CLASSES = ALZ_CLASSES_4
TUMOR_CLASSES_3 = ["glioma", "meningioma", "pituitary"]
TUMOR_CLASSES_4 = ["glioma", "meningioma", "notumor", "pituitary"]
ALZ_IMG_SIZE = 224

PLAIN_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SCAN_FILETYPES = [
    ("Images and scans", ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.dcm", "*.dicom", "*.ima", "*.nii", "*.gz", "*.zip")),
    ("Image files", ("*.jpg", "*.jpeg", "*.png", "*.bmp")),
    ("DICOM / NIfTI / zipped DICOM", ("*.dcm", "*.dicom", "*.ima", "*.nii", "*.gz", "*.zip")),
    ("All files (DICOM often has no extension)", "*"),
]
# Shown whenever the 4-class stage classifier is applied to a slice of a raw scan.
# Evidence and reproduction: docs/DICOM_VOLUME_ANALYSIS.md.
RAW_SCAN_STAGE_CAUTION = (
    "Caution: this slice comes from a raw scan. The dementia-stage classifier was trained on skull-stripped, "
    "atlas-aligned slices; on raw T1 slices from 20 research participants it labelled 180 of 180 slices as "
    "demented, including every cognitively normal control. Treat this stage label as unreliable. "
    "Use 'Analyze Volume' for whole-scan research scoring."
)

def build_tumor_model(arch: str, num_classes: int):
    if arch == "efficientnet_b3":
        model = efficientnet_b3(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.3), nn.Linear(in_features, num_classes)
        )
        return model
    raise ValueError(f"Unsupported architecture: {arch}")

def build_alzheimers_model(arch: str, num_classes: int):
    if arch == "mobilenet_v3_large":
        model = mobilenet_v3_large(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unsupported architecture: {arch}")

def infer_linear_outputs(model, fallback):
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Linear):
            return module.out_features
    return fallback

def get_tumor_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose(
        [
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

def get_alzheimers_transform():
    # Matches the MobileNetV3 training (ImageNet stats)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def get_gatekeeper_transform():
    # Gatekeeper (ResNet50) uses 224x224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

@dataclass
class ClassificationResult:
    label: str
    confidence: float
    model_name: str
    source_image: Image.Image
    cam_model: Optional[nn.Module] = None
    cam_transform: Optional[Callable] = None
    cam_class_idx: Optional[int] = None


class App(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=15)
        self.pack(fill="both", expand=True)
        self.master = master

        # Device selection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            try:
                import torch_directml as dml
                self.device = dml.device()
            except Exception:
                self.device = torch.device("cpu")

        # Models
        self.gatekeeper_model = None
        self.gatekeeper_mode = "legacy_3way"
        self.gatekeeper_classes = ["Normal", "tumor", "dementia"]
        self.tumor_model = None
        self.tumor_classes = []
        self.alz_model = None
        self.alz_classes = ALZ_CLASSES_4

        self.image_path = None
        self.tk_image = None
        self._original_pil_image = None
        self.heatmap_pil_image = None
        self.showing_heatmap = False
        self.last_model_name = None

        # Transforms
        self.tumor_tfms = get_tumor_transform()
        self.alz_tfms = get_alzheimers_transform()
        self.gate_tfms = get_gatekeeper_transform()

        # Scan volumes (DICOM series / NIfTI). None for ordinary image files.
        self.volume = None
        self.volume_window = None
        self.volume_series = []
        self.volume_result = None
        self.volume_analyzer = None
        self.scan_slice_image = None  # displayed slice; the 2D cascade classifies this
        self.slice_index = 0
        self.scan_queue = queue.Queue()
        self._header_fields = []  # (entry, value taken from a scan header, default) - see _clear_header_fields
        self._header_dob = None

        # Reporting
        self.report_queue = queue.Queue()
        self.last_report_html = None
        self.last_report_markdown = None

        self._build_ui()
        
        # Auto-load models
        self.load_models()

    def _build_ui(self):
        # Top bar for model status
        top_bar = ttk.Frame(self)
        top_bar.pack(fill="x", pady=(0, 10))
        self.model_status_label = ttk.Label(top_bar, text="Models: Loading...", anchor="w")
        self.model_status_label.pack(side="left", fill="x", expand=True)
        
        # Manual reload button
        ttk.Button(top_bar, text="Reload Models", command=self.load_models).pack(side="right")

        # Main layout with a resizable pane
        main_pane = ttk.PanedWindow(self, orient="horizontal")
        main_pane.pack(fill="both", expand=True)

        # --- Left Column: Image and Classification ---
        left_frame = ttk.Frame(main_pane, padding=5)
        main_pane.add(left_frame, weight=1)

        # 1. Input Notebook (Top)
        input_notebook = ttk.Notebook(left_frame)
        input_notebook.pack(side="top", fill="x", pady=(0, 10))

        # Tab 1: Patient Info
        patient_frame = ttk.Frame(input_notebook, padding=10)
        input_notebook.add(patient_frame, text="Patient Info")
        
        info_grid = ttk.Frame(patient_frame)
        info_grid.pack(fill="x", expand=True)
        info_grid.columnconfigure(1, weight=1)

        ttk.Label(info_grid, text="Name:").grid(row=0, column=0, sticky="w", pady=2)
        self.patient_name_entry = ttk.Entry(info_grid)
        self.patient_name_entry.grid(row=0, column=1, sticky="ew", padx=5)

        ttk.Label(info_grid, text="Patient ID:").grid(row=1, column=0, sticky="w", pady=2)
        self.patient_id_entry = ttk.Entry(info_grid)
        self.patient_id_entry.grid(row=1, column=1, sticky="ew", padx=5)

        ttk.Label(info_grid, text="DOB:").grid(row=2, column=0, sticky="w", pady=2)
        self.dob_entry = DateEntry(info_grid, date_pattern='yyyy-mm-dd', width=12, background='darkblue', foreground='white', borderwidth=2,
                                   selectmode='day', year=datetime.now().year - 40, month=1, day=1)
        self.dob_entry.grid(row=2, column=1, sticky="w", padx=5)

        # Tab 2: Exam Details
        exam_frame = ttk.Frame(input_notebook, padding=10)
        input_notebook.add(exam_frame, text="Exam Details")
        
        # Control Buttons for Exam Details
        self.exam_btn_frame = ttk.Frame(exam_frame)
        self.exam_btn_frame.pack(fill="x", pady=(0, 10))
        
        self.manual_btn = ttk.Button(self.exam_btn_frame, text="Manual Entry", command=self.enable_manual_input, state="disabled")
        self.manual_btn.pack(side="left", expand=True, fill="x", padx=(0, 5))
        
        self.ai_fill_btn = ttk.Button(self.exam_btn_frame, text="AI Auto-Detect", command=self.auto_detect_exam_details, state="disabled")
        self.ai_fill_btn.pack(side="right", expand=True, fill="x", padx=(5, 0))

        exam_grid = ttk.Frame(exam_frame)
        exam_grid.pack(fill="x", expand=True)
        exam_grid.columnconfigure(1, weight=1)

        ttk.Label(exam_grid, text="Reason:").grid(row=0, column=0, sticky="w", pady=2)
        self.reason_entry = ttk.Entry(exam_grid, state="disabled")
        self.reason_entry.grid(row=0, column=1, sticky="ew", padx=5)
        
        ttk.Label(exam_grid, text="History:").grid(row=1, column=0, sticky="w", pady=2)
        self.history_entry = ttk.Entry(exam_grid, state="disabled")
        self.history_entry.grid(row=1, column=1, sticky="ew", padx=5)

        ttk.Label(exam_grid, text="Comparison:").grid(row=2, column=0, sticky="w", pady=2)
        self.comparison_entry = ttk.Entry(exam_grid, state="disabled")
        self.comparison_entry.grid(row=2, column=1, sticky="ew", padx=5)
        
        ttk.Label(exam_grid, text="Technique:").grid(row=3, column=0, sticky="w", pady=2)
        self.technique_entry = ttk.Entry(exam_grid, state="disabled")
        self.technique_entry.grid(row=3, column=1, sticky="ew", padx=5)
        
        ttk.Label(exam_grid, text="Contrast:").grid(row=4, column=0, sticky="w", pady=2)
        self.contrast_entry = ttk.Entry(exam_grid, state="disabled")
        self.contrast_entry.grid(row=4, column=1, sticky="ew", padx=5)

        # 5. Classification Result (Bottom)
        result_frame = ttk.LabelFrame(left_frame, text="Classification Details", padding=10)
        result_frame.pack(side="bottom", fill="x")
        self.pred_label = ttk.Label(result_frame, text="Prediction: -", font=("Segoe UI", 11, "bold"))
        self.pred_label.pack(anchor="w")
        self.confidence_label = ttk.Label(result_frame, text="Confidence: -")
        self.confidence_label.pack(anchor="w")
        self.model_used_label = ttk.Label(result_frame, text="Model Used: -", foreground="#555", justify="left")
        self.model_used_label.pack(anchor="w", fill="x")
        self._wrap_to_width(self.model_used_label)
        # Where the last 3D map was saved: one line that stays on screen after the dialogs and can be selected
        # and copied. It is shown once there is something to show (see _show_saved_folder).
        self.saved_var = tk.StringVar(value="")
        self.saved_row = ttk.Frame(result_frame)
        ttk.Label(self.saved_row, text="3D map saved in:", foreground="#555").pack(side="left")
        self.saved_entry = ttk.Entry(self.saved_row, textvariable=self.saved_var, state="readonly")
        self.saved_entry.pack(side="left", fill="x", expand=True, padx=(4, 0))

        # 4. Progress Bar (Bottom, above results)
        self.progress_bar = ttk.Progressbar(left_frame, mode='indeterminate')
        self.progress_bar.pack(side="bottom", fill="x", pady=5)
        self.progress_bar.pack_forget() # Hidden initially

        # 3. Analysis Controls (Bottom, above progress bar)
        analysis_frame = ttk.LabelFrame(left_frame, text="Analysis", padding=10)
        analysis_frame.pack(side="bottom", fill="x", pady=10)
        
        self.generate_btn = ttk.Button(analysis_frame, text="Analyze & Generate Report", command=self.on_analyze_and_generate, state="disabled")
        self.generate_btn.pack(fill="x", expand=True)

        # Whole-scan analysis (enabled once a 3D DICOM/NIfTI volume is loaded)
        self.volume_btn = ttk.Button(
            analysis_frame, text="Analyze Volume (AD/CN research model)", command=self.on_analyze_volume, state="disabled"
        )
        self.volume_btn.pack(fill="x", expand=True, pady=(6, 0))

        # 2. Image/Scan Frame (Middle - Expands)
        img_frame = ttk.LabelFrame(left_frame, text="Scan", padding=10)
        img_frame.pack(side="top", fill="both", expand=True)
        
        # Scan Buttons (Upload) - Pack to BOTTOM first so they stay visible
        scan_row = ttk.Frame(img_frame)
        scan_row.pack(side="bottom", fill="x", pady=(5, 0))
        self.choose_img_btn = ttk.Button(scan_row, text="Scan", command=self.on_choose_image)
        self.choose_img_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.choose_folder_btn = ttk.Button(scan_row, text="DICOM Folder...", command=self.on_choose_dicom_folder)
        self.choose_folder_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.series_btn = ttk.Button(scan_row, text="Series...", command=self.on_choose_series, state="disabled")
        self.series_btn.pack(side="right")

        # Slice navigation for volumes (hidden until a volume is loaded)
        self.slice_frame = ttk.Frame(img_frame)
        self.slice_scale = ttk.Scale(self.slice_frame, from_=0, to=1, orient="horizontal", command=self._on_slice_scale)
        self.slice_scale.pack(side="top", fill="x")
        self.scan_info_label = ttk.Label(
            self.slice_frame, text="", foreground="#555", font=("Segoe UI", 8), wraplength=320, justify="left"
        )
        self.scan_info_label.pack(side="top", fill="x")
        self._wrap_to_width(self.scan_info_label)

        # Grad-CAM toggle + caption (disabled until a heatmap is generated)
        self.heatmap_caption_label = ttk.Label(
            img_frame, text="", foreground="#555", font=("Segoe UI", 8), wraplength=320, justify="left"
        )
        self.heatmap_caption_label.pack(side="bottom", fill="x", pady=(2, 0))
        self._wrap_to_width(self.heatmap_caption_label)

        heatmap_row = ttk.Frame(img_frame)
        heatmap_row.pack(side="bottom", fill="x", pady=(5, 0))
        self.heatmap_toggle_btn = ttk.Button(
            heatmap_row, text="Show Heatmap", command=self._toggle_heatmap_view, state="disabled"
        )
        self.heatmap_toggle_btn.pack(side="left", fill="x", expand=True)
        self.save_volume_btn = ttk.Button(
            heatmap_row, text="Save 3D Map...", command=self.on_save_volume_result, state="disabled"
        )
        self.save_volume_btn.pack(side="right", padx=(5, 0))

        # Canvas - Pack to TOP and expand to fill remaining space
        self.canvas = tk.Canvas(img_frame, width=350, height=350, bg="#f0f0f0", relief="sunken", borderwidth=1)
        self.canvas.pack(side="top", fill="both", expand=True)
        self._displayed_image = None
        self._resize_job = None
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # --- Right Column: Generated Report ---
        right_frame = ttk.Frame(main_pane, padding=5)
        main_pane.add(right_frame, weight=2)
        
        report_frame = ttk.LabelFrame(right_frame, text="Generated Radiology Report", padding=10)
        report_frame.pack(fill="both", expand=True)
        
        self.report_html = HTMLLabel(report_frame, html="<p>Report will be generated here.</p>")
        self.report_html.pack(fill="both", expand=True)
        
        self.save_pdf_btn = ttk.Button(right_frame, text="Save as PDF", command=self.on_save_pdf, state="disabled")
        self.save_pdf_btn.pack(pady=5)
    def load_models(self):
        status_texts = []
        
        # 1. Load Binary Router if present, otherwise fall back to legacy 3-way gatekeeper.
        self.gatekeeper_model = None
        self.gatekeeper_mode = "legacy_3way"
        self.gatekeeper_classes = ["Normal", "tumor", "dementia"]

        if os.path.isfile(BINARY_ROUTER_MODEL_PATH) and build_pipeline_model:
            try:
                checkpoint = torch.load(BINARY_ROUTER_MODEL_PATH, map_location=self.device, weights_only=False)
                if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
                    raise ValueError("binary_router.pt is missing pipeline checkpoint metadata")

                self.gatekeeper_classes = checkpoint.get("class_names", ["tumor", "dementia"])
                model = build_pipeline_model(
                    checkpoint.get("arch", "resnet50"),
                    len(self.gatekeeper_classes),
                    pretrained=False,
                )
                model.load_state_dict(checkpoint["model_state"])
                model.eval().to(self.device)
                self.gatekeeper_model = model
                self.gatekeeper_mode = "binary"

                if build_pipeline_transforms:
                    self.gate_tfms = build_pipeline_transforms(
                        train=False,
                        image_size=int(checkpoint.get("image_size", 224)),
                    )
                status_texts.append("Binary Router: Ready")
            except Exception as e:
                print(f"Error loading binary router model: {e}")
                status_texts.append("Binary Router: Error")

        elif os.path.isfile(GATEKEEPER_MODEL_PATH):
            try:
                # Load the file first
                loaded_obj = torch.load(GATEKEEPER_MODEL_PATH, map_location=self.device, weights_only=False)
                
                if isinstance(loaded_obj, nn.Module):
                    # It's a full model
                    self.gatekeeper_model = loaded_obj
                else:
                    # It's a state dict
                    model = GatekeeperClassifier(num_classes=3, freeze_base=False)
                    model.load_state_dict(loaded_obj)
                    self.gatekeeper_model = model
                
                self.gatekeeper_model.eval().to(self.device)
                status_texts.append("Gatekeeper: Ready")
            except Exception as e:
                print(f"Error loading gatekeeper model: {e}")
                status_texts.append("Gatekeeper: Error")
        else:
            status_texts.append("Binary Router/Gatekeeper: Not Found")

        # 2. Load Tumor Model
        if os.path.isfile(TUMOR_MODEL_PATH):
            try:
                checkpoint = torch.load(TUMOR_MODEL_PATH, map_location=self.device, weights_only=False)
                tumor_tfms = get_tumor_transform()
                
                # Check if it's a dictionary (checkpoint) or a full model
                if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                    arch = checkpoint.get("arch", "efficientnet_b3")
                    self.tumor_classes = checkpoint.get("class_names", ["glioma", "meningioma", "notumor", "pituitary"])
                    model = build_tumor_model(arch, len(self.tumor_classes))
                    model.load_state_dict(checkpoint["model_state"])
                    if build_pipeline_transforms is None:
                        raise RuntimeError("Pipeline preprocessing is unavailable")
                    tumor_tfms = build_pipeline_transforms(
                        train=False,
                        image_size=int(checkpoint.get("image_size", 224)),
                    )
                else:
                    # Assume it's a full model object or state dict
                    # If it's a state dict, we need to know the arch. If it's a full model, just use it.
                    if isinstance(checkpoint, nn.Module):
                        model = checkpoint
                        output_count = infer_linear_outputs(model, 4)
                        self.tumor_classes = TUMOR_CLASSES_3 if output_count == 3 else TUMOR_CLASSES_4
                    else:
                        # Fallback for state dict only
                        model = build_tumor_model("efficientnet_b3", 4)
                        model.load_state_dict(checkpoint)
                        self.tumor_classes = TUMOR_CLASSES_4

                model.eval().to(self.device)
                self.tumor_model = model
                self.tumor_tfms = tumor_tfms
                status_texts.append("Tumor: Ready")
            except Exception as e:
                print(f"Error loading tumor model: {e}")
                status_texts.append("Tumor: Error")
        else:
            status_texts.append("Tumor: Not Found")

        # 3. Load Alzheimer's Model
        if os.path.isfile(ALZHEIMERS_MODEL_PATH):
            try:
                checkpoint = torch.load(ALZHEIMERS_MODEL_PATH, map_location=self.device, weights_only=False)
                alz_tfms = get_alzheimers_transform()
                if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                    arch = checkpoint.get("arch", "mobilenet_v3_large")
                    self.alz_classes = checkpoint.get("class_names", ALZ_CLASSES_4)
                    model = build_alzheimers_model(arch, len(self.alz_classes))
                    model.load_state_dict(checkpoint["model_state"])
                    if build_pipeline_transforms is None:
                        raise RuntimeError("Pipeline preprocessing is unavailable")
                    alz_tfms = build_pipeline_transforms(
                        train=False,
                        image_size=int(checkpoint.get("image_size", 224)),
                    )
                else:
                    # Older training scripts saved the full model object.
                    model = checkpoint
                    output_count = infer_linear_outputs(model, 4)
                    self.alz_classes = ALZ_CLASSES_3 if output_count == 3 else ALZ_CLASSES_4
                model.eval().to(self.device)
                self.alz_model = model
                self.alz_tfms = alz_tfms
                status_texts.append("Alzheimer's: Ready")
            except Exception as e:
                print(f"Error loading alzheimer model: {e}")
                status_texts.append("Alzheimer's: Error")
        else:
            status_texts.append("Alzheimer's: Not Found")

        self.model_status_label.configure(text=" | ".join(status_texts))
        
        if self.tumor_model or self.alz_model:
            if self.image_path:
                self.generate_btn.configure(state="normal")

    def on_choose_image(self):
        path = filedialog.askopenfilename(
            title="Select MRI image, DICOM file, NIfTI volume or zipped DICOM study",
            filetypes=SCAN_FILETYPES,
            initialdir=os.path.abspath(os.path.join("data", "evaluation", "images")),
        )
        if path:
            self._open_path(path)

    def on_choose_dicom_folder(self):
        path = filedialog.askdirectory(title="Select a folder containing a DICOM study or series")
        if path:
            self._open_path(path)

    def _open_path(self, path):
        """Plain image files open directly; anything else is read as a scan off the main thread."""
        if os.path.isfile(path) and os.path.splitext(path)[1].lower() in PLAIN_IMAGE_EXTENSIONS:
            self._reset_scan_state()
            self.image_path = path
            self._clear_heatmap()
            self._display_image(path)

            # Enable Exam Detail Controls
            self.manual_btn.configure(state="normal")
            self.ai_fill_btn.configure(state="normal")
        else:
            self._start_scan_load(path)

    def _clear_heatmap(self):
        self.heatmap_pil_image = None
        self.showing_heatmap = False
        self.heatmap_toggle_btn.configure(text="Show Heatmap", state="disabled")
        self.heatmap_caption_label.configure(text="")

    def _clear_header_fields(self):
        """Form values copied from one scan's header must never follow the user to another image."""
        for entry, value, default in self._header_fields:
            if entry.get() == value:  # untouched since it was filled in; user edits are kept
                state = str(entry.cget("state"))
                entry.configure(state="normal")
                entry.delete(0, tk.END)
                entry.insert(0, default)
                entry.configure(state=state)
        self._header_fields = []
        if self._header_dob is not None:
            try:
                if self.dob_entry.get_date() == self._header_dob:
                    self.dob_entry.set_date(self._header_dob.replace(year=datetime.now().year - 40, month=1, day=1))
            except Exception:
                pass
            self._header_dob = None

    def _fill_from_header(self, entry, value, default=""):
        entry.delete(0, tk.END)
        entry.insert(0, value)
        self._header_fields.append((entry, value, default))

    def _show_saved_folder(self, folder: str = "") -> None:
        """Show where the 3D map was saved, or take the line away when there is nothing saved for this scan."""

        self.saved_var.set(folder)
        if folder:
            self.saved_row.pack(anchor="w", fill="x", pady=(2, 0))
            self.saved_entry.xview_moveto(1.0)  # a long path shows its end, which names the folder
        else:
            self.saved_row.pack_forget()

    def _reset_scan_state(self):
        self._clear_header_fields()
        self._show_saved_folder()
        self.volume = None
        self.volume_window = None
        self.volume_series = []
        self.volume_result = None
        self.scan_slice_image = None
        self.slice_index = 0
        self.slice_frame.pack_forget()
        self.scan_info_label.configure(text="")
        self.volume_btn.configure(state="disabled")
        self.save_volume_btn.configure(state="disabled")
        self.series_btn.configure(state="disabled")

    def _set_busy(self, busy, message=None):
        """Long scan work runs in a thread; block conflicting actions meanwhile."""
        state = "disabled" if busy else "normal"
        for button in (self.choose_img_btn, self.choose_folder_btn):
            button.configure(state=state)
        if busy:
            for button in (self.generate_btn, self.volume_btn, self.save_volume_btn, self.series_btn):
                button.configure(state="disabled")
            if message:
                self.pred_label.configure(text=message)
            self.progress_bar.pack(fill="x", pady=5)
            self.progress_bar.start()
        else:
            self.progress_bar.stop()
            self.progress_bar.pack_forget()
            if self.image_path and (self.tumor_model or self.alz_model) and str(self.reason_entry.cget("state")) == "normal":
                self.generate_btn.configure(state="normal")
            if self.volume is not None:
                self.volume_btn.configure(state="normal" if scan_view.is_volume(self.volume) else "disabled")
                self.series_btn.configure(state="normal" if len(self.volume_series) > 1 else "disabled")
                self.save_volume_btn.configure(
                    state="normal" if self.volume_result is not None and self.volume_result.cam_grid is not None else "disabled"
                )

    def _start_scan_load(self, path, series_key=None):
        self._set_busy(True, "Reading scan...")
        threading.Thread(target=self._load_scan_worker, args=(path, series_key), daemon=True).start()
        self.master.after(100, self._poll_scan_queue)

    def _load_scan_worker(self, path, series_key):
        try:
            series = []
            if path.lower().endswith((".nii", ".nii.gz")):
                from src.mri_volume import load_nifti
                volume = load_nifti(path)
            else:
                from src import dicom_io
                series = dicom_io.discover_series(path)
                if not series:
                    raise dicom_io.NotDicomError("No DICOM image series was found in the selection.")
                volume = dicom_io.load_series(dicom_io.select_series(series, key=series_key))
            self.scan_queue.put(("loaded", path, volume, series))
        except Exception as e:
            self.scan_queue.put(("error", "Scan Error", f"Could not read the scan:\n{e}"))

    def _poll_scan_queue(self):
        try:
            message = self.scan_queue.get_nowait()
        except queue.Empty:
            self.master.after(100, self._poll_scan_queue)
            return
        kind = message[0]
        if kind == "loaded":
            self._on_scan_loaded(*message[1:])
        elif kind == "analysis":
            self._on_volume_result(message[1])
        elif kind == "saved":
            self._set_busy(False)
            self.pred_label.configure(text=f"Prediction: {self.volume_result.predicted_label} (research AD/CN score)")
            folder = os.path.abspath(message[1])
            self._show_saved_folder(folder)
            text = f"3D Grad-CAM written to:\n{folder}\n\n" + "\n".join(sorted(message[2]))
            if "viewer.html" not in message[2]:
                messagebox.showinfo(
                    "Saved",
                    text + "\n\nThe interactive viewer (viewer.html) could not be written. The files listed are complete.",
                )
            elif messagebox.askyesno(
                "Saved",
                text + "\n\nOpen the interactive 3D viewer (viewer.html) in your browser?\n"
                "If no browser window appears, open that file from the folder above.",
            ):
                self._open_viewer(Path(folder, "viewer.html"))
        else:
            self._set_busy(False)
            self.pred_label.configure(text="Prediction: -")
            messagebox.showerror(message[1], message[2])

    @staticmethod
    def _wrap_to_width(label) -> None:
        """Wrap a label's text at the width the label is actually given, so no line is cut off at the panel edge."""

        label.bind("<Configure>", lambda event: label.configure(wraplength=max(80, event.width - 4)))

    #: Python's own launchers can take about five seconds to report that nothing started. A launcher that ran
    #: longer than this did show a browser (it returned when the browser was closed), whatever it reports then.
    VIEWER_LAUNCH_GRACE_SECONDS = 8.0
    #: How long the app waits to hear from the launcher. After that it is taken to be holding a browser open,
    #: and whatever it reports later is not read.
    VIEWER_LAUNCH_WAIT_SECONDS = 15.0

    def _open_viewer(self, viewer: Path) -> None:
        """Hand ``viewer.html`` to the default browser; say where the file is when no browser could be started.

        Some launchers do not return until the browser is closed, so the call runs on its own thread and the
        window stays responsive meanwhile. A launcher can also report success without opening anything, which
        cannot be detected: the question before this step and the "3D map saved in" line name the folder for that case.
        """

        target = viewer.resolve()
        outcome: queue.Queue = queue.Queue()

        def worker():
            began = time.monotonic()
            try:
                started = bool(webbrowser.open(target.as_uri()))
            except Exception:  # webbrowser raises its own Error, and launchers fail in their own ways
                started = False
            outcome.put(started or time.monotonic() - began > self.VIEWER_LAUNCH_GRACE_SECONDS)

        threading.Thread(target=worker, daemon=True).start()
        deadline = time.monotonic() + self.VIEWER_LAUNCH_WAIT_SECONDS

        def check():
            try:
                started = outcome.get_nowait()
            except queue.Empty:
                if time.monotonic() < deadline:
                    self.master.after(200, check)
                return
            if not started:
                messagebox.showinfo("Viewer", f"No browser could be started. Open this file in a browser:\n{target}")

        self.master.after(200, check)

    def _on_scan_loaded(self, path, volume, series):
        self._reset_scan_state()
        self._clear_heatmap()
        self.image_path = path
        self.volume = volume
        self.volume_series = series
        self.volume_window = scan_view.display_window(volume)
        self.slice_index = volume.shape[2] // 2
        self._configure_slice_slider(volume.shape[2])
        self._show_scan_slice()
        self.pred_label.configure(text="Prediction: -")
        self.confidence_label.configure(text="Confidence: -")
        self.model_used_label.configure(text="Model Used: -")

        # The header already says who was scanned and how; use it rather than retyping.
        identity = volume.identity
        if identity is not None:
            for entry, value in ((self.patient_name_entry, scan_view.format_person_name(identity.name)),
                                 (self.patient_id_entry, identity.patient_id)):
                if value:
                    self._fill_from_header(entry, value)
            birth_date = scan_view.parse_dicom_date(identity.birth_date)
            if birth_date is not None:
                try:
                    self.dob_entry.set_date(birth_date)
                    self._header_dob = birth_date
                except Exception:
                    pass
        self.manual_btn.configure(state="normal")
        self.ai_fill_btn.configure(state="normal")
        self.enable_manual_input()
        if volume.source.get("format", "").startswith("dicom"):
            self._fill_from_header(self.technique_entry, scan_view.technique_from_header(volume), "Standard MRI Brain Protocol")
            self._fill_from_header(self.contrast_entry, scan_view.contrast_from_header(volume), "None")
        self._set_busy(False)

        guess = volume.acquisition.get("contrast_guess") or {}
        if guess.get("label") not in (None, "T1w"):
            messagebox.showwarning(
                "Check the series",
                f"This series looks like {guess.get('label')} rather than T1-weighted MRI "
                f"({'; '.join(guess.get('reasons', []))}).\n\nThe classifiers were trained on T1 images only."
                + ("\n\nUse 'Series...' to pick another series from this study." if len(series) > 1 else ""),
            )

    def _configure_slice_slider(self, count):
        if count > 1:
            self.slice_scale.configure(from_=0, to=count - 1)
            self.slice_scale.set(self.slice_index)
        # Always shown for scans: the caption underneath carries the scan description.
        self.slice_frame.pack(side="bottom", fill="x", pady=(5, 0), before=self.canvas)

    def _on_slice_scale(self, value):
        if self.volume is None:
            return
        index = int(round(float(value)))
        if index == self.slice_index:
            return
        self.slice_index = index
        if self.volume_result is None:
            self._clear_heatmap()  # a 2D heatmap belongs to the slice it was computed on
        self._show_scan_slice()

    def _show_scan_slice(self):
        """Render the slice under the slider: the raw scan, or the analysed grid once a result exists."""
        if self.volume is None:
            return
        result = self.volume_result
        if result is None:
            plain = scan_view.axial_slice_image(self.volume, self.slice_index, self.volume_window, annotate=False)
            self.scan_slice_image = plain
            self._original_pil_image = plain
            self._render_pil_image(plain)
            self.scan_info_label.configure(text=scan_view.scan_caption(self.volume, self.slice_index))
            return

        zlo, _ = result.prepared.cranial_extent
        z = zlo + self.slice_index
        plain = Image.fromarray(result.prepared.render_slice(z)).convert("RGB")
        self.scan_slice_image = plain
        self._original_pil_image = plain
        in_slab = z in result.cam_indices
        overlay = result.slice_overlay(z) if result.cam_grid is not None and in_slab else None
        self.heatmap_pil_image = overlay
        shown = overlay if (self.showing_heatmap and overlay is not None) else plain
        self._render_pil_image(shown)
        first, last = result.prepared.protocol_indices[0], result.prepared.protocol_indices[-1]
        self.scan_info_label.configure(
            text=f"Model grid 1.5 mm, axial slice {z}  |  trained band {first}-{last}: "
            + ("inside" if in_slab else "outside, no model evidence here")
        )

    def on_choose_series(self):
        if len(self.volume_series) < 2:
            return
        dialog = tk.Toplevel(self.master)
        dialog.title("Choose a series")
        dialog.transient(self.master)
        ttk.Label(dialog, text="Series in this study (most likely 3D T1 first):", padding=8).pack(anchor="w")
        listbox = tk.Listbox(dialog, width=90, height=min(16, len(self.volume_series)))
        for series in self.volume_series:
            listbox.insert(tk.END, series.label)
        listbox.selection_set(0)
        listbox.pack(fill="both", expand=True, padx=8)

        def load_selected():
            selection = listbox.curselection()
            if selection:
                key = self.volume_series[selection[0]].key
                dialog.destroy()
                self._start_scan_load(self.image_path, series_key=key)

        ttk.Button(dialog, text="Load series", command=load_selected).pack(pady=8)
        dialog.grab_set()

    # ------------------------------------------------------------------ volume analysis
    def on_analyze_volume(self):
        if self.volume is None:
            return
        self.report_html.set_html("<p><i>Scoring the whole scan and computing slice-wise Grad-CAM...</i></p>")
        self._show_saved_folder()
        self._set_busy(True, "Prediction: Analyzing volume...")
        threading.Thread(target=self._volume_worker, daemon=True).start()
        self.master.after(100, self._poll_scan_queue)

    def _volume_worker(self):
        try:
            from src import volume_analysis
            if self.volume_analyzer is None:
                self.volume_analyzer = volume_analysis.DementiaVolumeAnalyzer(device=self.device)
            self.scan_queue.put(("analysis", self.volume_analyzer.analyze(self.volume)))
        except Exception as e:
            self.scan_queue.put(("error", "Volume Analysis Error", f"The volume could not be analysed:\n{e}"))

    def _on_volume_result(self, result):
        self.volume_result = result
        zlo, zhi = result.prepared.cranial_extent
        middle = (result.cam_indices or result.prepared.protocol_indices)
        self.slice_index = middle[len(middle) // 2] - zlo
        self._configure_slice_slider(zhi - zlo + 1)
        self.showing_heatmap = result.cam_grid is not None
        self.heatmap_toggle_btn.configure(
            text="Show Original" if self.showing_heatmap else "Show Heatmap",
            state="normal" if result.cam_grid is not None else "disabled",
        )
        self.heatmap_caption_label.configure(
            text="Slice-wise Grad-CAM, scaled once for the whole volume. Not a localization of pathology."
        )
        members = ", ".join(f"{float(row[1]):.2f}" for row in result.member_probabilities)
        self.pred_label.configure(text=f"Prediction: {result.predicted_label} (research AD/CN score)")
        self.confidence_label.configure(
            text=f"P({result.class_names[1]}): {float(result.probabilities[1]):.3f}   members: {members}"
        )
        self.model_used_label.configure(text="Model Used: AD/CN research ensemble (not validated)")
        self.last_model_name = "research AD/CN ensemble"
        patient_info = {
            "name": self.patient_name_entry.get(),
            "dob": self.dob_entry.get_date().strftime('%Y-%m-%d'),
            "patient_id": self.patient_id_entry.get(),
        }
        self.last_report_html = scan_view.volume_report_html(result, patient_info, self.technique_entry.get())
        self.last_report_markdown = None
        self.report_html.set_html(self.last_report_html)
        self.save_pdf_btn.configure(state="normal")
        self._show_scan_slice()
        self._set_busy(False)

    def on_save_volume_result(self):
        if self.volume_result is None or self.volume is None:
            return
        directory = filedialog.askdirectory(title="Choose a folder for the 3D Grad-CAM volumes")
        if not directory:
            return
        self._set_busy(True, "Saving 3D Grad-CAM...")

        def worker():
            try:
                written = self.volume_result.save(directory, self.volume)
                self.scan_queue.put(("saved", directory, [os.path.basename(path) for path in written.values()]))
            except Exception as e:
                self.scan_queue.put(("error", "Save Error", f"Could not write the 3D map:\n{e}"))

        threading.Thread(target=worker, daemon=True).start()
        self.master.after(100, self._poll_scan_queue)

    def _current_image_png(self):
        """PNG bytes of what is being analysed: the displayed scan slice, or the image file."""
        if self.scan_slice_image is not None:
            buffer = io.BytesIO()
            self.scan_slice_image.save(buffer, format="PNG")
            return buffer.getvalue(), "image/png"
        with open(self.image_path, "rb") as f:
            return f.read(), "image/jpeg"

    def enable_manual_input(self):
        # Unlock fields and fill with generic defaults if empty
        widgets = [self.reason_entry, self.history_entry, self.comparison_entry, self.technique_entry, self.contrast_entry]
        defaults = ["Neurological symptoms", "Not provided", "None", "Standard MRI Brain Protocol", "None"]
        
        for w, d in zip(widgets, defaults):
            w.configure(state="normal")
            if not w.get():
                w.insert(0, d)
        
        # Now allow analysis
        self.generate_btn.configure(state="normal")

    def auto_detect_exam_details(self):
        if not self.image_path: return
        
        self.ai_fill_btn.configure(text="Detecting...", state="disabled")
        self.manual_btn.configure(state="disabled")
        
        # Run in thread
        thread = threading.Thread(target=self._run_ai_autodetect, daemon=True)
        thread.start()

    def _run_ai_autodetect(self):
        try:
            image_bytes, _ = self._current_image_png()

            prompt = (
                "Analyze only basic MRI acquisition metadata from the image appearance. "
                "If a detail is not directly supported, return 'Not provided'. "
                "Do not infer diagnosis, pathology, lesion size, lesion location, or a clinical reason from visible abnormalities. "
                "Return valid JSON only:\n"
                "{\n"
                '  "technique": "string",\n'
                '  "contrast": "string",\n'
                '  "reason": "Not provided",\n'
                '  "comparison": "None"\n'
                "}"
            )

            response = ollama.chat(
                model='llava:7b',
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_bytes]
                }],
            )
            
            # Parse JSON
            import json
            import re
            content = response['message']['content']
            match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            json_str = match.group(1) if match else content
            json_str = re.sub(r',\s*([\]}])', r'\1', json_str) # Sanitize
            
            data = json.loads(json_str)
            
            # Update UI in main thread
            self.master.after(0, lambda: self._update_exam_fields(data))

        except Exception as e:
            print(f"Auto-detect error: {e}")
            # Fallback to manual
            self.master.after(0, lambda: self._update_exam_fields(None))

    def _update_exam_fields(self, data):
        self.ai_fill_btn.configure(text="AI Auto-Detect", state="normal")
        self.manual_btn.configure(state="normal")
        self.enable_manual_input() # Unlock first and enable generate_btn
        
        if data:
            self.technique_entry.delete(0, tk.END)
            self.technique_entry.insert(0, data.get("technique", "Standard MRI"))
            
            self.contrast_entry.delete(0, tk.END)
            self.contrast_entry.insert(0, data.get("contrast", "None"))
            
            self.reason_entry.delete(0, tk.END)
            self.reason_entry.insert(0, data.get("reason", "Evaluation"))
            
            self.comparison_entry.delete(0, tk.END)
            self.comparison_entry.insert(0, data.get("comparison", "None"))
        else:
            messagebox.showwarning("AI Detection", "Could not auto-detect details. Switched to manual mode.")
        
        # Ensure analysis is enabled
        self.generate_btn.configure(state="normal")

    def _display_image(self, path: str):
        try:
            img = Image.open(path).convert("RGB")
            self._original_pil_image = img
            self._render_pil_image(img)
        except Exception as e:
            messagebox.showerror("Image Error", f"Failed to load image:\n{e}")

    def _render_pil_image(self, img: Image.Image):
        # Force layout update to get correct dimensions
        self.master.update_idletasks()
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if w <= 1 or h <= 1: w, h = 350, 350
        self._displayed_image = img
        if self.volume is not None:
            # Scan slices are small (often 224-256 px): enlarge to use the canvas,
            # then mark L/R/A/P at display size so the letters stay legible.
            scale = min(w / img.width, h / img.height)
            size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
            thumb = scan_view.annotate_orientation(img.resize(size, resample=Image.Resampling.BILINEAR))
        else:
            thumb = img.copy()
            thumb.thumbnail((w, h))
        self.tk_image = ImageTk.PhotoImage(thumb)
        self.canvas.delete("all")
        self.canvas.create_image(w // 2, h // 2, image=self.tk_image)

    def _on_canvas_resize(self, _event):
        # Redraw once the user has finished resizing; the canvas size never depends on the image.
        if self._displayed_image is None:
            return
        if self._resize_job is not None:
            self.master.after_cancel(self._resize_job)
        self._resize_job = self.master.after(120, self._redraw_after_resize)

    def _redraw_after_resize(self):
        self._resize_job = None
        if self._displayed_image is not None:
            self._render_pil_image(self._displayed_image)

    def _toggle_heatmap_view(self):
        if self.volume_result is not None:
            self.showing_heatmap = not self.showing_heatmap
            self.heatmap_toggle_btn.configure(text="Show Original" if self.showing_heatmap else "Show Heatmap")
            self._show_scan_slice()
            return
        if self.heatmap_pil_image is None or self._original_pil_image is None:
            return
        self.showing_heatmap = not self.showing_heatmap
        if self.showing_heatmap:
            self._render_pil_image(self.heatmap_pil_image)
            self.heatmap_toggle_btn.configure(text="Show Original")
        else:
            self._render_pil_image(self._original_pil_image)
            self.heatmap_toggle_btn.configure(text="Show Heatmap")

    def on_analyze_and_generate(self):
        if not self.image_path: return
        if self.volume_result is not None:
            # Back to the raw scan: the single-slice classifiers work on native slices.
            native_z = self.volume.shape[2] // 2
            self.volume_result = None
            self.save_volume_btn.configure(state="disabled")
            self.slice_index = native_z
            self._configure_slice_slider(self.volume.shape[2])
            self._clear_heatmap()
            self._show_scan_slice()

        self.generate_btn.configure(state="disabled")
        self.choose_img_btn.configure(state="disabled")
        self.save_pdf_btn.configure(state="disabled")
        self.pred_label.configure(text="Prediction: Analyzing...")
        self.confidence_label.configure(text="Confidence: -")
        self.model_used_label.configure(text="Model Used: -")
        self.report_html.set_html("<p><i>Running multi-stage analysis and generating grounded report...</i></p>")
        self.progress_bar.pack(fill="x", pady=5)
        self.progress_bar.start()
        self.update_idletasks()

        try:
            result = self._run_cascaded_classification()
            pred_label, pred_conf, model_name = result.label, result.confidence, result.model_name
            self.last_model_name = model_name

            self.pred_label.configure(text=f"Prediction: {pred_label}")
            self.confidence_label.configure(text=f"Confidence: {pred_conf:.2f}%")
            self.model_used_label.configure(text=f"Model Used: {model_name}")

            self.heatmap_pil_image = self._run_grad_cam(result)
            self.showing_heatmap = False
            stage_caution = self._raw_scan_stage_caution(model_name)
            if self.heatmap_pil_image is not None:
                self.showing_heatmap = True
                self._render_pil_image(self.heatmap_pil_image)
                self.heatmap_toggle_btn.configure(text="Show Original", state="normal")
                self.heatmap_caption_label.configure(
                    text=(
                        f"Grad-CAM: regions that most influenced the {model_name} prediction. "
                        "Not a diagnostic localization of pathology. " + stage_caution
                    )
                )
            else:
                self.heatmap_toggle_btn.configure(text="Show Heatmap", state="disabled")
                self.heatmap_caption_label.configure(text=stage_caution)
            if stage_caution:
                self.model_used_label.configure(text=f"Model Used: {model_name} - raw scan slice, outside training domain")

            confidence_str = f"{pred_conf:.1f}%"
            patient_info = {
                "name": self.patient_name_entry.get(),
                "dob": self.dob_entry.get_date().strftime('%Y-%m-%d'),
                "patient_id": self.patient_id_entry.get(),
            }

            # Collect user inputs
            user_inputs = {
                "reason": self.reason_entry.get(),
                "history": self.history_entry.get(),
                "comparison": self.comparison_entry.get(),
                "technique": self.technique_entry.get(),
                "contrast": self.contrast_entry.get()
            }

            self._start_report_generation(pred_label, confidence_str, patient_info, user_inputs, model_name)

        except Exception as e:
            messagebox.showerror("Analysis Error", f"An error occurred:\n{e}")
            self.generate_btn.configure(state="normal")
            self.choose_img_btn.configure(state="normal")
            self.progress_bar.stop()
            self.progress_bar.pack_forget()

    def _raw_scan_stage_caution(self, model_name):
        """Non-empty when the 4-class stage model was just applied to a slice of a raw scan."""
        if self.volume is not None and model_name and "Alzheimer" in model_name:
            return RAW_SCAN_STAGE_CAUTION
        return ""

    @torch.no_grad()
    def _run_cascaded_classification(self):
        # A loaded scan is classified on the slice currently displayed; files are read from disk.
        scan_slice = getattr(self, "scan_slice_image", None)
        img = scan_slice.convert("RGB") if scan_slice is not None else Image.open(self.image_path).convert("RGB")
        
        # 1. Gatekeeper Phase (Routing)
        target_domain = "tumor" # Default fallback
        gate_conf = 0.0

        if self.gatekeeper_model:
            # ResNet50 typically uses 224x224
            tensor = self.gate_tfms(img).unsqueeze(0).to(self.device)
            # Use appropriate autocast for the device
            if self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    output = self.gatekeeper_model(tensor)
            else:
                output = self.gatekeeper_model(tensor)

            if self.gatekeeper_mode == "binary":
                probs = torch.softmax(output, dim=1).squeeze(0)
                top_prob, top_idx = torch.topk(probs, 1)
                class_idx = top_idx.item()
                gate_conf = top_prob.item()
                target_domain = self.gatekeeper_classes[class_idx].lower()
                print(f"Binary router: {target_domain} scan detected (confidence {gate_conf*100:.2f}%)")
            else:
                # Legacy multi-class output: [Normal, Tumor, Dementia]
                probs = torch.softmax(output, dim=1).squeeze(0)
                top_prob, top_idx = torch.topk(probs, 1)
                class_idx = top_idx.item()
                gate_conf = top_prob.item()

                if class_idx == 0:
                    target_domain = "normal"
                    print(f"Gatekeeper: Normal/No Tumor detected (confidence {gate_conf*100:.2f}%)")
                    return ClassificationResult("Normal", gate_conf * 100.0, "Gatekeeper Model", img)
                elif class_idx == 1:
                    target_domain = "tumor"
                    print(f"Gatekeeper: Tumor scan detected (confidence {gate_conf*100:.2f}%)")
                elif class_idx == 2:
                    target_domain = "dementia"
                    print(f"Gatekeeper: Dementia detected (confidence {gate_conf*100:.2f}%)")

        # 2. Specialized Classification Phase
        if target_domain == "dementia" and self.alz_model:
            tensor_alz = self.alz_tfms(img).unsqueeze(0).to(self.device)
            if self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    logits = self.alz_model(tensor_alz)
            else:
                logits = self.alz_model(tensor_alz)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            top_prob, top_idx = torch.topk(probs, 1)
            return ClassificationResult(
                self.alz_classes[top_idx.item()], top_prob.item() * 100.0, "Alzheimer's/Dementia Model", img,
                cam_model=self.alz_model, cam_transform=self.alz_tfms, cam_class_idx=top_idx.item(),
            )

        elif target_domain == "tumor" and self.tumor_model:
            tensor = self.tumor_tfms(img).unsqueeze(0).to(self.device)
            if self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    logits = self.tumor_model(tensor)
            else:
                logits = self.tumor_model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            top_prob, top_idx = torch.topk(probs, 1)
            return ClassificationResult(
                self.tumor_classes[top_idx.item()], top_prob.item() * 100.0, "Brain Tumor Model", img,
                cam_model=self.tumor_model, cam_transform=self.tumor_tfms, cam_class_idx=top_idx.item(),
            )

        # Fallback if specific model fails but we have a general prediction or fallback model
        if target_domain == "normal":
             return ClassificationResult("Normal", gate_conf * 100.0, "Gatekeeper Model", img)

        # Fallback if preferred model isn't loaded but the other is
        if self.tumor_model:
            tensor = self.tumor_tfms(img).unsqueeze(0).to(self.device)
            logits = self.tumor_model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            top_prob, top_idx = torch.topk(probs, 1)
            return ClassificationResult(
                self.tumor_classes[top_idx.item()], top_prob.item() * 100.0, "Brain Tumor Model (Fallback)", img,
                cam_model=self.tumor_model, cam_transform=self.tumor_tfms, cam_class_idx=top_idx.item(),
            )

        if self.alz_model:
            tensor_alz = self.alz_tfms(img).unsqueeze(0).to(self.device)
            logits = self.alz_model(tensor_alz)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            top_prob, top_idx = torch.topk(probs, 1)
            return ClassificationResult(
                self.alz_classes[top_idx.item()], top_prob.item() * 100.0, "Alzheimer's Model (Fallback)", img,
                cam_model=self.alz_model, cam_transform=self.alz_tfms, cam_class_idx=top_idx.item(),
            )

        raise Exception("No models loaded or available.")

    def _run_grad_cam(self, result: "ClassificationResult") -> Optional[Image.Image]:
        if result.cam_model is None or result.cam_transform is None or result.cam_class_idx is None:
            return None
        try:
            target_layer = get_target_layer(result.cam_model)
            tensor = result.cam_transform(result.source_image).unsqueeze(0).to(self.device)
            cam = GradCAM(result.cam_model, target_layer).generate(tensor, result.cam_class_idx)
            if cam is None:
                return None
            return overlay_cam_on_image(result.source_image, cam)
        except Exception as e:
            print(f"Grad-CAM generation error: {e}")
            return None

    def _start_report_generation(self, prediction, confidence_str, patient_info, user_inputs, model_name):
        while not self.report_queue.empty():
            self.report_queue.get_nowait()

        thread = threading.Thread(
            target=self._generate_report_threaded,
            args=(prediction, confidence_str, patient_info, user_inputs, model_name),
            daemon=True
        )
        thread.start()
        self.master.after(100, self._check_report_queue)

    def _generate_report_threaded(self, prediction, confidence_str, patient_info, user_inputs, model_name):
        try:
            report_date = datetime.now().strftime("%B %d, %Y")
            report = build_grounded_report(
                prediction=prediction,
                confidence=confidence_str,
                patient_info=patient_info,
                exam_details=user_inputs,
                report_date=report_date,
                model_used=model_name,
                source_image=self._source_description(),
            )
            html_text = render_report_html(report)
            caution = self._raw_scan_stage_caution(model_name)
            if caution:
                html_text = f'<p style="color:#a00000;"><b>{caution}</b></p>' + html_text
            self.last_report_html = html_text
            self.last_report_markdown = render_report_markdown(report)
            self.report_queue.put(html_text)

        except Exception as e:
            self.report_queue.put(f"<p><b>Error generating report:</b></p><pre>{e}</pre>")

    def _source_description(self):
        """What the report says was analysed; a scan is described, never by a patient-bearing path."""
        if self.volume is None:
            return self.image_path
        return f"{scan_view.scan_caption(self.volume, self.slice_index)} (single-slice classification)"

    def _check_report_queue(self):
        try:
            html_result = self.report_queue.get_nowait()
            self.last_report_html = html_result
            self.report_html.set_html(html_result)
            self.progress_bar.stop()
            self.progress_bar.pack_forget()
            self.generate_btn.configure(state="normal")
            self.choose_img_btn.configure(state="normal")
            self.save_pdf_btn.configure(state="normal")
        except queue.Empty:
            self.master.after(100, self._check_report_queue)

    def on_save_pdf(self):
        if not self.last_report_html: return

        filepath = filedialog.asksaveasfilename(
            title="Save Report as PDF",
            defaultextension=".pdf",
            filetypes=[("PDF Documents", "*.pdf"), ("All Files", "*.*")],
        )
        if not filepath: return

        password = simpledialog.askstring("PDF Encryption", "Enter a password (optional):", show='*')

        try:
            # Use the already formatted HTML
            pdf_html = self.last_report_html
            
            # Embed image
            image_data_uri = None
            try:
                image_bytes, mime = self._current_image_png()
                image_data_uri = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('utf-8')}"
            except Exception: pass

            if image_data_uri:
                pdf_html += f'<div style="text-align: center; margin-top: 20px;"><img src="{image_data_uri}" alt="MRI Scan" style="max-width: 300px; height: auto;"></div>'

            # Embed Grad-CAM heatmap, if one was generated for this analysis
            if self.heatmap_pil_image is not None:
                try:
                    heatmap_buffer = io.BytesIO()
                    self.heatmap_pil_image.save(heatmap_buffer, format="PNG")
                    heatmap_data_uri = "data:image/png;base64," + base64.b64encode(heatmap_buffer.getvalue()).decode('utf-8')
                    model_label = self.last_model_name or "specialist"
                    pdf_html += (
                        '<div style="text-align: center; margin-top: 20px;">'
                        f'<img src="{heatmap_data_uri}" alt="Grad-CAM heatmap" style="max-width: 300px; height: auto;">'
                        '<p style="font-size: 0.85em; color: #555; max-width: 400px; margin: 6px auto 0;">'
                        f'Grad-CAM visualization: highlights the image regions that most influenced the {model_label} '
                        'classifier\'s predicted class. This is a visualization of model attention, not a validated '
                        'diagnostic localization of pathology, and must be interpreted only alongside direct '
                        'radiologist review of the original MRI images.'
                        '</p></div>'
                    )
                except Exception:
                    pass

            # Add CSS for Page Numbers in WeasyPrint
            css = """
            @page {
                size: Letter;
                margin: 2cm;
                @bottom-right {
                    content: "Page " counter(page) " of " counter(pages);
                    font-family: sans-serif;
                    font-size: 9pt;
                    color: #555;
                }
            }
            body { font-family: sans-serif; font-size: 10pt; } 
            h1, h2, h3 { color: #333; } 
            table { border-collapse: collapse; width: 100%; } 
            td, th { padding: 4px; text-align: left; border-bottom: 1px solid #ddd; }
            """
            html_with_style = f"<style>{css}</style>{pdf_html}"
            
            pdf_buffer = io.BytesIO()
            WeasyHTML(string=html_with_style, base_url=self.image_path).write_pdf(pdf_buffer)
            pdf_buffer.seek(0)

            if password:
                reader = PdfReader(pdf_buffer)
                writer = PdfWriter()
                for page in reader.pages: writer.add_page(page)
                writer.encrypt(password)
                with open(filepath, "wb") as f: writer.write(f)
            else:
                with open(filepath, "wb") as f: f.write(pdf_buffer.read())
            
            messagebox.showinfo("Success", f"Report saved to:\n{filepath}")

        except Exception as e:
            messagebox.showerror("PDF Error", f"Failed to save PDF:\n{e}")

def main():
    root = tk.Tk()
    root.title(APP_TITLE)
    root.geometry("1200x880")
    try:
        style = ttk.Style()
        style.theme_use("vista" if sys.platform.startswith("win") else "clam")
    except:
        pass
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
