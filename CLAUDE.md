# CLAUDE.md — AI Assistant Guide for Tumor-Database

## Project Overview

This is a **medical imaging AI system** for classifying brain MRI scans. It uses a hierarchical deep learning pipeline to detect and classify brain tumors (glioma, meningioma, pituitary) and Alzheimer's-related dementia stages (mild, moderate, very mild). A Tkinter-based GUI generates professional radiology-style PDF reports, optionally augmented by a local LLM (Ollama/llava:7b).

---

## Repository Structure

```
Tumor-Database/
├── src/                            # All Python source code
│   ├── radiology_report_gui.py     # Main GUI application (886 lines) — primary entry point
│   ├── train_complete_suite.py     # Master training script for all 3 models (394 lines)
│   ├── train_gatekeeper.py         # Standalone Gatekeeper (ResNet50) training
│   ├── train_alzheimers.py         # Standalone Dementia model (MobileNetV3) training
│   ├── gatekeeper_model.py         # ResNet50 architecture definition (35 lines)
│   ├── data_loader.py              # Dataset loading and transforms (134 lines)
│   ├── mri_volume.py               # Canonical RAS+ volume container, NIfTI I/O, PHI kept separate
│   ├── dicom_io.py                 # DICOM files/folders/zips -> MRIVolume (geometry from the header)
│   ├── volume_analysis.py          # Whole-scan AD/CN research scoring + stacked 3D Grad-CAM
│   ├── train_dementia_volume.py    # Patient-level, cross-validated training on labelled 3D scans
│   ├── oasis1.py                   # OASIS-1 index + loader (data obtained separately, never committed)
│   └── scan_view.py                # Slice rendering / header-derived form values for the GUI
├── data/
│   ├── brain_tumor/
│   │   ├── Training/{glioma,meningioma,notumor,pituitary}/   # ~5,700 training images
│   │   └── Testing/{glioma,meningioma,notumor,pituitary}/    # ~1,300 test images
│   ├── alzheimers/
│   │   ├── MildDemented/
│   │   ├── ModerateDemented/
│   │   ├── NonDemented/
│   │   └── VeryMildDemented/                                  # ~44,000 total images
│   └── evaluation/
│       ├── images/                   # 100-image radiologist test set (anonymized)
│       ├── ground_truth/             # radiologist_test_key.csv (labels)
│       ├── model_results/            # model_predictions.csv
│       └── radiologist_results/      # Per-radiologist xlsx files
├── models/
│   ├── gatekeeper_classifier.pt      # ResNet50 router (~17MB, git-ignored)
│   ├── brain_tumor_classifier.pt     # EfficientNet-B3 (~43MB, git-ignored)
│   ├── alzheimers_classifier.pt      # MobileNetV3-Large (~17MB, git-ignored)
│   └── backup/                       # Model checkpoints
├── data_visualization/
│   ├── visualize_performance.py      # General performance metrics + charts
│   ├── evaluate_tumor_only.py        # Dedicated tumor model evaluation
│   └── compare_rad_vs_ai.py          # Radiologist vs AI comparison analysis
├── training_logs/                    # PNG plots + JSON history from training runs
├── Sample Reports/                   # Anonymized sample PDF outputs
├── .github/workflows/                # GitHub Actions (Gemini integration)
├── README.md                         # Installation and usage guide
├── GEMINI.md                         # Development log and architectural decisions
└── requirements.txt                  # Python dependencies
```

---

## Architecture: Hierarchical Classification Pipeline

The system uses a **3-stage routing pipeline** — never collapse this into a single model:

```
Input MRI Image (224×224 RGB)
        │
        ▼
┌─────────────────────────────────┐
│  Stage 1: Gatekeeper (ResNet50) │  3-way classifier
│  → class 0: Normal              │
│  → class 1: Tumor               │
│  → class 2: Dementia            │
└─────────────────────────────────┘
        │               │
        ▼               ▼
┌──────────────┐  ┌──────────────────────────────┐
│ Stage 2:     │  │ Stage 3:                     │
│ Tumor        │  │ Dementia Specialist           │
│ Specialist   │  │ (MobileNetV3-Large)           │
│ (EfficientNet│  │ → MildDemented               │
│  -B3)        │  │ → ModerateDemented           │
│ → Glioma     │  │ → VeryMildDemented           │
│ → Meningioma │  └──────────────────────────────┘
│ → Pituitary  │
└──────────────┘
```

**Key design decisions (do not change without strong justification):**
- The Gatekeeper unifies `notumor` (brain_tumor dataset) and `NonDemented` (Alzheimer's dataset) into a single **"Normal"** class (label 0). This avoids overlap between the two specialist domains.
- Each model is saved as a **complete PyTorch object** (`torch.save(model, path)`), not just state_dict. Load with `torch.load(path, map_location=device)`. *This describes the legacy training scripts only.* The experiment pipeline, the research rounds and `src/train_dementia_volume.py` save a dict holding a state dict plus metadata (`model_state` or `model`); that is the format of the current production checkpoints, and the GUI loads both. Match the format of the loader you are targeting.
- Device priority: CUDA → DirectML (Windows) → CPU.

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | PyTorch + TorchVision |
| Model Architectures | ResNet50, EfficientNet-B3, MobileNetV3-Large |
| Image Processing | OpenCV, Pillow, torchvision.transforms |
| GUI | Tkinter + tkcalendar + tkhtmlview |
| PDF Generation | WeasyPrint (HTML/CSS → PDF) |
| LLM Integration | Ollama (`llava:7b`, local, optional) |
| Data Analysis | pandas, NumPy, scikit-learn, matplotlib, seaborn |
| Dataset Download | kagglehub |
| Scan Input | pydicom (+ pylibjpeg decoders), nibabel, SciPy |

---

## Running the Project

### Prerequisites

```bash
pip install -r requirements.txt
# Linux also needs: sudo apt install python3-tk libgtk-3-dev
# macOS: brew install gobject-introspection gtk+3
# Ollama (optional): https://ollama.com — pull llava:7b
```

### Main Application (GUI)

```bash
python -m src.radiology_report_gui
```

- GUI loads all 3 models from `models/` at startup
- Upload a brain MRI → classify → generate PDF report
- AI auto-detection tab uses `llava:7b` via Ollama for image understanding

### Training All Models

```bash
python src/train_complete_suite.py
```

Trains sequentially: Gatekeeper → Tumor Specialist → Dementia Specialist. Saves models to `models/` and plots to `training_logs/`.

Individual model training:
```bash
python src/train_gatekeeper.py
python src/train_alzheimers.py
```

### DICOM and Volume Analysis

```bash
python -m src.dicom_io <file|folder|zip>                       # list series
python -m src.dicom_io <path> --to-nifti scan.nii.gz           # canonical NIfTI + sidecar
python -m src.volume_analysis <scan> --output results/         # research AD/CN score + 3D Grad-CAM
python scripts/verify_dicom_volume_pipeline.py --openneuro-root data/external/openneuro_ds007561_v1
```

See `docs/DICOM_VOLUME_ANALYSIS.md` for what was verified and the limits.

### Patient-Level Volume Training

```bash
python -m src.oasis1 data/external/oasis1 --qc oasis_qc.png      # cohort counts + orientation sheet: look at it first
python -m src.train_dementia_volume --oasis-root data/external/oasis1 --output training_logs/experiments/dementia_volume/oasis1
python -m src.train_dementia_volume --manifest cohort.csv --cohort-name NAME --output <new folder>
```

Splits are by person; every person is scored once by a fold model that never saw them. See
`docs/OASIS1_VOLUME_TRAINING.md`. As of 2026-11-14 this has not been run on OASIS-1 (the data is not in the repo).

### Tests

```bash
python -m pytest -q tests/                                     # full suite
xvfb-run -a python -m pytest -q tests/test_gui_scan_flow.py    # GUI scan flow needs tkinter + a display
python -m unittest discover -s tests -p 'test_*.py'            # what CI runs (numpy/scipy/Pillow/pydicom only)
```

New tests must skip, not error, when torch / nibabel / pydicom / a display are missing: the CI
verification job installs only `requirements-verification.txt`.

### Evaluation and Visualization

```bash
python data_visualization/evaluate_tumor_only.py    # Tumor model on 100-image test set
python data_visualization/visualize_performance.py  # Training metrics and confusion matrices
python data_visualization/compare_rad_vs_ai.py      # Radiologist vs AI performance comparison
```

---

## Data Conventions

- **Image format:** JPG/PNG/JPEG, resized to **224×224** for all models
- **Normalization:** ImageNet statistics — `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
- **Train/val split:** 80/20 with `seed=42` — always use this seed for reproducibility
- **Data augmentation (training only):** random horizontal flip, rotation ±15°, color jitter; never augment validation/test sets
- **Class balance:** `data_loader.py` uses undersampling to handle class imbalance in Alzheimer's dataset

---

## Code Conventions

### Style
- Python 3.10+; type hints in function signatures where practical
- Descriptive variable names: `pred_label`, `confidence_str`, `self.gatekeeper_model`
- Use `logging` module for training progress (not bare `print`)
- `try/except` blocks around model I/O and GUI operations
- Keep GUI operations off the main thread for long-running tasks (use `threading`)

### PyTorch Patterns
```python
# Inference — always use this pattern
model.eval()
with torch.no_grad():
    output = model(input_tensor)

# Training — always call these
model.train()
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Save complete model object (not state_dict)
torch.save(model, "models/my_model.pt")

# Load with device mapping
model = torch.load("models/my_model.pt", map_location=device)
```

### Model Architecture Pattern
Each model file follows this structure:
1. Architecture function: `build_<name>_model(num_classes, device) → nn.Module`
2. Transform function: `get_<name>_transform() → transforms.Compose`
3. Dataset class (if custom needed)
4. Training loop with epoch logging
5. Save model artifact

### File Naming
- Model files: `<name>_classifier.pt`
- Training plots: `training_logs/<name>_<metric>.png`
- Evaluation outputs: `data_visualization/comparison/<description>.png`

---

## Performance Benchmarks (as of Feb 2026)

| Model | Task | Accuracy |
|-------|------|----------|
| Gatekeeper (ResNet50) | Normal/Tumor/Dementia routing | ~80% |
| Tumor Specialist (EfficientNet-B3) | Glioma/Meningioma/Pituitary | ~80-83% |
| Dementia Specialist (MobileNetV3) | 4-stage Alzheimer's | ~80% |
| **Radiologist Average** | Brain tumor classification | **37-39.67%** |

The AI outperforms the 3-radiologist human baseline by ~40 percentage points on the 100-image evaluation set.

---

## Git Workflow

- **Main branch:** `master`
- **Feature/AI branches:** prefix `claude/` for Claude sessions, `gemini/` for Gemini sessions
- **Commit message conventions:**
  - `feat:` — new features
  - `refactor:` — code restructuring
  - `docs:` — documentation changes
  - `fix:` — bug fixes
  - `train:` / `eval:` — model training and evaluation work

### What is Git-Ignored
- `models/*.pt` — trained model files (too large; regenerate with training scripts)
- `models/Collab_Train/Train_Config/data.zip` — Colab training data archive

---

## Key Files for AI Assistants

When making changes, these are the most critical files to understand:

| Priority | File | Why |
|----------|------|-----|
| High | `src/radiology_report_gui.py` | Central application; model loading, inference pipeline, GUI |
| High | `src/train_complete_suite.py` | Authoritative training configuration for all models |
| High | `src/data_loader.py` | Dataset abstractions used by all training scripts |
| Medium | `src/gatekeeper_model.py` | Gatekeeper architecture definition |
| Medium | `data_visualization/compare_rad_vs_ai.py` | Evaluation methodology and metrics |
| Reference | `GEMINI.md` | Detailed development history and architectural rationale |
| Reference | `README.md` | Installation and platform-specific setup |

---

## Common Tasks and Pitfalls

### Adding a New Model
1. Create `src/train_<name>.py` following existing training script patterns
2. Register model loading in `src/radiology_report_gui.py` (model init section)
3. Add transform function to `src/data_loader.py` or new model file
4. Save to `models/<name>_classifier.pt`

### Modifying the Gatekeeper
- The unified "Normal" class (index 0) **must** remain the first class; downstream label mapping depends on this ordering
- Any change to class indices requires updating `radiology_report_gui.py` label maps

### Adding GUI Features
- Long operations (model inference, LLM calls) must run in a background thread; use `threading.Thread`
- PDF generation uses WeasyPrint — pass HTML strings, not file paths
- Calendar widget from `tkcalendar` (not standard Tkinter)

### Cross-Platform Notes
- GPU detection tries CUDA first, then `torch_directml` (Windows), then CPU — do not assume CUDA
- WeasyPrint requires GTK3 on Windows (separate installer); Linux needs `libgtk-3-dev`
- Ollama must be running locally before launching the GUI if using AI auto-detection

### Working With Scans (DICOM / NIfTI)
- Every reader returns an `MRIVolume`: voxels in closest-canonical RAS+ order plus the voxel-to-world affine. Keep that single convention; it is what lets a slice-wise Grad-CAM stack overlay the scan.
- Take geometry from the header (`ImagePositionPatient`, `ImageOrientationPatient`, `PixelSpacing`), never from file names, `InstanceNumber` or `SliceThickness`.
- Identifying header fields live only in `MRIVolume.identity`. Never log them or write them to exports, sidecars or result JSON.
- `prepare_clinical_volume` must stay pixel-identical to `docs/kevin_paper_20261011/preprocess_clinical_t1.py`; the clinical checkpoints were trained on exactly that input. Re-run `scripts/verify_dicom_volume_pipeline.py` after touching it.
- When stacking per-slice Grad-CAM into a volume, keep the raw maps and scale once for the whole volume. Per-slice min-max normalisation makes every slice look equally important.
- The four-class stage model was trained on skull-stripped, atlas-aligned slices. On raw with-skull slices it labels everything as demented; do not present its label on a raw scan as valid.

### Training on Labelled Scans
- The person is the unit: split, sample, count and report by person. Never let slices or repeat scans of one person fall on both sides of a split.
- Report internal cross-validation as exactly that. It is not external validation, and a cohort that earlier rounds already inspected is not a fresh test.
- Per-person predictions and patient tables stay in git-ignored locations (`data/external/`, `training_logs/experiments/`, a run's `private/`). Commit aggregate summaries only.
- State a fact about a dataset (counts, scanner, access route) only if it was checked against a source in that session or is already recorded in the project's dataset registry.
- Open dict-style checkpoints with `torch.load(..., weights_only=True)` (`volume_analysis.load_clinical_checkpoint` does). Only the legacy whole-object checkpoints need full unpickling; never load one of those from outside this repository.

### Frozen Evidence (do not edit)
- `research/dementia_round3/*.py` and `tests/test_round3*.py` are hash-pinned in `docs/kevin_paper_20261011/round3/implementation_verification.json`.
- `docs/kevin_paper_20261011/**` and the review bundles are frozen experiment records.
- Checkpoints under `models/` are hash-pinned across the evidence trail. Never overwrite one; a new model gets a new file name.
- `.gitattributes` marks the frozen bundles `-text`. Several sealed CSVs there have CRLF line endings on purpose: `csv.DictWriter` wrote them that way and their checksums cover those bytes. Never renormalise them (`git add --renormalize`) or let an editor convert them; one changed byte fails `scripts/check_explainability_bundle.py` and `scripts/check_validation_review.py`.
- When a recorded hash does not match, restore the original bytes. Never edit the recorded hash.
- Put new functionality in new modules rather than changing the pinned ones.

### Do Not
- Do not merge `notumor` and `NonDemented` into separate "Normal" classes — this breaks the Gatekeeper routing logic
- Do not change the checkpoint format a loader expects. Legacy scripts save complete model objects; pipeline, research and volume-trainer checkpoints are dicts with a state dict plus metadata (see Key design decisions)
- Do not run data augmentation on validation or test splits
- Do not hardcode device as `cuda` — always use the device detection pattern from existing code

---

## CI/CD

GitHub Actions workflows in `.github/workflows/` are for Gemini AI integration (triage, review, dispatch). There are no automated test or build pipelines currently. Evaluation is run manually via the scripts in `data_visualization/`.
