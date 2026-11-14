# AI-Powered Neuro-Radiology Report Generator

> **2026-09-20 review: submission blocked.** A fresh decoded-RGB audit found 125 cross-split duplicate groups (301 tumor-image rows) in the historical primary manifest. Dementia has no decoded-pixel overlap, but upstream augmentation and patient independence remain unresolved. The perceptual sensitivity manifest has zero file/pixel/identical-dHash cross-split overlap. See [the final review](docs/FINAL_REVIEW_20260920.md) before interpreting historical tables.

> **2026-10-07 current-checkpoint check:** The existing models score **100/100** on the radiologist image benchmark, with 100 finite, nonflat Grad-CAM maps and unchanged predictions after explanation. This is a reused benchmark: its images match 66 training, 9 validation, and 25 test images in the historical strict manifest. No new training or independent accuracy improvement is claimed. See [results, fixes, and reproduction](docs/evaluation_100_20261007/README.md).

> **2026-10-11 Kevin dementia experiment:** Six research runs completed 305 epochs. The validation-selected new checkpoint and the old baseline both score 100/100 on the reused reader benchmark; the new checkpoint falls to 96/100 under one fixed added-noise diagnostic. The separate clinical pilot performed poorly. Overall superiority and clinical readiness are not established, and the production checkpoint is unchanged. See the [complete paper materials](docs/kevin_paper_20261011/publication/README.md), [recreated figures](docs/kevin_paper_20261011/publication/FIGURES.md), and [all six research checkpoints](models/kevin_20261011/README.md).

> **2026-11-14 DICOM input and volume analysis:** The app and CLI now read DICOM series (files, folders, zips; classic and Enhanced multi-frame) and NIfTI volumes, and can score a whole T1 scan with the research AD/CN ensemble and stack per-slice Grad-CAM into a 3D map that overlays the scan. Geometry agrees with `dcm2niix` on 98/98 test series, and the model input is pixel-identical to the frozen research preprocessing on 20/20 real scans. **No model was retrained.** The same check showed that the four-class stage model labels raw with-skull slices as demented regardless of diagnosis (180/180 slices, including all 14 cognitively normal participants), so its label is flagged as unreliable on raw scans. See [DICOM input and volume analysis](docs/DICOM_VOLUME_ANALYSIS.md).

This project is an advanced medical imaging analysis tool. It uses deep learning to analyze MRI scans for both **Brain Tumors** and **Alzheimer's/Dementia** signs, then generates grounded draft reports from structured classifier evidence. The reporting path is deterministic by default so it does not ask an LLM to invent lesion measurements, locations, mass effect, atrophy measurements, or other unsupported clinical findings.

## Features

-   **Hierarchical Classification:** Uses a multi-stage AI pipeline for maximum accuracy:
    -   **Current Router:** A `ResNet50` binary classifier routes an MRI to the **Tumor** or **Dementia** specialist. The GUI also supports older three-way gatekeeper checkpoints.
    -   **Specialized Classifiers:**
        -   **Brain Tumor Classifier:** `EfficientNet-B3` (PyTorch) for Glioma, Meningioma, Pituitary, or No Tumor.
        -   **Alzheimer's Classifier:** `MobileNetV3-Large` (PyTorch) for Mild, Moderate, Very Mild, or NonDemented.
    -   **Checkpoint Preprocessing:** Classification and Grad-CAM use the specialist checkpoint's image size and training normalization.
-   **DICOM and NIfTI Input:** Opens a DICOM file, a study folder or a zipped study (classic series and Enhanced multi-frame MR) or a NIfTI volume. Slices are ordered and spaced from the header geometry, the most likely 3D T1 series is chosen automatically, and patient name, ID, date of birth and technique are taken from the header. Identifying fields are never written to exports.
-   **Whole-Scan Research Analysis with 3D Grad-CAM:** Scores a T1 volume with the research AD/CN ensemble and computes Grad-CAM for every slice of the analysed slab, scaled once for the whole volume and saved as NIfTI on both the model grid and the scan's own grid. Research output only; see the limits in [the DICOM document](docs/DICOM_VOLUME_ANALYSIS.md).
-   **Grounded Report Maker:** Generates draft reports from the classifier label, confidence, user-entered exam details, and model validation context. Unsupported findings are explicitly marked as not assessed instead of being hallucinated.
-   **Optional AI Metadata Assist:** The GUI can use a multimodal AI model only for basic acquisition metadata. The prompt refuses diagnosis/pathology inference and falls back to manual entry when uncertain.
-   **Standardized Reporting:** Generates reports with a strict evidence schema for consistency across Findings, Impression, Technique, classifier evidence, and safety limitations.
-   **PDF Export:** Saves draft reports as PDF documents with the MRI image embedded, timestamped footer, page numbering, and optional password encryption.
-   **User-Friendly GUI:** A modern Tkinter interface with tabs for Patient Info and Exam Details.
-   **Data Visualization:** Includes a suite to benchmark model performance and generate accuracy heatmaps.

## Project Structure

-   `src/radiology_report_gui.py`: The main application.
-   `src/train_complete_suite.py`: The master script to train all 3 models sequentially.
-   `src/gatekeeper_model.py`: Definition of the routing model.
-   `src/dicom_io.py`, `src/mri_volume.py`: DICOM/NIfTI reading into one canonical volume representation.
-   `src/volume_analysis.py`: Whole-scan AD/CN research scoring and the stacked 3D Grad-CAM map.
-   `scripts/verify_dicom_volume_pipeline.py`: Regenerates the DICOM/volume verification record.
-   `data_visualization/`:
    -   `visualize_performance.py`: General model benchmark tools.
    -   `compare_rad_vs_ai.py`: Script to compare AI accuracy vs human radiologist.
    -   `evaluate_tumor_only.py`: Dedicated evaluation script for the Brain Tumor classifier.
-   `models/`:
    -   `binary_router.pt`: Current ResNet50 binary router model (`gatekeeper_classifier.pt` is a legacy GUI fallback).
    -   `brain_tumor_classifier.pt`: EfficientNet-B3 model.
    -   `alzheimers_classifier.pt`: MobileNetV3 model.
-   `data/`:
    -   `evaluation/`: Contains test images, ground truth keys, and radiologist/AI results.
    -   `alzheimers/`: Training data for dementia.
    -   `brain_tumor/`: Training data for tumors.

## Installation Guide

### Prerequisites
- **Python 3.10+**
- **Ollama** (optional, only for metadata assist; grounded report generation does not require an LLM)

### 🐧 Linux (Ubuntu/Debian)

1.  **Install System Dependencies**
    The GUI requires Tkinter, and the PDF generator (`WeasyPrint`) needs specific system libraries.
    ```bash
    sudo apt update
    sudo apt install python3-tk libpango-1.0-0 libharfbuzz0b libpangoft2-1.0-0
    ```

2.  **Install Python Dependencies**
    ```bash
    pip install -r requirements.txt --break-system-packages
    ```
    *(Note: Using a virtual environment `venv` is recommended if you prefer not to install system-wide.)*

3.  **Optional: Install & Setup Ollama**
    -   Install Ollama:
        ```bash
        curl -fsSL https://ollama.com/install.sh | sh
        ```
    -   Start the Ollama server:
        ```bash
        ollama serve
        ```
    -   In a new terminal, pull the required multimodal model:
        ```bash
        ollama pull llava:7b
        ```

### 🍎 macOS

1.  **Install System Dependencies (via Homebrew)**
    You need Homebrew installed first.
    ```bash
    brew install python-tk pango libffi
    ```

2.  **Install Python Dependencies**
    ```bash
    pip3 install -r requirements.txt
    ```

3.  **Optional: Install & Setup Ollama**
    -   Download and install Ollama from [ollama.com/download/mac](https://ollama.com/download/mac).
    -   Open the Ollama application.
    -   Run the following in your terminal to download the model:
        ```bash
        ollama pull llava:7b
        ```

### 🪟 Windows

1.  **Install Python**
    Download and install Python 3.10+ from [python.org](https://www.python.org/downloads/). Ensure you check **"Add Python to PATH"** during installation.

2.  **Install GTK3 Runtime (Crucial for PDF Export)**
    The PDF generation library (`WeasyPrint`) requires the GTK3 runtime.
    -   Download and install the [GTK3 Installer for Windows](https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases).
    -   *Alternative (via MSYS2):*
        1.  Install MSYS2 from [msys2.org](https://www.msys2.org/).
        2.  Run `pacman -S mingw-w64-x86_64-gtk3` in the MSYS2 terminal.
        3.  Add `C:\msys64\mingw64\bin` to your System PATH environment variable.

3.  **Install Python Dependencies**
    Open Command Prompt (cmd) or PowerShell and run:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Optional: Install & Setup Ollama**
    -   Download the Windows installer from [ollama.com/download/windows](https://ollama.com/download/windows).
    -   Run the installer.
    -   Open PowerShell or Command Prompt and run:
        ```bash
        ollama pull llava:7b
        ```

## How to Run

### Clean Research Training Flow
For strict tumor-vs-dementia retraining, hierarchical evaluation, the single 8-class baseline, and Colab multimodal/LoRA work, use the execution guide:

```bash
python -m src.experiment_pipeline create-manifest
python -m src.experiment_pipeline train --task binary --epochs 30 --batch-size 32 --pretrained
python -m src.experiment_pipeline train --task tumor --epochs 30 --batch-size 32 --pretrained
python -m src.experiment_pipeline train --task dementia --epochs 30 --batch-size 32 --pretrained
python -m src.experiment_pipeline evaluate-hierarchical --split test
python -m src.experiment_pipeline train --task eight_class --epochs 30 --batch-size 32 --pretrained
```

See `docs/ML_EXECUTION_FLOW.md` and `docs/PUBLICATION_NOTES.md` for the full task order, leakage controls, Colab plan, and publication tables. See `docs/MANUSCRIPT_FULL_DRAFT.md` for the current venue-neutral full paper draft. See `docs/DATASET_PROVENANCE.md` for dataset source, attribution, license, and citation notes. See `docs/PUBLICATION_SUBMISSION_STATUS.md` for the current reviewer-facing package check, including primary metrics, robustness metrics, probability evidence, active warnings, and remaining submission tasks. See `docs/GROUNDED_REPORTING.md` for the non-hallucinating report-generation path and current specialist accuracy table.

For cloud execution, dispatch the **ML Experiment Suite** GitHub Actions workflow in `smoke` mode for validation or `full` mode on a CUDA/GPU runner. For a single Google Colab run that trains the full suite and downloads a model/metrics bundle, use `docs/COLAB_FULL_TRAINING.md`. For multimodal LLM benchmarking and LoRA fine-tuning, open `notebooks/multimodal_llm_lora_colab.ipynb` in Google Colab.

### Publication Package Checks

Regenerate reviewer-facing tables, figures, and the submission status gate:

```bash
python3 scripts/build_publication_tables.py
python3 scripts/build_publication_figures.py
python3 scripts/check_publication_package.py
```

The status gate is lightweight and does not retrain models.

### Main GUI Application
Run the main GUI application (using the module flag to ensure imports work correctly):

```bash
python -m src.radiology_report_gui
```

1.  **Patient Info:** Enter Patient Name, ID, and Date of Birth (filled from the header when a DICOM scan is loaded).
2.  **Scan:** Click **"Scan"** to load an MRI image, a DICOM file, a NIfTI volume or a zipped DICOM study, or **"DICOM Folder..."** to load a study folder. For a scan, use the slider to page through axial slices and **"Series..."** to pick another series.
3.  **Exam Details:** Switch to the "Exam Details" tab.
    -   Click **"AI Auto-Detect"** to let the optional AI assistant fill basic acquisition metadata only.
    -   Or click **"Manual Entry"** to fill the details yourself.
4.  **Analyze:** Once details are confirmed, click **"Analyze & Generate Report"**.
5.  **Export:** Review the generated report and click **"Save as PDF"**.
6.  **Whole scan (optional, research):** With a 3D T1 scan loaded, click **"Analyze Volume (AD/CN research model)"**, page through the stacked heatmap with the slider, and use **"Save 3D Map..."** to write the NIfTI volumes.

### DICOM and Volume Analysis CLI

```bash
python -m src.dicom_io /path/to/study                         # list series, most likely 3D T1 first
python -m src.dicom_io /path/to/study --to-nifti scan.nii.gz  # canonical NIfTI + sidecar without identifying fields
python -m src.volume_analysis /path/to/study --output results/  # research AD/CN score + 3D Grad-CAM (NIfTI, montage)
```

The volume analysis needs the research checkpoints (`git lfs pull --include="models/kevin_round2_20261011/*.pt"`). Its output is a research score with documented external-validation failures, not a diagnosis; read [docs/DICOM_VOLUME_ANALYSIS.md](docs/DICOM_VOLUME_ANALYSIS.md) first.

### Grounded Report CLI

Generate a deterministic report from a classifier result:

```bash
python -m src.grounded_report --prediction glioma --confidence 97.2% --model-used "Brain Tumor Model" --format markdown
```

Or use hierarchical inference output:

```bash
python -m src.hierarchical_inference path/to/mri.png > prediction.json
python -m src.grounded_report --prediction-json prediction.json --format html --output grounded_report.html
```

### Performance Visualization
To generate performance graphs and heatmaps for the models:

```bash
python data_visualization/visualize_performance.py
```
This will run a subset of the data through both models and generate `.png` plots in the `data_visualization` folder showing accuracy and confusion matrices.

To run the dedicated Brain Tumor model evaluation (producing detailed metrics and ROC curves):
```bash
python data_visualization/evaluate_tumor_only.py
```
