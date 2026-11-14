"""The DICOM / volume workflow through the real Tk application.

Needs PyTorch, pydicom, nibabel, tkinter *and* a display; run it locally with
``xvfb-run python -m pytest tests/test_gui_scan_flow.py`` on a headless box.
Skipped everywhere else. Uses synthetic scans and randomly initialised
networks only: it pins the workflow, not model quality.
"""

import importlib
import shutil
import sys
import tempfile
import time
import types
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

try:
    import nibabel  # noqa: F401
    import pydicom  # noqa: F401
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None

try:
    import tkinter as tk
    from tkinter import ttk
except ModuleNotFoundError:
    tk = None


def _display_available():
    if tk is None:
        return False
    try:
        root = tk.Tk()
    except Exception:
        return False
    root.destroy()
    return True


RUNNABLE = torch is not None and _display_available()


@unittest.skipUnless(RUNNABLE, "needs PyTorch, pydicom, nibabel, tkinter and a display (use xvfb-run)")
class GuiScanFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import dicom_synth as synth
        except ModuleNotFoundError:
            from tests import dicom_synth as synth
        from test_volume_analysis import head_phantom

        from src import volume_analysis

        # Import the GUI with only its reporting integrations stubbed when absent.
        stubs = {}
        for name, exports in {"ollama": [], "tkhtmlview": ["HTMLLabel"], "tkcalendar": ["DateEntry"],
                              "weasyprint": ["HTML"], "pypdf": ["PdfReader", "PdfWriter"]}.items():
            try:
                importlib.import_module(name)
            except Exception:
                module = types.ModuleType(name)
                for export in exports:
                    setattr(module, export, Mock())
                stubs[name] = module
        sys.modules.update(stubs)
        try:
            cls.gui = importlib.import_module("src.radiology_report_gui")
        finally:
            for name in stubs:
                sys.modules.pop(name, None)

        class FakeDateEntry(ttk.Entry):
            def __init__(self, master=None, **_options):
                super().__init__(master)
                self._date = date(1986, 1, 1)

            def set_date(self, value):
                self._date = value

            def get_date(self):
                return self._date

        class FakeHTMLLabel(ttk.Label):
            def __init__(self, master=None, html="", **_options):
                super().__init__(master)
                self.html = html

            def set_html(self, html):
                self.html = html

        cls.directory = Path(tempfile.mkdtemp(prefix="gui_scan_flow_"))
        cls.addClassCleanup(shutil.rmtree, cls.directory, ignore_errors=True)

        data, affine = head_phantom()
        study = synth.generate_uid()
        sagittal, sagittal_affine = synth.reoriented(data, affine, (1, 2, 0), (True, False, True))
        synth.write_series(cls.directory / "study" / "t1", sagittal, sagittal_affine, study_uid=study, series_number=5, extension="")
        synth.write_series(cls.directory / "study" / "t2", data[:, :, ::4], affine @ np.diag([1, 1, 4, 1]), study_uid=study,
                           series_number=3, tags=synth.T2_TSE)
        (cls.directory / "empty").mkdir()
        cls.plain_image = cls.directory / "slice.png"
        Image.new("RGB", (96, 80), (40, 90, 160)).save(cls.plain_image)

        checkpoints = []
        for seed in (1, 2):
            torch.manual_seed(seed)
            path = cls.directory / f"clinical_seed{seed}.pt"
            torch.save({"model": volume_analysis.build_clinical_slice_model().state_dict(),
                        "architecture": volume_analysis.CLINICAL_ARCHITECTURE, "seed": seed, "epoch": 1}, path)
            checkpoints.append(path)

        cls.dialogs = []
        cls.opened = []
        cls.answer = True  # what the person answers to a yes/no question
        cls.browser = True  # what starting a browser returns; an exception instance is raised instead

        def ask(title, text):
            cls.dialogs.append(("askyesno", title, text))
            return cls.answer

        def launch(url):
            cls.opened.append(url)
            if isinstance(cls.browser, Exception):
                raise cls.browser
            return cls.browser

        missing = str(cls.directory / "no_such_model.pt")
        cls.patches = [
            patch.object(cls.gui, "DateEntry", FakeDateEntry),
            patch.object(cls.gui, "HTMLLabel", FakeHTMLLabel),
            *(patch.object(cls.gui, name, missing) for name in
              ("BINARY_ROUTER_MODEL_PATH", "GATEKEEPER_MODEL_PATH", "TUMOR_MODEL_PATH", "ALZHEIMERS_MODEL_PATH")),
            *(patch.object(cls.gui.messagebox, name, side_effect=lambda title, text, _kind=name: cls.dialogs.append((_kind, title, text)))
              for name in ("showinfo", "showwarning", "showerror")),
            # Questions are answered with cls.answer, and a browser launch is recorded instead of made.
            patch.object(cls.gui.messagebox, "askyesno", side_effect=ask),
            patch.object(cls.gui.webbrowser, "open", side_effect=launch),
        ]
        for item in cls.patches:
            item.start()
            cls.addClassCleanup(item.stop)

        cls.root = tk.Tk()
        cls.root.geometry("1200x880")
        cls.addClassCleanup(cls.root.destroy)
        cls.app = cls.gui.App(cls.root)
        cls.app.volume_analyzer = volume_analysis.DementiaVolumeAnalyzer(checkpoints, device=torch.device("cpu"))
        # A stand-in 4-class stage model so the single-slice path can run without trained weights.
        cls.app.alz_model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 4)).eval()

    def pump(self, condition=lambda: True, timeout=120.0):
        deadline = time.time() + timeout
        while True:
            self.root.update()
            if condition() and str(self.app.choose_img_btn.cget("state")) == "normal":
                break
            if time.time() > deadline:
                self.fail("GUI did not reach the expected state")
            time.sleep(0.02)
        for _ in range(5):
            self.root.update()

    def open_study(self):
        self.dialogs.clear()
        self.app._open_path(str(self.directory / "study"))
        self.pump(lambda: self.app.volume is not None and self.app.volume.shape[2] > 40)

    def test_1_dicom_folder_loads_best_t1_and_fills_the_form_from_the_header(self):
        self.open_study()
        app = self.app
        self.assertEqual(app.volume.shape, (88, 104, 92))
        self.assertEqual(app.volume.acquisition["contrast_guess"]["label"], "T1w")
        self.assertEqual(len(app.volume_series), 2)
        self.assertEqual(self.dialogs, [])  # a T1 series loads without complaint
        self.assertEqual(app.patient_name_entry.get(), "PHANTOM SYNTHETIC")
        self.assertEqual(app.patient_id_entry.get(), "FAKE-ID-0042")
        self.assertEqual(app.dob_entry.get_date(), date(1940, 2, 29))
        self.assertIn("3D T1w", app.technique_entry.get())
        self.assertEqual(app.contrast_entry.get(), "None recorded in image header")
        self.assertEqual(app.slice_frame.winfo_manager(), "pack")
        for button, state in ((app.volume_btn, "normal"), (app.series_btn, "normal"), (app.save_volume_btn, "disabled"),
                              (app.generate_btn, "normal")):
            self.assertEqual(str(button.cget("state")), state)
        self.assertEqual(app.slice_index, 46)
        self.assertEqual(app.scan_slice_image.size, (88, 104))
        self.assertIn("axial slice 47 of 92", app.scan_info_label.cget("text"))
        self.assertNotIn("SYNTHETIC", app.scan_info_label.cget("text"))

        before = np.asarray(app.scan_slice_image).copy()
        app.slice_scale.set(20)
        self.pump(lambda: app.slice_index == 20)
        self.assertFalse(np.array_equal(np.asarray(app.scan_slice_image), before))

    def test_2_single_slice_path_classifies_the_displayed_slice_and_flags_the_stage_model(self):
        self.open_study()
        app = self.app
        seen = []
        handle = app.alz_model.register_forward_pre_hook(lambda _module, inputs: seen.append(inputs[0].clone()))
        try:
            result = app._run_cascaded_classification()
        finally:
            handle.remove()
        self.assertEqual(result.source_image.size, app.scan_slice_image.size)
        torch.testing.assert_close(seen[0], app.alz_tfms(app.scan_slice_image.convert("RGB")).unsqueeze(0), rtol=0, atol=0)
        self.assertIn("Alzheimer", result.model_name)
        self.assertEqual(app._raw_scan_stage_caution(result.model_name), self.gui.RAW_SCAN_STAGE_CAUTION)
        self.assertEqual(app._raw_scan_stage_caution("Brain Tumor Model"), "")
        self.assertIn("single-slice classification", app._source_description())
        self.assertNotIn(str(self.directory), app._source_description())  # no patient-bearing path in the report
        payload, mime = app._current_image_png()
        self.assertEqual(mime, "image/png")
        self.assertEqual(payload[:8], b"\x89PNG\r\n\x1a\n")

    def test_3_volume_analysis_shows_the_stacked_heatmap_and_saves_the_3d_map(self):
        self.open_study()
        app = self.app
        app.on_analyze_volume()
        self.pump(lambda: app.volume_result is not None)
        result = app.volume_result
        self.assertIn("research AD/CN score", app.pred_label.cget("text"))
        self.assertIn("P(AD):", app.confidence_label.cget("text"))
        self.assertIn("not validated", app.model_used_label.cget("text"))
        self.assertIn("not a diagnosis", app.last_report_html)
        self.assertIn("specificity 0/10", app.report_html.html)
        zlo, zhi = result.prepared.cranial_extent
        self.assertEqual(float(app.slice_scale.cget("to")), float(zhi - zlo))
        self.assertIn(zlo + app.slice_index, result.cam_indices)
        self.assertTrue(app.showing_heatmap)
        self.assertIsNotNone(app.heatmap_pil_image)
        self.assertEqual(str(app.save_volume_btn.cget("state")), "normal")
        self.assertEqual(str(app.heatmap_toggle_btn.cget("text")), "Show Original")
        app._toggle_heatmap_view()
        self.assertFalse(app.showing_heatmap)
        self.assertEqual(str(app.heatmap_toggle_btn.cget("text")), "Show Heatmap")
        app._toggle_heatmap_view()

        app.slice_scale.set(0)  # below the trained band: anatomy only, and the caption says so
        self.pump(lambda: app.slice_index == 0)
        self.assertIsNone(app.heatmap_pil_image)
        self.assertIn("outside", app.scan_info_label.cget("text"))

        target = self.directory / "saved"
        with patch.object(self.gui.filedialog, "askdirectory", return_value=str(target)):
            app.on_save_volume_result()
        self.pump(lambda: any(kind == "askyesno" and title == "Saved" for kind, title, _ in self.dialogs))
        for name in ("result.json", "anatomy.nii.gz", "gradcam.nii.gz", "gradcam_native.nii.gz", "montage.png", "projections.png", "viewer.html"):
            self.assertTrue((target / name).is_file(), name)
        self.assertNotIn("SYNTHETIC", (target / "result.json").read_text())
        # Saving offers the interactive viewer; saying yes opens the saved file, which names nobody.
        question = next(text for kind, _, text in self.dialogs if kind == "askyesno")
        self.assertIn("Open the interactive 3D viewer (viewer.html) in your browser?", question)
        self.assertIn("If no browser window appears, open that file from the folder above.", question)
        self.pump(lambda: bool(self.opened))  # the browser is started from a worker thread
        self.assertEqual(self.opened, [(target / "viewer.html").resolve().as_uri()])
        viewer = (target / "viewer.html").read_text()
        self.assertNotIn("SYNTHETIC", viewer)
        self.assertNotIn("FAKE-ID", viewer)

        def save_again(name, dialogs=1):
            """Save into a new folder; returns the folder and the dialogs that the save produced."""

            folder, before = self.directory / name, len(self.dialogs)
            with patch.object(self.gui.filedialog, "askdirectory", return_value=str(folder)):
                app.on_save_volume_result()
            self.pump(lambda: len(self.dialogs) >= before + dialogs and str(app.save_volume_btn.cget("state")) == "normal")
            settled = time.time() + 0.5
            while time.time() < settled:  # anything else that was on its way has arrived
                self.root.update()
                time.sleep(0.02)
            return folder, self.dialogs[before:]

        # Saying no opens nothing and leaves the app ready.
        type(self).answer = False
        folder, shown = save_again("saved_no")
        self.assertEqual([kind for kind, _, _ in shown], ["askyesno"])
        self.assertEqual(len(self.opened), 1)
        self.assertTrue((folder / "viewer.html").is_file())
        self.assertEqual(str(app.volume_btn.cget("state")), "normal")

        # When no browser can be started, the app says where the file is instead of doing nothing.
        type(self).answer = True
        for outcome in (False, self.gui.webbrowser.Error("could not locate runnable browser")):
            type(self).browser = outcome
            folder, shown = save_again(f"saved_browser_{type(outcome).__name__}", dialogs=2)
            self.assertEqual([(kind, title) for kind, title, _ in shown], [("askyesno", "Saved"), ("showinfo", "Viewer")])
            self.assertIn(str((folder / "viewer.html").resolve()), shown[1][2])
        type(self).browser = True

        # When the viewer could not be written there is nothing to offer, and the app says so.
        from src import volume_viewer

        opened = len(self.opened)
        with patch.object(volume_viewer, "build_viewer", side_effect=RuntimeError("no viewer today")):
            with self.assertLogs("src.volume_analysis", level="ERROR"):
                folder, shown = save_again("saved_without_viewer")
        self.assertEqual([(kind, title) for kind, title, _ in shown], [("showinfo", "Saved")])
        self.assertIn("viewer.html) could not be written", shown[0][2])
        self.assertFalse((folder / "viewer.html").exists())
        self.assertTrue((folder / "gradcam.nii.gz").is_file())
        self.assertEqual(len(self.opened), opened)

        # Running the single-slice analysis returns the viewer to the raw scan.
        with patch.object(app, "_start_report_generation"):
            app.on_analyze_and_generate()
        self.assertIsNone(app.volume_result)
        self.assertEqual(float(app.slice_scale.cget("to")), float(app.volume.shape[2] - 1))
        self.assertIn("unreliable", app.heatmap_caption_label.cget("text"))
        self.assertIn("outside training domain", app.model_used_label.cget("text"))
        app.generate_btn.configure(state="normal")
        app.choose_img_btn.configure(state="normal")
        app.progress_bar.stop()

    def test_4_non_t1_series_is_flagged_and_bad_input_is_reported(self):
        self.open_study()
        app = self.app
        t2 = next(series for series in app.volume_series if series.contrast.label == "T2w")
        self.dialogs.clear()
        app._start_scan_load(app.image_path, series_key=t2.key)
        self.pump(lambda: app.volume is not None and app.volume.shape[2] < 40)
        self.assertEqual([(kind, title) for kind, title, _ in self.dialogs], [("showwarning", "Check the series")])
        self.assertIn("T2w", self.dialogs[0][2])

        self.dialogs.clear()
        app._open_path(str(self.directory / "empty"))
        self.pump(lambda: bool(self.dialogs))
        self.assertEqual(self.dialogs[0][:2], ("showerror", "Scan Error"))
        self.assertIn("No DICOM image series", self.dialogs[0][2])

    def test_5_header_values_never_follow_the_user_to_another_image(self):
        self.open_study()
        app = self.app
        self.assertEqual(app.patient_name_entry.get(), "PHANTOM SYNTHETIC")
        app.patient_id_entry.delete(0, tk.END)
        app.patient_id_entry.insert(0, "typed by the user")
        app._open_path(str(self.plain_image))
        self.pump()
        self.assertIsNone(app.volume)
        self.assertIsNone(app.scan_slice_image)
        self.assertEqual(app.patient_name_entry.get(), "")  # header-derived: cleared
        self.assertEqual(app.patient_id_entry.get(), "typed by the user")  # user edit: kept
        self.assertNotEqual(app.dob_entry.get_date(), date(1940, 2, 29))
        self.assertEqual(app.technique_entry.get(), "Standard MRI Brain Protocol")
        self.assertEqual(app.contrast_entry.get(), "None")
        self.assertEqual(app.slice_frame.winfo_manager(), "")
        self.assertEqual(str(app.volume_btn.cget("state")), "disabled")
        self.assertEqual(app._raw_scan_stage_caution("Alzheimer's/Dementia Model"), "")
        self.assertEqual(app._source_description(), str(self.plain_image))
        result = app._run_cascaded_classification()
        self.assertEqual(result.source_image.size, (96, 80))  # read from the file, as before


if __name__ == "__main__":
    unittest.main()
