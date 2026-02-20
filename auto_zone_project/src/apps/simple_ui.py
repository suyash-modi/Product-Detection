import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import re
import sys
import os
import json
import math
import threading
import time
import numpy as np  # Needed for color checking

# Add 'src' to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.io.video import open_video
from src.core.detector import ProductDetector
from src.core.zone_creator import create_zones
from src.core.zone_analytics import (
    get_person_boxes,
    compute_zone_occupancy,
    DwellTracker,
)
from src.io.storage import save_zones
from src.io.extractor import extract_product_crops
from src.services.product_search import search_product_free

# Pipeline: OpenVINO → Crop → BLIP VQA → Build Query → DuckDuckGo → Parse Price

# --- CONFIGURATION ---
import os
_base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.exists(os.path.join(_base, "yolov8x_openvino_model", "yolov8x.xml")):
    MODEL_PATH = os.path.join(_base, "yolov8x_openvino_model", "yolov8x.xml")
else:
    MODEL_PATH = os.path.join(_base, "models", "yolov8x.xml")
LABELS_PATH = os.path.join(_base, "models", "labels.json")
EVIDENCE_DIR = os.path.join(_base, "data", "evidence")
# Max evidence clip length: 5 minutes at ~30 fps
MAX_RECORDING_FRAMES = 5 * 60 * 30  # 9000

# UI theme
_BG_DARK = "#1a1d23"
_BG_PANEL = "#252830"
_BG_CARD = "#2d323d"
_ACCENT_GREEN = "#34c759"
_ACCENT_BLUE = "#0a84ff"
_ACCENT_PURPLE = "#af52de"
_ACCENT_ORANGE = "#ff9f0a"
_TEXT = "#e5e5ea"
_TEXT_DIM = "#8e8e93"

class RetailAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Retail Auto-Zone — Live Dashboard")
        self.root.geometry("1280x780")
        self.root.configure(bg=_BG_DARK)
        self.root.minsize(1000, 600)

        # Main layout: video (left) | panel (right)
        self.video_frame = tk.Frame(root, bg="black")
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack(expand=True)

        self.controls = tk.Frame(root, width=380, bg=_BG_PANEL)
        self.controls.pack(side=tk.RIGHT, fill=tk.Y)
        self.controls.pack_propagate(False)

        # Control header
        header = tk.Frame(self.controls, bg=_BG_PANEL)
        header.pack(fill=tk.X, padx=16, pady=(16, 8))
        tk.Label(header, text="Control Panel", font=("Segoe UI", 18, "bold"), fg=_TEXT, bg=_BG_PANEL).pack(anchor="w")

        self.btn_detect = tk.Button(
            self.controls, text="🔍  DETECT ALL",
            font=("Segoe UI", 11, "bold"), bg=_ACCENT_GREEN, fg="white",
            activebackground="#2da44e", activeforeground="white",
            relief=tk.FLAT, cursor="hand2", height=2,
            command=self.run_full_detection
        )
        self.btn_detect.pack(fill=tk.X, padx=16, pady=6)

        self.btn_retry = tk.Button(
            self.controls, text="🔄  RETRY / ADD MISSED",
            font=("Segoe UI", 11, "bold"), bg=_ACCENT_BLUE, fg="white",
            activebackground="#0066cc", activeforeground="white",
            relief=tk.FLAT, cursor="hand2", height=2,
            command=self.run_retry
        )
        self.btn_retry.pack(fill=tk.X, padx=16, pady=6)

        self.btn_search = tk.Button(
            self.controls, text="💰  SEARCH & GET PRICE",
            font=("Segoe UI", 11, "bold"), bg=_ACCENT_PURPLE, fg="white",
            activebackground="#8944ab", activeforeground="white",
            relief=tk.FLAT, cursor="hand2", height=2,
            command=self.run_search_and_price
        )
        self.btn_search.pack(fill=tk.X, padx=16, pady=6)

        self.lbl_status = tk.Label(
            self.controls, text="Initializing...", fg=_TEXT_DIM, bg=_BG_PANEL,
            font=("Segoe UI", 10)
        )
        self.lbl_status.pack(pady=(4, 12))

        # Results area (scrollable summary)
        results_frame = tk.Frame(self.controls, bg=_BG_PANEL)
        results_frame.pack(fill=tk.BOTH, expand=True)
        tk.Label(results_frame, text="Products & Prices", font=("Segoe UI", 12, "bold"), fg=_TEXT, bg=_BG_PANEL).pack(anchor="w", padx=16)
        self.result_text = tk.Text(
            results_frame, height=10, width=38, font=("Consolas", 9),
            bg=_BG_CARD, fg=_TEXT, insertbackground=_TEXT, relief=tk.FLAT,
            wrap=tk.WORD, padx=10, pady=8
        )
        self.result_text.pack(fill=tk.X, padx=16, pady=(4, 12))

        # Zone Dashboard
        dash_frame = tk.Frame(self.controls, bg=_BG_PANEL)
        dash_frame.pack(fill=tk.BOTH, expand=True)
        dash_header = tk.Frame(dash_frame, bg=_BG_PANEL)
        dash_header.pack(fill=tk.X, padx=16, pady=(0, 6))
        tk.Label(dash_header, text="Zone Dashboard", font=("Segoe UI", 12, "bold"), fg=_TEXT, bg=_BG_PANEL).pack(side=tk.LEFT)
        self.lbl_dashboard_live = tk.Label(dash_header, text="", font=("Segoe UI", 9), fg=_ACCENT_ORANGE, bg=_BG_PANEL)
        self.lbl_dashboard_live.pack(side=tk.RIGHT)

        # Dashboard table: header row + body (rows added in _refresh_dashboard_ui)
        self.dashboard_table = tk.Frame(dash_frame, bg=_BG_CARD, padx=8, pady=8)
        self.dashboard_table.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 16))
        header_row = tk.Frame(self.dashboard_table, bg=_BG_CARD)
        header_row.pack(fill=tk.X, pady=(0, 6))
        for col, (text, w) in enumerate([("Zone", 12), ("Avg Dwell", 8), ("Interactions", 6), ("Status", 12)]):
            lbl = tk.Label(header_row, text=text, font=("Segoe UI", 9, "bold"), fg=_TEXT_DIM, bg=_BG_CARD, width=w, anchor="w")
            lbl.grid(row=0, column=col, sticky="w", padx=(0, 8))
        self.dashboard_rows_container = tk.Frame(self.dashboard_table, bg=_BG_CARD)
        self.dashboard_rows_container.pack(fill=tk.BOTH, expand=True)
        self.dashboard_rows = []  # list of (name_lbl, dwell_lbl, count_lbl, status_lbl) per zone

        # Variables
        self.zones = []
        self.cap = None
        self.model = None
        self.labels = {}
        self.current_frame = None
        self.dwell_tracker = DwellTracker(0)
        self._analytics_running = True
        self._analytics_thread = None
        self._infer_lock = threading.Lock()  # OpenVINO allows only one infer() at a time
        self._last_person_boxes = []  # from analytics thread; drawn on live video
        # Phase 4: evidence recording when person enters/leaves zone
        self._zones_recording = set()
        self._recording_buffers = {}
        self._recording_lock = threading.Lock()

        # Load AI
        self.lbl_status.config(text="Loading AI... (app may freeze briefly)", fg=_ACCENT_ORANGE)
        self.root.update()
        self.load_ai()

        # Start Camera
        try:
            self.cap = open_video(0)
            self.update_video()
        except Exception as e:
            messagebox.showerror("Camera Error", f"Could not open camera: {e}")

        # Start zone analytics thread (person-in-zone, dwell, interactions)
        self._start_analytics_thread()

    def load_ai(self):
        try:
            print("Loading model...")
            self.model = ProductDetector(MODEL_PATH)
            with open(LABELS_PATH) as f:
                self.labels = json.load(f)
            self.lbl_status.config(text="AI ready. System online.", fg=_ACCENT_GREEN)
        except Exception as e:
            self.lbl_status.config(text=f"AI failed: {e}", fg="#ff453a")

    def _start_analytics_thread(self):
        def loop():
            while self._analytics_running:
                time.sleep(0.5)
                if not self.model or not self.zones or self.current_frame is None:
                    continue
                try:
                    with self._infer_lock:
                        raw = self.model.infer(self.current_frame)
                    person_boxes = get_person_boxes(raw, self.current_frame.shape)
                    occupied = compute_zone_occupancy(person_boxes, self.zones)
                    prev_occupied = self.dwell_tracker.get_occupied_now()
                    self.dwell_tracker.update(occupied, time.time())
                    # Phase 4: evidence — who entered vs left
                    entered = occupied - prev_occupied
                    left = prev_occupied - occupied
                    with self._recording_lock:
                        for i in entered:
                            self._zones_recording.add(i)
                            self._recording_buffers[i] = []
                        for i in left:
                            frames_to_save = list(self._recording_buffers.get(i, []))
                            self._recording_buffers.pop(i, None)
                            self._zones_recording.discard(i)
                            if frames_to_save and i < len(self.zones):
                                z = self.zones[i]
                                product_name = (z.get("product") or "zone").replace(" ", "_")[:20]
                                threading.Thread(
                                    target=self._save_evidence_video,
                                    args=(frames_to_save, i, product_name),
                                    daemon=True,
                                ).start()
                    self.root.after(0, self._refresh_dashboard_ui)
                except Exception:
                    pass

        self._analytics_thread = threading.Thread(target=loop, daemon=True)
        self._analytics_thread.start()

    def _save_evidence_video(self, frames, zone_id, product_name):
        """Save recorded frames to data/evidence/ as MP4 (runs in background thread)."""
        if not frames:
            return
        os.makedirs(EVIDENCE_DIR, exist_ok=True)
        h, w = frames[0].shape[:2]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_name = re.sub(r"[^\w\-]", "", product_name)
        filename = f"zone{zone_id}_{safe_name}_{timestamp}.mp4"
        path = os.path.join(EVIDENCE_DIR, filename)
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(path, fourcc, 30.0, (w, h))
            for f in frames:
                out.write(f)
            out.release()
            print(f"[Evidence] Saved: {path}")
        except Exception as e:
            print(f"[Evidence] Failed to save {path}: {e}")

    def _refresh_dashboard_ui(self):
        """Update dashboard table and live indicator from dwell_tracker + zones."""
        stats = self.dwell_tracker.get_stats(time.time())
        occupied_now = self.dwell_tracker.get_occupied_now()
        n = len(self.zones)
        if occupied_now:
            self.lbl_dashboard_live.config(text="LIVE")
        else:
            self.lbl_dashboard_live.config(text="")

        # Rebuild rows to match zone count
        for w in self.dashboard_rows_container.winfo_children():
            w.destroy()
        self.dashboard_rows.clear()

        if n == 0:
            tk.Label(
                self.dashboard_rows_container,
                text="Run DETECT ALL to see zone analytics.\nWhen someone enters a zone, dwell time and interactions update here.",
                font=("Segoe UI", 9), fg=_TEXT_DIM, bg=_BG_CARD, justify=tk.LEFT
            ).pack(anchor="w", pady=8)
            return

        for i in range(n):
            row_frame = tk.Frame(self.dashboard_rows_container, bg=_BG_CARD)
            row_frame.pack(fill=tk.X, pady=2)
            z = self.zones[i] if i < len(self.zones) else {}
            s = stats[i] if i < len(stats) else {"avg_dwell_seconds": 0, "interaction_count": 0, "is_occupied": False, "current_dwell_seconds": 0}
            name = (z.get("product") or "Zone").replace(" ", "\n")[:14]
            dwell_str = f"{s['avg_dwell_seconds']}s" if s["avg_dwell_seconds"] else "—"
            count_str = str(s["interaction_count"])
            status_str = f"LIVE ({s.get('current_dwell_seconds', 0)}s)" if s["is_occupied"] else "—"
            status_fg = _ACCENT_ORANGE if s["is_occupied"] else _TEXT_DIM
            name_lbl = tk.Label(row_frame, text=name, font=("Segoe UI", 9), fg=_TEXT, bg=_BG_CARD, anchor="w", width=12)
            dwell_lbl = tk.Label(row_frame, text=dwell_str, font=("Segoe UI", 9), fg=_TEXT, bg=_BG_CARD, anchor="w", width=8)
            count_lbl = tk.Label(row_frame, text=count_str, font=("Segoe UI", 9), fg=_TEXT, bg=_BG_CARD, anchor="w", width=6)
            status_lbl = tk.Label(row_frame, text=status_str, font=("Segoe UI", 9, "bold"), fg=status_fg, bg=_BG_CARD, anchor="w", width=12)
            name_lbl.grid(row=0, column=0, sticky="w", padx=(0, 8))
            dwell_lbl.grid(row=0, column=1, sticky="w", padx=(0, 8))
            count_lbl.grid(row=0, column=2, sticky="w", padx=(0, 8))
            status_lbl.grid(row=0, column=3, sticky="w", padx=(0, 8))
            self.dashboard_rows.append((name_lbl, dwell_lbl, count_lbl, status_lbl))

    def update_video(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                # Phase 4: record evidence with zone overlay drawn on each frame
                with self._recording_lock:
                    for i in list(self._zones_recording):
                        buf = self._recording_buffers.get(i)
                        if buf is not None and len(buf) < MAX_RECORDING_FRAMES:
                            rec_frame = frame.copy()
                            for j, z in enumerate(self.zones):
                                x, y, w, h = z['bbox']
                                if j == i:
                                    cv2.rectangle(rec_frame, (x, y), (x + w, y + h), (0, 165, 255), 3)
                                    cv2.putText(rec_frame, "REC", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                                    cv2.putText(rec_frame, (z.get('product') or 'Zone'), (x, y + h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                                else:
                                    cv2.rectangle(rec_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                    cv2.putText(rec_frame, (z.get('product') or 'Zone'), (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            buf.append(rec_frame)
                display_img = frame.copy()
                occupied_now = self.dwell_tracker.get_occupied_now() if self.zones else set()

                for i, z in enumerate(self.zones):
                    x, y, w, h = z['bbox']
                    if i in occupied_now:
                        cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 165, 255), 3)  # orange
                        cv2.putText(display_img, "LIVE", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    else:
                        cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    results = z.get('search_results') or []
                    if results:
                        # Use result with highest *reasonable* price (ignore bogus parses like 2264000)
                        def _price_val(res):
                            raw = (res.get('price') or '').strip()
                            digits = re.sub(r'[^\d.]', '', re.sub(r'[?\uFF1F\uFFFD\u00A0]+', '', raw))
                            try:
                                return float(digits) if digits else 0.0
                            except ValueError:
                                return 0.0
                        _MAX_REASONABLE_INR = 500_000  # ignore parses above this (e.g. 2264000)
                        sane = [res for res in results if _price_val(res) <= _MAX_REASONABLE_INR]
                        r = max(sane, key=_price_val) if sane else min(results, key=_price_val)
                        # Line 1: short product title; strip "Buy " and site prefixes
                        title = (r.get('title') or '').strip()
                        # If brand/search is unclear, don't change the original detection label for this zone.
                        if title.strip().lower() in ("not a clear image", "not clearly visible"):
                            cv2.putText(display_img, z['product'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            continue
                        for prefix in ('Buy now ', 'Buy Now ', 'Buy ', 'Amazon.com: ', 'Amazon.in: ', 'Amazon: ', 'Flipkart: ', 'Myntra: '):
                            if title.lower().startswith(prefix.lower()):
                                title = title[len(prefix):].strip()
                                break
                        # Title only till first dash (source shown below); strip any ??? and trailing price from title
                        if ' - ' in title:
                            title = title.split(' - ', 1)[0].strip()
                        title = re.sub(r'\?+', '', title)  # remove all ? so title never shows ????
                        title = re.sub(r'\s*[\d,]+\.?\d*\s*$', '', title).strip()  # remove trailing price from title
                        line1 = (title[:25] + '..') if len(title) > 25 else title if title else z['product']
                        line1 = line1.strip() or z['product']
                        # Line 2: use "Rs." (OpenCV putText can't render ₹, shows ???)
                        raw_price = (r.get('price') or '').strip()
                        digits_only = re.sub(r'[^\d.]', '', raw_price)
                        # Avoid leading '.' from patterns like "Rs. 20000"
                        digits_only = digits_only.lstrip('.')
                        price_display = digits_only if digits_only else ''
                        source = (r.get('source') or '').strip()
                        if source.lower() == 'duckduckgo':
                            source = 'Shop'
                        source = source or 'Shop'
                        if price_display:
                            line2 = f"Rs. {price_display}  {source}"
                        else:
                            # Make it clear we tried but found no price
                            line2 = f"No price found  {source}"
                        cv2.putText(display_img, line1, (x, y - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(display_img, line2, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                    else:
                        cv2.putText(display_img, z['product'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                h, w, _ = rgb.shape
                if w > 800:
                    scale = 800/w
                    rgb = cv2.resize(rgb, (800, int(h*scale)))
                
                img = Image.fromarray(rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

        self.root.after(30, self.update_video) 

    def _set_buttons_state(self, state):
        """Enable or disable action buttons (normal/disabled)."""
        for btn in (self.btn_detect, self.btn_retry, self.btn_search):
            btn.config(state=state)

    def run_full_detection(self):
        """Wipes everything and starts fresh - runs in background so video keeps streaming."""
        if not self.model: return
        if self.current_frame is None: return
        frame = self.current_frame.copy()
        self._set_buttons_state("disabled")
        self.lbl_status.config(text="Scanning... (video continues)", fg=_ACCENT_BLUE)

        def worker():
            try:
                with self._infer_lock:
                    raw = self.model.infer(frame)
                zones = create_zones(raw, frame.shape, self.labels, conf_thresh=0.50)
                for z in zones:
                    if 'avg_color' in z: continue
                    x, y, w, h = z['bbox']
                    x, y = max(0, x), max(0, y)
                    roi = frame[y:y+h, x:x+w]
                    if roi.size > 0:
                        z['avg_color'] = np.mean(roi, axis=(0, 1)).tolist()
                products_dir = os.path.join(_base, "data", "products")
                extract_product_crops(frame, zones, products_dir)
                save_zones(zones, os.path.join(_base, "data", "zones", "gui_zones.json"))
                self.root.after(0, lambda: self._on_detect_done(zones))
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda err=err: self._on_detect_error(err))

        threading.Thread(target=worker, daemon=True).start()

    def _on_detect_done(self, zones):
        for i, z in enumerate(zones):
            z["zone_id"] = i
        self.zones = zones
        self.dwell_tracker = DwellTracker(len(zones))
        with self._recording_lock:
            self._zones_recording.clear()
            self._recording_buffers.clear()
        self.refresh_summary()
        self._refresh_dashboard_ui()
        self.lbl_status.config(text=f"Detected {len(self.zones)} products.", fg=_ACCENT_GREEN)
        self._set_buttons_state("normal")

    def _on_detect_error(self, err):
        self.lbl_status.config(text=f"Error: {err}", fg="#ff453a")
        self._set_buttons_state("normal")

    def run_retry(self):
        """Adds missed items AND removes Ghosts - runs in background."""
        if not self.model: return
        if self.current_frame is None: return
        frame = self.current_frame.copy()
        zones_copy = list(self.zones)
        self._set_buttons_state("disabled")
        self.lbl_status.config(text="Refinement scan... (video continues)", fg=_ACCENT_BLUE)

        def worker():
            try:
                valid_zones = []
                removed_count = 0
                for z in zones_copy:
                    x, y, w, h = z['bbox']
                    if 'avg_color' not in z:
                        valid_zones.append(z)
                        continue
                    roi = frame[y:y+h, x:x+w]
                    if roi.size == 0:
                        removed_count += 1
                        continue
                    current_avg = np.mean(roi, axis=(0, 1))
                    saved_avg = np.array(z['avg_color'])
                    if np.linalg.norm(current_avg - saved_avg) > 45:
                        removed_count += 1
                    else:
                        valid_zones.append(z)

                with self._infer_lock:
                    raw = self.model.infer(frame)
                candidates = create_zones(raw, frame.shape, self.labels, conf_thresh=0.35)
                added_count = 0
                for cand in candidates:
                    cx_new = cand['bbox'][0] + cand['bbox'][2]/2
                    cy_new = cand['bbox'][1] + cand['bbox'][3]/2
                    is_new = True
                    for ex in valid_zones:
                        cx_old = ex['bbox'][0] + ex['bbox'][2]/2
                        cy_old = ex['bbox'][1] + ex['bbox'][3]/2
                        if math.sqrt((cx_new - cx_old)**2 + (cy_new - cy_old)**2) < 60:
                            is_new = False
                            break
                    if is_new:
                        valid_zones.append(cand)
                        added_count += 1

                for z in valid_zones:
                    if 'avg_color' in z: continue
                    x, y, w, h = z['bbox']
                    roi = frame[y:y+h, x:x+w]
                    if roi.size > 0:
                        z['avg_color'] = np.mean(roi, axis=(0, 1)).tolist()
                products_dir = os.path.join(_base, "data", "products")
                extract_product_crops(frame, valid_zones, products_dir)
                save_zones(valid_zones, os.path.join(_base, "data", "zones", "gui_zones.json"))
                msg = ""
                if added_count > 0: msg += f"Added {added_count} new. "
                if removed_count > 0: msg += f"Removed {removed_count} moved. "
                if not msg: msg = "No changes found."
                self.root.after(0, lambda: self._on_retry_done(valid_zones, msg))
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda err=err: self._on_retry_error(err))

        threading.Thread(target=worker, daemon=True).start()

    def _on_retry_done(self, zones, msg):
        for i, z in enumerate(zones):
            z["zone_id"] = i
        self.zones = zones
        self.dwell_tracker.resize(len(zones))
        with self._recording_lock:
            self._zones_recording.clear()
            self._recording_buffers.clear()
        self.refresh_summary()
        self._refresh_dashboard_ui()
        self.lbl_status.config(text=msg, fg=_ACCENT_GREEN)
        self._set_buttons_state("normal")

    def _on_retry_error(self, err):
        self.lbl_status.config(text=f"Error: {err}", fg="#ff453a")
        self._set_buttons_state("normal")

    def run_search_and_price(self):
        """Search each detected product - runs in background, video keeps streaming."""
        if not self.zones:
            messagebox.showinfo("No Products", "Detect products first (DETECT ALL or RETRY).")
            return
        zones_ref = self.zones
        self._set_buttons_state("disabled")
        self.lbl_status.config(text="Searching prices... (video continues)", fg=_ACCENT_BLUE)

        def worker():
            try:
                seen_products = set()
                got_api_results = False
                for z in zones_ref:
                    product = z['product']
                    if product in seen_products:
                        donors = [r for r in zones_ref if r.get('product') == product and r.get('search_results')]
                        if donors:
                            z['search_results'] = donors[0].get('search_results', [])
                        continue
                    seen_products.add(product)
                    crop_path = z.get("crop_path")
                    results = search_product_free(crop_path, product)
                    if results:
                        z['search_results'] = results
                        got_api_results = True
                    else:
                        z['search_results'] = []
                for z in zones_ref:
                    if not z.get('search_results'):
                        for other in zones_ref:
                            if other.get('product') == z['product'] and other.get('search_results'):
                                z['search_results'] = other['search_results']
                                break
                save_zones(zones_ref, os.path.join(_base, "data", "zones", "gui_zones.json"))
                # Only say "Prices found." when at least one result has a price
                any_price = any(
                    (r.get("price") or "").strip()
                    for z in zones_ref for r in (z.get("search_results") or [])
                )
                status = "Prices found." if any_price else ("Done. (No prices in results.)" if got_api_results else "Done.")
                self.root.after(0, lambda: self._on_search_done(status))
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda err=err: self._on_search_error(err))

        threading.Thread(target=worker, daemon=True).start()

    def _on_search_done(self, status):
        self.refresh_summary()
        self.lbl_status.config(text=status, fg=_ACCENT_GREEN)
        self._set_buttons_state("normal")

    def _on_search_error(self, err):
        self.lbl_status.config(text=f"Error: {err}", fg="#ff453a")
        self._set_buttons_state("normal")

    def refresh_summary(self):
        self.result_text.delete(1.0, tk.END)
        counts = {}
        search_by_product = {}  # product -> first zone with search_results
        for z in self.zones:
            p = z['product']
            counts[p] = counts.get(p, 0) + 1
            if z.get('search_results') and p not in search_by_product:
                search_by_product[p] = z

        text = f"TOTAL ZONES: {len(self.zones)}\n" + "-"*20 + "\n"
        for p, c in counts.items():
            text += f"{p.upper()}: {c}\n"
        text += "\n" + "-"*20 + "\n"

        # Show search results: Source | Title | Price (never show duckduckgo)
        for product in counts:
            text += f"\n{product.upper()}:\n"
            z = search_by_product.get(product)
            if z and z.get('search_results'):
                # Show highest *reasonable* price first (ignore bogus parses > 500k INR)
                def _panel_price_val(res):
                    raw = (res.get('price') or '').strip()
                    digits = re.sub(r'[^\d.]', '', re.sub(r'[?\uFF1F\uFFFD\u00A0]+', '', raw))
                    try:
                        return float(digits) if digits else 0.0
                    except ValueError:
                        return 0.0
                _max_inr = 500_000
                def _sort_key(res):
                    v = _panel_price_val(res)
                    return (0 if v <= _max_inr else 1, -v)  # sane first, then by highest
                sorted_results = sorted(z['search_results'], key=_sort_key)
                for r in sorted_results[:3]:
                    title = (r.get('title') or '').strip()
                    for prefix in ('Buy now ', 'Buy Now ', 'Buy ', 'Amazon.com: ', 'Amazon.in: ', 'Amazon: ', 'Flipkart: ', 'Myntra: '):
                        if title.lower().startswith(prefix.lower()):
                            title = title[len(prefix):].strip()
                            break
                    if ' - ' in title:
                        title = title.split(' - ', 1)[0].strip()
                    source = (r.get('source') or '').strip()
                    if source.lower() == 'duckduckgo':
                        source = 'Shop'
                    source = source or 'Shop'
                    raw = (r.get('price') or '').strip()
                    price = re.sub(r'[^\d.,\sRs₹INR]', '', raw, flags=re.I).strip() or "—"  # drop ??? and junk
                    if not re.search(r'\d', price):
                        price = "—"
                    text += f"  Source: {source}  |  Title: {title}\n"
                    text += f"  Price: {price}\n"
            else:
                text += "  [Click SEARCH & GET PRICE]\n"

        self.result_text.insert(tk.END, text)

if __name__ == "__main__":
    root = tk.Tk()
    app = RetailAIApp(root)
    root.mainloop()