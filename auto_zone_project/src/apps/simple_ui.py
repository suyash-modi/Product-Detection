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
import numpy as np # Needed for color checking

# Add 'src' to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.io.video import open_video
from src.core.detector import ProductDetector
from src.core.zone_creator import create_zones
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

class RetailAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Retail Auto-Zone (Production PoC)")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f0f0")

        # Layout Setup
        self.video_frame = tk.Frame(root, bg="black")
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack(expand=True)

        self.controls = tk.Frame(root, width=350, bg="white")
        self.controls.pack(side=tk.RIGHT, fill=tk.Y)
        self.controls.pack_propagate(False)

        tk.Label(self.controls, text="Control Panel", font=("Segoe UI", 16, "bold"), bg="white").pack(pady=20)
        
        self.btn_detect = tk.Button(self.controls, text="🔍 DETECT ALL", 
                                    font=("Segoe UI", 12, "bold"), bg="#28a745", fg="white",
                                    height=2, command=self.run_full_detection)
        self.btn_detect.pack(fill=tk.X, padx=20, pady=10)

        self.btn_retry = tk.Button(self.controls, text="RETRY / ADD MISSED", 
                                   font=("Segoe UI", 12, "bold"), bg="#007bff", fg="white",
                                   height=2, command=self.run_retry)
        self.btn_retry.pack(fill=tk.X, padx=20, pady=10)

        self.btn_search = tk.Button(self.controls, text="SEARCH & GET PRICE", 
                                    font=("Segoe UI", 12, "bold"), bg="#6f42c1", fg="white",
                                    height=2, command=self.run_search_and_price)
        self.btn_search.pack(fill=tk.X, padx=20, pady=10)

        self.lbl_status = tk.Label(self.controls, text="Initializing...", fg="gray", bg="white", font=("Segoe UI", 10))
        self.lbl_status.pack(pady=10)

        self.result_text = tk.Text(self.controls, height=18, width=35, font=("Consolas", 9), bg="#f8f9fa", relief=tk.FLAT, wrap=tk.WORD)
        self.result_text.pack(padx=20)

        # Variables
        self.zones = [] 
        self.cap = None
        self.model = None
        self.labels = {}
        self.current_frame = None

        # Load AI
        self.lbl_status.config(text="⏳ Loading AI... (App will freeze for 10s)", fg="orange")
        self.root.update() 
        self.load_ai()

        # Start Camera
        try:
            self.cap = open_video(0)
            self.update_video()
        except Exception as e:
            messagebox.showerror("Camera Error", f"Could not open camera: {e}")

    def load_ai(self):
        try:
            print("⏳ Loading Model...")
            self.model = ProductDetector(MODEL_PATH)
            with open(LABELS_PATH) as f:
                self.labels = json.load(f)
            self.lbl_status.config(text="✅ AI Ready. System Online.", fg="green")
        except Exception as e:
            self.lbl_status.config(text=f"❌ AI Failed: {e}", fg="red")

    def update_video(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                display_img = frame.copy()
                
                for z in self.zones:
                    x, y, w, h = z['bbox']
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
        self.lbl_status.config(text="Scanning... (video continues)", fg="blue")

        def worker():
            try:
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
        self.zones = zones
        self.refresh_summary()
        self.lbl_status.config(text=f"Detected {len(self.zones)} products.", fg="green")
        self._set_buttons_state("normal")

    def _on_detect_error(self, err):
        self.lbl_status.config(text=f"Error: {err}", fg="red")
        self._set_buttons_state("normal")

    def run_retry(self):
        """Adds missed items AND removes Ghosts - runs in background."""
        if not self.model: return
        if self.current_frame is None: return
        frame = self.current_frame.copy()
        zones_copy = list(self.zones)
        self._set_buttons_state("disabled")
        self.lbl_status.config(text="Refinement scan... (video continues)", fg="blue")

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
        self.zones = zones
        self.refresh_summary()
        self.lbl_status.config(text=msg, fg="green")
        self._set_buttons_state("normal")

    def _on_retry_error(self, err):
        self.lbl_status.config(text=f"Error: {err}", fg="red")
        self._set_buttons_state("normal")

    def run_search_and_price(self):
        """Search each detected product - runs in background, video keeps streaming."""
        if not self.zones:
            messagebox.showinfo("No Products", "Detect products first (DETECT ALL or RETRY).")
            return
        zones_ref = self.zones
        self._set_buttons_state("disabled")
        self.lbl_status.config(text="Searching prices... (video continues)", fg="blue")

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
        self.lbl_status.config(text=status, fg="green")
        self._set_buttons_state("normal")

    def _on_search_error(self, err):
        self.lbl_status.config(text=f"Error: {err}", fg="red")
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