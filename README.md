# Product Detection (Retail Auto-Zone)

Retail analytics app with live camera: detect products, track customer interactions, and record evidence automatically.

## Features

- **Phase 1: Product Detection** – Detect products in real-time using YOLO/OpenVINO, analyze with BLIP VQA, and search for prices online.
- **Phase 2: Zone Management** – Create interaction zones around detected products with expanded bounding boxes for better tracking.
- **Phase 3: Live Analytics Dashboard** – Track per-zone metrics: average dwell time, interaction counts, and live occupancy status. Updates in real-time when customers enter/leave zones.
- **Phase 4: Evidence Recording** – Automatically record video clips (up to 5 minutes) when customers interact with zones. Saved to `data/evidence/` with zone overlays included.

See [FLOW.md](FLOW.md) for detailed flow of all phases.

## Prerequisites

- **Python 3.10+**
- **Webcam** (used as default video source)
- **~2GB+ RAM** for models (YOLO OpenVINO + optional BLIP)

## How to run

### 1. Go to the project folder

```bash
cd "c:\Python Projects\Product Detection\auto_zone_project"
```

(Or from repo root: `cd auto_zone_project`)

### 2. Create and activate a virtual environment (recommended)

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Export YOLO to OpenVINO

If you don’t already have the OpenVINO model (e.g. `models/yolov8x.xml` or `yolov8x_openvino_model/yolov8x.xml`), export it once:

```bash
python export_model.py
```

This downloads `yolov8x.pt` and exports it to OpenVINO. The app looks for `yolov8x_openvino_model/yolov8x.xml` first, then `models/yolov8x.xml`.

### 5. (Optional) Environment variables

Create a `.env` in the **workspace root** (`Product Detection`) or inside `auto_zone_project` to tune behavior:

- `USE_CLIP=1` – use CLIP (if supported); `0` to disable.
- `USE_PRODUCT_ANALYZER=1` – use BLIP VQA for brand/type/color; `0` to use YOLO labels only for search.

If you use `.env`, install:

```bash
pip install python-dotenv
```

### 6. Start the app

From **inside** `auto_zone_project`:

```bash
python src/apps/simple_ui.py
```

Or:

```bash
python -m src.apps.simple_ui
```

The Tkinter window opens with:
- **Live camera feed** (left) – Shows detected zones, prices, and live occupancy indicators
- **Control panel** (right) – Buttons for detection, search, and zone management
- **Zone Dashboard** – Real-time metrics: dwell time, interactions, and live status per zone

**Evidence clips** are automatically saved to `auto_zone_project/data/evidence/` when customers leave zones. Each clip includes zone overlays showing which zone was recorded.

## Quick reference

| Step              | Command / action                          |
|-------------------|-------------------------------------------|
| Enter project     | `cd auto_zone_project`                    |
| Create venv       | `python -m venv venv`                     |
| Activate venv     | `.\venv\Scripts\Activate.ps1` (Windows)  |
| Install deps      | `pip install -r requirements.txt`        |
| Export model      | `python export_model.py` (once, if needed)|
| Run app           | `python src/apps/simple_ui.py`            |

## Troubleshooting

- **“Could not open camera”** – Another app may be using the webcam, or no camera is detected. Try closing other camera apps.
- **“AI Failed” / model not found** – Run `python export_model.py` from `auto_zone_project` so `yolov8x.xml` exists under `models/` or `yolov8x_openvino_model/`.
- **Import errors** – Run the app from `auto_zone_project` (e.g. `python src/apps/simple_ui.py`), not from the repo root or from `src/apps/`.
