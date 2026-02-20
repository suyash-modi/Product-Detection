# Retail Auto-Zone — Phase Flow Documentation

This document describes the flow of all 4 phases in the Retail Auto-Zone system.

---

## Phase 1: Product Detection & Price Search

### Flow

1. **Camera Initialization**
   - App starts → Opens webcam (default: camera 0)
   - Loads YOLO OpenVINO model (`yolov8x.xml`)
   - Video feed displays in real-time

2. **Product Detection** (User clicks "DETECT ALL")
   - Captures current frame
   - Runs YOLO inference → Detects objects (products, people, etc.)
   - Filters out "person" class (class_id = 0)
   - Creates zones from product detections:
     - Expands bounding boxes by 15% on all sides (for interaction zones)
     - Applies NMS (Non-Maximum Suppression) to remove duplicates
     - Maps class IDs to product labels (e.g., "cell phone", "clock")
   - Saves zones to `data/zones/gui_zones.json`
   - Extracts product crops → Saves to `data/products/`
   - Updates UI: Shows green rectangles around detected products

3. **Product Analysis** (Optional, via BLIP VQA)
   - For each detected product crop:
     - Uses BLIP VQA to ask: "What brand?", "What color?", "What product type?"
     - Extracts structured info (brand, color, type, features)
   - Controlled by `.env`: `USE_PRODUCT_ANALYZER=1` (default) or `0`

4. **Price Search** (User clicks "SEARCH & GET PRICE")
   - For each zone:
     - Builds search query from product label + BLIP analysis (if available)
     - Searches DuckDuckGo Shopping API
     - Parses results: title, price, source, link
     - Filters unreasonable prices (>500k INR)
   - Updates UI: Shows product title and price above each zone
   - Saves search results back to `gui_zones.json`

5. **Retry/Refinement** (User clicks "RETRY / ADD MISSED")
   - Re-scans current frame with lower confidence threshold (0.35)
   - Removes "ghost" zones (products that moved/disappeared) via color comparison
   - Adds newly detected products that weren't in the original scan
   - Updates zones and saves

### Outputs
- `data/zones/gui_zones.json` – Zone definitions with product info, bboxes, search results
- `data/products/product_*.jpg` – Cropped product images
- UI: Live video with product overlays, prices, and search results

---

## Phase 2: Zone Management

### Flow

1. **Zone Creation** (During Phase 1 detection)
   - Each detected product becomes a "zone"
   - Zone structure:
     ```json
     {
       "product": "cell phone",
       "confidence": 0.84,
       "bbox": [x, y, width, height],
       "avg_color": [R, G, B],
       "crop_path": "path/to/crop.jpg",
       "search_results": [...],
       "zone_id": 0
     }
     ```

2. **Zone Expansion**
   - Original product bbox expanded by 15% on all sides
   - Purpose: Create larger interaction area (customer doesn't need to be exactly on the product)

3. **Zone Persistence**
   - Zones saved to JSON after detection/search
   - Zones can be reloaded from JSON (if needed)
   - Zones cleared when new detection runs ("DETECT ALL")

4. **Zone Visualization**
   - Green rectangle: Zone without active customer
   - Orange rectangle + "LIVE": Zone with customer currently present
   - Product name and price displayed above/below zone

### Outputs
- Zone definitions in `gui_zones.json`
- Visual overlays on live video feed

---

## Phase 3: Live Analytics Dashboard

### Flow

1. **Person Detection** (Runs continuously every ~0.5 seconds)
   - Analytics thread runs in background
   - Grabs current camera frame
   - Runs same YOLO model → Gets all detections
   - Filters for "person" class (class_id = 0) with confidence ≥ 0.15
   - Converts person bounding boxes to image coordinates

2. **Zone Occupancy Detection**
   - For each person box:
     - Checks if person overlaps any zone (expanded by 35% for proximity)
     - Uses centroid check OR intersection area (>5% of person box)
   - Result: Set of zone indices that have at least one person inside

3. **Dwell Time & Interaction Tracking** (`DwellTracker`)
   - **Enter event**: Person detected in zone (wasn't there before)
     - Records enter timestamp
     - Increments interaction count for that zone
   - **Exit event**: Person leaves zone (was there, now gone)
     - Calculates dwell time = exit_time - enter_time
     - Appends dwell time to zone's history
     - Clears enter timestamp
   - **Average dwell**: Mean of all completed dwell times for that zone

4. **Dashboard Updates** (Real-time)
   - Dashboard refreshes every ~0.5 seconds
   - Shows per zone:
     - **Zone name** (product name)
     - **Avg Dwell** (average seconds, or "—" if none)
     - **Interactions** (count of enter events)
     - **Status** ("—" or "LIVE (Xs)" showing current dwell)
   - "LIVE" indicator appears next to "Zone Dashboard" header when any zone is occupied

5. **Visual Feedback**
   - Video feed: Zones turn orange with "LIVE" label when occupied
   - Dashboard: Status column shows "LIVE (Xs)" with orange text for active zones

### Outputs
- Real-time dashboard metrics
- Visual indicators on video feed
- Dwell time and interaction counts per zone

---

## Phase 4: Evidence Recording

### Flow

1. **Recording Start** (Person enters zone)
   - Analytics thread detects transition: zone was empty → now occupied
   - Adds zone to `_zones_recording` set
   - Initializes empty frame buffer for that zone: `_recording_buffers[zone_id] = []`

2. **Frame Capture** (While person is in zone)
   - Main video loop (`update_video()`) runs at ~30 fps
   - For each zone being recorded:
     - Creates annotated frame copy:
       - Draws all zones (green rectangles + product names)
       - Highlights recording zone in orange with "REC" label
       - Adds product name below the recording zone
     - Appends annotated frame to buffer (up to 5 minutes = 9000 frames)

3. **Recording Stop** (Person leaves zone)
   - Analytics thread detects transition: zone was occupied → now empty
   - Removes zone from `_zones_recording`
   - Takes frame buffer copy
   - Clears buffer
   - Spawns background thread to save video

4. **Video Saving** (`_save_evidence_video()`)
   - Creates `data/evidence/` directory if needed
   - Generates filename: `zone<N>_<product>_<timestamp>.mp4`
     - Example: `zone0_cell_phone_20250218_143052.mp4`
   - Uses OpenCV `VideoWriter`:
     - Codec: `mp4v`
     - FPS: 30
     - Resolution: Same as camera frames
   - Writes all buffered frames to MP4 file
   - Releases writer
   - Prints confirmation: `[Evidence] Saved: <path>`

5. **Evidence File Structure**
   - Location: `auto_zone_project/data/evidence/`
   - Format: MP4 video files
   - Content: Full camera frames with zone overlays
   - Duration: From person enter to person exit (max 5 minutes)
   - Naming: `zone<id>_<product>_YYYYMMDD_HHMMSS.mp4`

### Outputs
- MP4 video files in `data/evidence/`
- Each file = one customer interaction (enter → leave)
- Videos include zone overlays showing which zone was recorded

---

## Complete System Flow (All Phases Combined)

```
App Start
  ↓
Load YOLO Model
  ↓
Open Camera
  ↓
Start Analytics Thread (Phase 3)
  ↓
[User clicks "DETECT ALL"]
  ↓
Phase 1: Detect Products → Create Zones
  ↓
[User clicks "SEARCH & GET PRICE"]
  ↓
Phase 1: Search Prices → Update Zones
  ↓
[Live Camera Feed Running]
  ↓
Analytics Thread (every 0.5s):
  ├─ Detect persons in frame
  ├─ Compute zone occupancy
  ├─ Update dwell tracker (Phase 3)
  ├─ Detect enter/exit events
  └─ Start/stop recording (Phase 4)
  ↓
Video Loop (every ~33ms):
  ├─ Capture frame
  ├─ Draw zone overlays
  ├─ Record frames for active zones (Phase 4)
  └─ Update UI
  ↓
[Person enters zone]
  ↓
Phase 3: Increment interaction count, start dwell timer
Phase 4: Start recording frames
  ↓
[Person stays in zone]
  ↓
Phase 3: Update current dwell time
Phase 4: Continue recording frames
  ↓
[Person leaves zone]
  ↓
Phase 3: Calculate dwell time, add to average
Phase 4: Stop recording, save video to disk
  ↓
[Repeat for next interaction]
```

---

## Key Components

- **`ProductDetector`** (`src/core/detector.py`) – YOLO OpenVINO inference
- **`create_zones()`** (`src/core/zone_creator.py`) – Parse detections, expand boxes, create zones
- **`DwellTracker`** (`src/core/zone_analytics.py`) – Track dwell times and interactions per zone
- **`get_person_boxes()`** (`src/core/zone_analytics.py`) – Extract person detections from YOLO output
- **`compute_zone_occupancy()`** (`src/core/zone_analytics.py`) – Determine which zones have people
- **`RetailAIApp`** (`src/apps/simple_ui.py`) – Main UI, orchestrates all phases

---

## Configuration

- **Model**: YOLOv8x (OpenVINO format) – Detects products and people
- **Person confidence**: 0.15 (low threshold for better detection)
- **Zone expansion**: 15% (for interaction zones)
- **Occupancy expansion**: 35% (for "near zone" detection)
- **Recording max length**: 5 minutes (9000 frames at 30 fps)
- **Analytics frequency**: Every 0.5 seconds
- **Video FPS**: ~30 fps (camera dependent)

---

## Data Flow Summary

```
Camera Frame
  ↓
[Main Thread] → Display + Record (if zone active)
  ↓
[Analytics Thread] → Detect persons → Compute occupancy → Update tracker → Start/stop recording
  ↓
Dashboard Updates (real-time)
Evidence Files (on exit)
```
