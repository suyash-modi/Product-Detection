"""
Zone analytics: person-in-zone detection, dwell time, and interaction counts.

- Parses YOLO raw output for person (class 0) boxes.
- Computes which zones are occupied (person bbox overlaps zone).
- Tracks per-zone: average dwell time, number of interactions, and live occupancy.
"""
import time
import numpy as np


def get_person_boxes(raw_output, original_shape, conf_thresh=0.15):
    """
    Extract person (class_id=0) bounding boxes from YOLO raw output.
    Returns list of [x, y, w, h] in original image coordinates.
    Uses low threshold (0.15) so distant/partial people still count.
    """
    detections = raw_output[0].transpose()
    h_orig, w_orig = original_shape[:2]
    x_factor = w_orig / 1280
    y_factor = h_orig / 1280
    person_boxes = []

    for row in detections:
        classes_scores = row[4:]
        class_id = np.argmax(classes_scores)
        if class_id != 0:
            continue
        score = float(classes_scores[0])
        if score < conf_thresh:
            continue
        cx, cy, w, h = row[0], row[1], row[2], row[3]
        left = int((cx - w / 2) * x_factor)
        top = int((cy - h / 2) * y_factor)
        width = int(w * x_factor)
        height = int(h * y_factor)
        left = max(0, left)
        top = max(0, top)
        width = min(w_orig - left, width)
        height = min(h_orig - top, height)
        if width > 0 and height > 0:
            person_boxes.append([left, top, width, height])

    return person_boxes


def _expand_bbox(bbox, expand_ratio=0.35):
    """Expand bbox by ratio on all sides so 'near' the zone counts as in zone. Returns [x, y, w, h]."""
    x, y, w, h = bbox
    pad_w = w * expand_ratio
    pad_h = h * expand_ratio
    return [x - pad_w, y - pad_h, w + 2 * pad_w, h + 2 * pad_h]


def bbox_overlap(person_box, zone_bbox, expand_zone=True):
    """
    True if person is in or near zone. If expand_zone, we expand the zone so
    standing near the product counts as in zone.
    """
    x1, y1, w1, h1 = person_box
    if expand_zone:
        x2, y2, w2, h2 = _expand_bbox(zone_bbox, 0.35)
    else:
        x2, y2, w2, h2 = zone_bbox
    # Person centroid inside (expanded) zone
    cx = x1 + w1 / 2
    cy = y1 + h1 / 2
    if x2 <= cx <= x2 + w2 and y2 <= cy <= y2 + h2:
        return True
    # Any overlap (person and zone intersect)
    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1 + w1, x2 + w2)
    iy2 = min(y1 + h1, y2 + h2)
    if ix1 < ix2 and iy1 < iy2:
        area_person = w1 * h1
        inter = (ix2 - ix1) * (iy2 - iy1)
        if area_person > 0 and inter / area_person > 0.05:
            return True
    return False


def compute_zone_occupancy(person_boxes, zones):
    """
    Returns set of zone indices (0-based) that have at least one person inside.
    """
    occupied = set()
    for i, z in enumerate(zones):
        bbox = z.get("bbox")
        if not bbox:
            continue
        for pb in person_boxes:
            if bbox_overlap(pb, bbox, expand_zone=True):
                occupied.add(i)
                break
    return occupied


class DwellTracker:
    """
    Tracks per-zone: dwell times (list of seconds), interaction count, and current occupancy.
    Call update(occupied_zone_indices, current_time) each tick.
    """

    def __init__(self, num_zones):
        self.num_zones = num_zones
        self.dwell_times = [[] for _ in range(num_zones)]  # list of float (seconds) per zone
        self.interaction_count = [0] * num_zones
        self.enter_time = [None] * num_zones  # when person entered this zone
        self._last_occupied = set()

    def resize(self, num_zones):
        if num_zones == self.num_zones:
            return
        self.dwell_times = self.dwell_times[:num_zones] + [[] for _ in range(num_zones - self.num_zones)]
        self.interaction_count = self.interaction_count[:num_zones] + [0] * (num_zones - len(self.interaction_count))
        self.enter_time = self.enter_time[:num_zones] + [None] * (num_zones - len(self.enter_time))
        self.num_zones = num_zones
        self._last_occupied = set()

    def update(self, occupied_zone_indices, current_time=None):
        if current_time is None:
            current_time = time.time()
        occupied = set(occupied_zone_indices) if occupied_zone_indices is not None else set()
        # Clamp to valid zone indices
        occupied = {i for i in occupied if 0 <= i < self.num_zones}

        for i in range(self.num_zones):
            if i in occupied:
                if self.enter_time[i] is None:
                    self.enter_time[i] = current_time
                    self.interaction_count[i] += 1
            else:
                if self.enter_time[i] is not None:
                    dwell = current_time - self.enter_time[i]
                    if dwell > 0.1:
                        self.dwell_times[i].append(dwell)
                    self.enter_time[i] = None

        self._last_occupied = occupied

    def get_stats(self, current_time=None):
        """Returns list of dicts: {avg_dwell_seconds, interaction_count, is_occupied, current_dwell} per zone."""
        if current_time is None:
            current_time = time.time()
        result = []
        for i in range(self.num_zones):
            times = self.dwell_times[i]
            avg = sum(times) / len(times) if times else 0.0
            is_occ = self.enter_time[i] is not None
            current_dwell = (current_time - self.enter_time[i]) if is_occ and self.enter_time[i] else 0.0
            result.append({
                "avg_dwell_seconds": round(avg, 1),
                "interaction_count": self.interaction_count[i],
                "is_occupied": is_occ,
                "current_dwell_seconds": round(current_dwell, 1) if is_occ else 0.0,
            })
        return result

    def get_occupied_now(self):
        return set(self._last_occupied)
