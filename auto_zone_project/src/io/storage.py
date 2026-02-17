import json
import os

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False


class _ZoneEncoder(json.JSONEncoder):
    """Handle numpy types in zone data."""
    def default(self, obj):
        if _NUMPY and isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if _NUMPY and isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_zones(zones, output_path):
    # Ensure the folder exists (e.g., data/zones)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(zones, f, indent=4, cls=_ZoneEncoder)
    
    # Print confirmation
    print(f"[OK] Saved {len(zones)} zones to: {output_path}")