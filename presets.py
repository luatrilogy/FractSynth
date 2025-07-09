import json
import os

presets = {
    "Drifting": {"r": 3.5, "gain": 0.2, "base_freq": 220, "type": "Logistic"},
    "Chaotic Glass": {"r": 3.99, "gain": 0.2, "base_freq": 440, "type": "Henon"},
    "Pitch Lava": {"r": 3.82, "gain": 0.2, "base_freq": 300, "type": "Lorenz"},
    "WarmPad": {"r": 3.7, "gain": 0.4, "base_freq": 220, "type": "Henon"},
    "GlassBell": {"r": 3.9, "gain": 0.2, "base_freq": 440, "type": "Duffing"},
}

def save_presets(path="presets.json"):
    with open(path, "w") as f:
        json.dump(presets, f, indent=2)

def load_presets(path="presets.json"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load()
    return {}