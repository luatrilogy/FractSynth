import json
import os

presets = {
    "preset1": {"r": 3.5, "gain": 0.2, "base_freq": 220, "type": "Logistic"},
    "preset2": {"r": 3.99, "gain": 0.2, "base_freq": 440, "type": "Henon"},
    "preset3": {"r": 3.82, "gain": 0.2, "base_freq": 300, "type": "Lorenz"},
    "preset4": {"r": 3.7, "gain": 0.4, "base_freq": 220, "type": "Henon"},
    "preset5": {"r": 3.9, "gain": 0.2, "base_freq": 440, "type": "Duffing"},
}

def save_presets(path="presets.json"):
    with open(path, "w") as f:
        json.dump(presets, f, indent=2)

def load_presets(path="presets.json"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load()
    return {}