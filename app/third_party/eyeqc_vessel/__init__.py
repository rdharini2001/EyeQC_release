"""Optional vessel analysis (not part of EyeQC and not used by its score).

    engine.predict_vessels(rgb, device="cpu")      artery / vein / vessel probability maps from RRWNet
    morphometry.measure_morphometry(...)           vessel measurements in model-grid pixels
    render.artery_vein_overlay(rgb, result)        images for display

The pretrained RRWNet weights (Hugging Face: j-morano/rrwnet-rite) are downloaded on first use and are licensed
CC BY-NC 4.0 (non-commercial).
"""
