"""Vessel-structure analysis add-on for the EyeQC webapp.

Internal engineering note (never surfaced to end users — see
``webapp/server.py``'s ``/api/vessel`` endpoint for the clinical-language
framing): this package wraps RRWNet (Morano, Aresta & Bogunovic, *Expert
Systems with Applications*, 2024), a deep artery/vein/vessel segmentation
network, ported and trimmed from the vendored reference implementation at
``third_party/rrwnet-03069bb673f87d4b3cee37df378e0f15e5ade280`` for CPU
single-image inference. It is self-contained: nothing here imports from
``third_party`` at runtime, so ``code_final`` stays deployable on its own.

Public surface:
    engine.load_rrwnet(device="cpu")          -- lazy singleton model loader
    engine.predict_vessels(rgb, device="cpu")  -- artery/vein/vessel maps
    engine.summarize_vessel_features(probs, roi) -- 10 scalar features
    render.vessel_overlay_figure(rgb, result)  -- artery/vein colour overlay
    render.vessel_skeleton_figure(rgb, result) -- vessel network skeleton map
"""
