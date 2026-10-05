"""Load an observation-backed geometry profile for synthetic cone generation."""

import copy
import json
from pathlib import Path


def apply_agn_profile(config, profile_path):
    """Return a configuration with geometry ranges anchored to a saved fit.

    The observation's axis is provenance; position angle stays freely rotated
    for augmentation. Pixel radii are converted by the observation crop width.
    """
    path = Path(profile_path)
    profile = json.loads(path.read_text())
    source = Path(profile["geometry_source"])
    if not source.is_file():
        raise FileNotFoundError(f"AGN geometry source is missing: {source}")
    fit = json.loads(source.read_text())
    pathway_path = Path(profile["pathway_reference"]) if "pathway_reference" in profile else None
    pathway_fit = json.loads(pathway_path.read_text()) if pathway_path else None
    crop = int(profile["source_crop_pixels"])
    if crop <= 0:
        raise ValueError("source_crop_pixels must be positive")
    opening = float(fit["opening_half_deg"])
    outer = float(fit["r_max_px"]) / crop
    inner = float(fit["r_min_px"]) / crop
    spreads = profile["geometry_spread"]
    values = {
        "opening_angle_deg": (opening, float(spreads["opening_half_deg"])),
        "radius_fraction": (outer, float(spreads["radius_fraction"])),
        "inner_radius_fraction": (inner, float(spreads["inner_radius_fraction"])),
    }
    result = copy.deepcopy(config)
    for section, overrides in profile.get("overrides", {}).items():
        if section not in result or not isinstance(result[section], dict):
            raise ValueError(f"Unknown AGN profile section: {section}")
        for key, value in overrides.items():
            if key not in result[section]:
                raise ValueError(f"Unknown AGN profile control: {section}.{key}")
            result[section][key] = value
    for key, (center, spread) in values.items():
        if spread < 0:
            raise ValueError(f"Negative geometry spread for {key}")
        low, high = center - spread, center + spread
        if key != "opening_angle_deg" and (low <= 0 or high >= 0.5):
            raise ValueError(f"Invalid scaled radius range for {key}: {[low, high]}")
        if key == "opening_angle_deg" and (low <= 0 or high >= 90):
            raise ValueError(f"Invalid opening range: {[low, high]}")
        result["geometry"][key] = [low, high]
    result["agn_profile"] = {
        "name": profile["name"],
        "profile_path": str(path),
        "geometry_source": str(source),
        "source_crop_pixels": crop,
        "measured_axis_deg": float(fit["cone_axis_deg"]),
        "measured_opening_half_deg": opening,
        "measured_r_min_px": float(fit["r_min_px"]),
        "measured_r_max_px": float(fit["r_max_px"]),
        "measured_mask_fraction": float(fit["cone_mask_pixels"]) / (crop * crop),
        "pathway_reference": str(pathway_path) if pathway_path else None,
        "pathway_preview": profile.get("pathway_preview"),
        "reference_predicted_fraction": (
            float(pathway_fit["cone_fraction_pct"]) / 100 if pathway_fit else None
        ),
        "status": "geometry anchored; morphology and instrument ranges exploratory",
    }
    result.setdefault("labels", {})["detection_snr"] = float(fit["mask_snr_cut"])
    result["labels"]["peak_fraction_floor"] = float(profile["peak_fraction_floor"])
    return result
