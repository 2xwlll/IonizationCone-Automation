#!/usr/bin/env python3
"""Build a small Cloudy grid for nearby-AGN [O III] spectral cubes.

Cloudy supplies the photoionization and grain microphysics.  This script keeps
its raw inputs and outputs beside a compact ``grid.npz`` so every synthetic
cube can be traced to a specific physical model.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import itertools
import json
import math
import subprocess
from pathlib import Path

import numpy as np


REST_LINES_A = {
    "H_beta_4861": 4861.32,
    "OIII_4959": 4958.91,
    "OIII_5007": 5006.84,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path("configs/cloudy/ngc1068_oiii_v1.json"))
    parser.add_argument("--cloudy", type=Path, required=True,
                        help="Path to the compiled Cloudy executable")
    parser.add_argument("--output", type=Path,
                        default=Path("data/cloudy/ngc1068_oiii_v1"))
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N models for a quick check")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of independent Cloudy processes")
    return parser.parse_args()


def model_name(params: dict) -> str:
    packed = json.dumps(params, sort_keys=True).encode()
    return "m_" + hashlib.sha1(packed).hexdigest()[:12]


def cloudy_input(params: dict, line_list_name: str) -> str:
    z_log = math.log10(params["metallicity_solar"])
    dust_log = math.log10(params["dust_scale"])
    commands = [
        f'title "NGC1068 OIII grid {params["model_id"]}"',
        "table agn",
        f'ionization parameter {params["log_u"]:.5f}',
        f'hden {params["log_nh_cm3"]:.5f}',
        "abundances ism no grains",
        f'metals {z_log:.6f} log',
        # A single equilibrium-temperature bin retains the ISM graphite and
        # silicate opacities while keeping a grid run tractable.  Full grain
        # size distributions can be enabled for the final calibrated grid.
        f'grains ism {dust_log:.6f} log no qheat single',
    ]
    if params["pah_scale"] > 0:
        commands.append(
            f'grains PAH {math.log10(params["pah_scale"]):.6f} log no qheat single'
        )
    commands.extend([
        f'stop column density {params["log_column_density_cm2"]:.5f}',
        "print last iteration",
        f'save line list "{params["model_id"]}.lin" "{line_list_name}" last no hash emergent absolute',
        f'save continuum "{params["model_id"]}.con" last units Angstrom',
    ])
    return "\n".join(commands) + "\n"


def parse_linelist(path: Path) -> dict[str, float]:
    """Read Cloudy's saved line list, accepting tab or whitespace formats."""
    # C25's ``save line list`` format is a tabular header followed by one row
    # per iteration.  The requested line order is preserved.
    tab_rows = [line.split("\t") for line in path.read_text().splitlines()
                if line.strip()]
    iteration_rows = [row for row in tab_rows
                      if row[0].strip().lower().startswith("iteration")]
    if iteration_rows:
        values = [float(value) for value in iteration_rows[-1][1:]]
        if len(values) == len(REST_LINES_A):
            return dict(zip(REST_LINES_A, values))

    wanted = {round(v, 2): k for k, v in REST_LINES_A.items()}
    found: dict[str, float] = {}
    for raw in path.read_text().splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        fields = raw.replace("\t", " ").split()
        numbers = []
        for token in fields:
            try:
                numbers.append(float(token.rstrip("A")))
            except ValueError:
                pass
        for wavelength, key in wanted.items():
            matching = [i for i, value in enumerate(numbers)
                        if abs(value - wavelength) < 0.08]
            if matching and len(numbers) > matching[0] + 1:
                # The final numeric column is the emergent line intensity.
                found[key] = float(numbers[-1])
    missing = sorted(set(REST_LINES_A) - set(found))
    if missing:
        raise ValueError(f"Missing {missing} in Cloudy output {path}")
    return found


def parse_emergent_continuum(path: Path,
                             target_wavelength_a: np.ndarray) -> np.ndarray:
    """Return outward line-free continuum as F_lambda on a rest-frame grid.

    Cloudy's save-continuum columns are nuFnu.  Column 4 is outward diffuse
    continuum plus lines and column 9 is the outward line contribution.  Their
    difference divided by wavelength gives a quantity proportional to F_lambda.
    """
    wavelength = []
    outward_continuum = []
    for raw in path.read_text().splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        fields = raw.replace("\t", " ").split()
        if len(fields) < 9:
            continue
        try:
            values = [float(value) for value in fields[:9]]
        except ValueError:
            continue
        wave_a = values[0]
        nu_fnu_continuum = max(values[3] - values[8], 0.0)
        if np.isfinite(wave_a) and wave_a > 0 and np.isfinite(nu_fnu_continuum):
            wavelength.append(wave_a)
            outward_continuum.append(nu_fnu_continuum / wave_a)
    wavelength = np.asarray(wavelength)
    outward_continuum = np.asarray(outward_continuum)
    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    outward_continuum = outward_continuum[order]
    keep = ((wavelength >= target_wavelength_a.min() - 100.0)
            & (wavelength <= target_wavelength_a.max() + 100.0))
    if keep.sum() < 2:
        raise ValueError(f"Insufficient optical continuum samples in {path}")
    wavelength = wavelength[keep]
    outward_continuum = outward_continuum[keep]
    # Linear interpolation is safer around zero-valued grain-absorbed bins.
    return np.interp(target_wavelength_a, wavelength, outward_continuum)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    executable = args.cloudy.resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)

    output = args.output.resolve()
    runs = output / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    line_list = runs / "oiii_lines.dat"
    line_list.write_text(
        "H  1  4861.32A\n"
        "O  3  4958.91A\n"
        "O  3  5006.84A\n"
    )

    grid = config["cloudy"]
    combinations = list(itertools.product(
        grid["log_u"], grid["log_nh_cm3"], grid["metallicity_solar"],
        grid["grain_models"],
    ))
    if args.limit is not None:
        combinations = combinations[:args.limit]
    cube_cfg = config["cube"]
    observed_wavelength = np.linspace(
        cube_cfg["wavelength_min_angstrom"],
        cube_cfg["wavelength_max_angstrom"],
        cube_cfg["wavelength_channels"],
    )
    continuum_rest_wavelength = observed_wavelength / (1.0 + config["redshift"])

    def run_one(index_and_values: tuple) -> tuple[int, dict, bool]:
        index, values = index_and_values
        log_u, log_nh, metallicity, grains = values
        params = {
            "log_u": float(log_u),
            "log_nh_cm3": float(log_nh),
            "metallicity_solar": float(metallicity),
            "log_column_density_cm2": float(grid["log_column_density_cm2"]),
            **grains,
        }
        params["model_id"] = model_name({**params, "output_schema": 2})
        stem = params["model_id"]
        (runs / f"{stem}.in").write_text(cloudy_input(params, line_list.name))
        saved_lines = runs / f"{stem}.lin"
        saved_continuum = runs / f"{stem}.con"
        try:
            intensities = parse_linelist(saved_lines)
            continuum = parse_emergent_continuum(
                saved_continuum, continuum_rest_wavelength
            )
            cached = True
        except (FileNotFoundError, ValueError):
            cached = False
            completed = subprocess.run(
                [str(executable), "-r", stem], cwd=runs,
                text=True, capture_output=True,
            )
            (runs / f"{stem}.stdout.txt").write_text(completed.stdout)
            (runs / f"{stem}.stderr.txt").write_text(completed.stderr)
            if completed.returncode:
                raise RuntimeError(
                    f"Cloudy failed for {stem}; see {runs / (stem + '.out')}"
                )
            intensities = parse_linelist(saved_lines)
            continuum = parse_emergent_continuum(
                saved_continuum, continuum_rest_wavelength
            )
        return index, {**params, **intensities,
                       "continuum_outward_flambda": continuum}, cached

    records_by_index = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(run_one, item)
                   for item in enumerate(combinations)]
        for completed_count, future in enumerate(as_completed(futures), start=1):
            index, record, cached = future.result()
            records_by_index[index] = record
            source = "cache" if cached else "Cloudy"
            ratio = record["OIII_5007"] / max(record["H_beta_4861"], 1e-300)
            print(f"[{completed_count}/{len(futures)}] {record['model_id']} "
                  f"({source}): OIII5007/Hbeta={ratio:.4f}",
                  flush=True)
    records = [records_by_index[index] for index in sorted(records_by_index)]

    keys = [
        "log_u", "log_nh_cm3", "metallicity_solar", "dust_scale",
        "pah_scale", *REST_LINES_A,
    ]
    arrays = {key: np.asarray([r[key] for r in records], dtype=np.float64)
              for key in keys}
    arrays["model_id"] = np.asarray([r["model_id"] for r in records])
    arrays["rest_wavelength_angstrom"] = np.asarray(list(REST_LINES_A.values()))
    arrays["line_names"] = np.asarray(list(REST_LINES_A))
    arrays["continuum_rest_wavelength_angstrom"] = continuum_rest_wavelength
    arrays["continuum_outward_flambda"] = np.stack(
        [r["continuum_outward_flambda"] for r in records]
    )
    np.savez_compressed(output / "grid.npz", **arrays)
    (output / "manifest.json").write_text(json.dumps({
        "cloudy_executable": str(executable),
        "cloudy_version": "C25.00_rc2",
        "config": str(args.config),
        "models": [{key: value for key, value in record.items()
                    if key != "continuum_outward_flambda"}
                   for record in records],
    }, indent=2))
    print(f"Saved {len(records)} physical models to {output / 'grid.npz'}")


if __name__ == "__main__":
    main()
