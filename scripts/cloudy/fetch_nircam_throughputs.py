"""Cache STScI's detector-averaged NIRCam total-system response tables."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import numpy as np
import requests

PAGE = "https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-filters"
ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "configs/instruments/jwst_nircam"


def main():
    page = requests.get(PAGE, timeout=60)
    page.raise_for_status()
    soup = BeautifulSoup(page.text, "html.parser")
    entries = []
    for row in soup.select("tr"):
        link = row.find("a", href=lambda h: h and "mean_system_throughput.txt" in h)
        if link is None:
            continue
        cells = row.find_all("td")
        name = link.get_text(strip=True)
        entries.append({
            "name": name, "source_url": urljoin(PAGE, link["href"]),
            "pivot_um": float(cells[1].get_text(strip=True)),
            "bandwidth_um": float(cells[2].get_text(strip=True)),
            "half_power_blue_um": float(cells[4].get_text(strip=True)),
            "half_power_red_um": float(cells[5].get_text(strip=True)),
            "channel": "short" if int(name[1:4]) < 240 else "long",
            "native_pixel_scale_arcsec": 0.031 if int(name[1:4]) < 240 else 0.063,
            "file": f"{name}.csv",
        })
    if len(entries) != 29 or len({e["name"] for e in entries}) != 29:
        raise ValueError("Expected exactly 29 unique NIRCam filters; inspect STScI page")

    def download(entry):
        response = requests.get(entry["source_url"], timeout=60)
        response.raise_for_status()
        data = np.loadtxt(StringIO(response.text), skiprows=1)
        if (data.ndim != 2 or data.shape[1] != 2 or not np.isfinite(data).all()
                or np.any(np.diff(data[:, 0]) <= 0)
                or np.any(data[:, 1] > 1)):
            raise ValueError(f"Invalid response for {entry['name']}")
        # Some measured tables contain small negative values outside the band.
        # Retain the exact source alongside the physical nonnegative response.
        entry["negative_samples_clipped"] = int(np.count_nonzero(data[:, 1] < 0))
        entry["source_minimum_throughput"] = float(data[:, 1].min())
        data[:, 1] = np.maximum(data[:, 1], 0)
        data[:, 0] *= 1e4  # official microns -> shared photometry Angstrom convention
        entry["source_sha256"] = sha256(response.content).hexdigest()
        return entry, data, response.content

    # Finish and validate all downloads before writing the catalog.
    with ThreadPoolExecutor(max_workers=4) as pool:
        downloaded = list(pool.map(download, entries))
    DEST.mkdir(parents=True, exist_ok=True)
    for entry, data, original in downloaded:
        (DEST / f"{entry['name']}_source.txt").write_bytes(original)
        np.savetxt(DEST / entry["file"], data, delimiter=",",
                   header="observed_wavelength_angstrom,total_system_throughput")
    catalog = {"source_page": PAGE, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
               "response": "detector-averaged total system; pupil-filter combinations included",
               "filters": entries}
    (DEST / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    print(f"Cached {len(entries)} NIRCam response curves in {DEST}")


if __name__ == "__main__":
    main()
