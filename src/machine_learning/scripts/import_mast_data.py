# src/import_mast_data.py

from astroquery.mast import Observations
from pathlib import Path
import os

def search_jwst_agn(target_name="NGC 1068"):
    """
    Search for JWST observations of a given AGN target.
    Loosely filtered to help debug data availability.
    """
    obs_table = Observations.query_object(target_name, radius="0.02 deg")
    print(f"Found {len(obs_table)} total observations for {target_name}")
    
    # Optional: narrow to JWST only
    jwst_obs = obs_table[obs_table['obs_collection'] == 'JWST']
    print(f"Found {len(jwst_obs)} JWST observations for {target_name}")

    return jwst_obs



def download_jwst_data(obs_table, download_dir="data/mast_raw"):
    if obs_table is None or len(obs_table) == 0:
        print("No observations to download.")
        return None

    if not Path(download_dir).exists():
        os.makedirs(download_dir)

    print("Fetching product list...")
    products = Observations.get_product_list(obs_table)

    # Be defensive: not all MAST products have the same columns
    if "productSubGroupDescription" in products.colnames:
        sci_products = products[products["productSubGroupDescription"] == "SCI"]
    else:
        print("Warning: 'productSubGroupDescription' not found. Using all products.")
        sci_products = products

    print(f"Found {len(sci_products)} products. Downloading...")
    manifest = Observations.download_products(
        sci_products,
        download_dir=download_dir,
        curl_flag=False
    )

    if manifest is not None and "Local Path" in manifest.colnames:
        print("Downloaded files:")
        print(manifest["Local Path"])
    else:
        print("No files downloaded.")

    return manifest


if __name__ == "__main__":
    obs = search_jwst_agn("NGC 1068", filters=["F335M", "F356W"])
    manifest = download_jwst_data(obs)

