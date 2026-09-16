from astroquery.mast import Observations

obs = Observations.query_object("NGC 1068", radius="0.1 deg")

mask = (
    (obs["obs_collection"] == "HST") &
    ([("F469N" in str(f)) for f in obs["filters"]])
)
results = obs[mask]
results.sort("t_exptime", reverse=True)
print(f"Found {len(results)} observations")
print(results["obs_id", "instrument_name", "filters", "t_exptime", "proposal_id"])

# Inspect available products for the best (longest) observation
products = Observations.get_product_list(results[0])
print(products["productSubGroupDescription", "productFilename"])

# download these suckers
# Grab FLC for all 5 x 900s exposures (skip the short ones)
long_exp = results[results["t_exptime"] == 900.0]

for i in range(len(long_exp)):
    products = Observations.get_product_list(long_exp[i])
    flc = products[products["productSubGroupDescription"] == "FLC"]
    Observations.download_products(flc, download_dir="data/ngc1068_raw")
