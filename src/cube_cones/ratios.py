# cube_lines_and_ratios.py

cube_lines_and_ratios = {
    # Ch1 cubes
    "jw03535-o008_t003_miri_ch1-short_s3d.fits": {
        "lines": ["FeII_5.34", "FeVIII_5.45", "MgVII_5.50", "MgV_5.61"],
        "ratios": [("FeVIII_5.45", "FeII_5.34"), ("MgVII_5.50", "MgV_5.61")],
        "half_width_um": 0.03,
        "edge_caution": True,
    },
    "jw03535-c1005_t003_miri_ch1-short_s3d.fits": {
        "lines": ["FeII_5.34", "FeVIII_5.45", "MgVII_5.50", "MgV_5.61"],
        "ratios": [("FeVIII_5.45", "FeII_5.34"), ("MgVII_5.50", "MgV_5.61")],
        "half_width_um": 0.03,
        "edge_caution": True,
    },
    "jw03535-o008_t003_miri_ch1-long_s3d.fits": {
        "lines": ["ArII_6.99", "NaIII_7.32", "Pf_alpha_7.46"],
        "ratios": [("ArII_6.99", "NaIII_7.32")],
        "half_width_um": 0.04,
        "edge_caution": True,  # Pf_alpha near top of range
    },
    "jw03535-c1005_t003_miri_ch1-long_s3d.fits": {
        "lines": ["ArII_6.99", "NaIII_7.32", "Pf_alpha_7.46"],
        "ratios": [("ArII_6.99", "NaIII_7.32")],
        "half_width_um": 0.04,
        "edge_caution": True,
    },

    # Ch2 cubes
    "jw03535-o008_t003_miri_ch2-short_s3d.fits": {
        "lines": ["NeVI_7.65", "FeVII_7.82", "ArV_7.90"],
        "ratios": [("NeVI_7.65", "FeVII_7.82"), ("NeVI_7.65", "ArV_7.90")],
        "half_width_um": 0.04,
    },
    "jw03535-c1005_t003_miri_ch2-short_s3d.fits": {
        "lines": ["NeVI_7.65", "FeVII_7.82", "ArV_7.90"],
        "ratios": [("NeVI_7.65", "FeVII_7.82"), ("NeVI_7.65", "ArV_7.90")],
        "half_width_um": 0.04,
    },
    "jw03535-c1005_t003_miri_ch2-medium_s3d.fits": {
        "lines": ["ArIII_8.99", "FeVII_9.53"],
        "ratios": [("ArIII_8.99", "FeVII_9.53")],
        "half_width_um": 0.04,
    },
    "jw03535-o008_t003_miri_ch2-medium_s3d.fits": {
        "lines": ["ArIII_8.99", "FeVII_9.53"],
        "ratios": [("ArIII_8.99", "FeVII_9.53")],
        "half_width_um": 0.04,
    },
    "jw03535-o008_t003_miri_ch2-long_s3d.fits": {
        "lines": ["SIV_10.51"],
        "ratios": [],
        "half_width_um": 0.04,
    },
    "jw03535-c1005_t003_miri_ch2-long_s3d.fits": {
        "lines": ["SIV_10.51"],
        "ratios": [],
        "half_width_um": 0.04,
    },

    # Ch3 cubes
    "jw03535-o008_t003_miri_ch3-short_s3d.fits": {
        "lines": ["Hu_alpha_12.37", "NeII_12.81", "ArV_13.10"],
        "ratios": [("NeII_12.81", "Hu_alpha_12.37"), ("ArV_13.10", "NeII_12.81")],
        "half_width_um": 0.05,
    },
    "jw03535-c1005_t003_miri_ch3-short_s3d.fits": {
        "lines": ["Hu_alpha_12.37", "NeII_12.81", "ArV_13.10"],
        "ratios": [("NeII_12.81", "Hu_alpha_12.37"), ("ArV_13.10", "NeII_12.81")],
        "half_width_um": 0.05,
    },
    "jw03535-o008_t003_miri_ch3-medium_s3d.fits": {
        "lines": ["NeV_14.32", "NeIII_15.56"],
        "ratios": [("NeV_14.32", "NeIII_15.56")],
        "half_width_um": 0.05,
    },
    "jw03535-c1005_t003_miri_ch3-medium_s3d.fits": {
        "lines": ["NeV_14.32", "NeIII_15.56"],
        "ratios": [("NeV_14.32", "NeIII_15.56")],
        "half_width_um": 0.05,
    },
    "jw03535-o008_t003_miri_ch3-long_s3d.fits": {
        "lines": ["NeIII_15.56"],
        "ratios": [],
        "half_width_um": 0.05,
    },
    "jw03535-c1005_t003_miri_ch3-long_s3d.fits": {
        "lines": ["NeIII_15.56"],
        "ratios": [],
        "half_width_um": 0.05,
    },

    # Ch4 cubes
    "jw03535-o008_t003_miri_ch4-short_s3d.fits": {
        "lines": ["SIII_18.71"],
        "ratios": [],
        "half_width_um": 0.06,
    },
    "jw03535-c1005_t003_miri_ch4-short_s3d.fits": {
        "lines": ["SIII_18.71"],
        "ratios": [],
        "half_width_um": 0.06,
    },
    "jw03535-o008_t003_miri_ch4-medium_s3d.fits": {
        "lines": ["ArIII_21.83", "NeV_24.32"],
        "ratios": [("ArIII_21.83", "NeV_24.32")],
        "half_width_um": 0.06,
    },
    "jw03535-c1005_t003_miri_ch4-medium_s3d.fits": {
        "lines": ["ArIII_21.83", "NeV_24.32"],
        "ratios": [("ArIII_21.83", "NeV_24.32")],
        "half_width_um": 0.06,
    },
    "jw03535-o008_t003_miri_ch4-long_s3d.fits": {
        "lines": ["OIV_25.89", "FeII_25.99"],
        "ratios": [("OIV_25.89", "FeII_25.99")],
        "half_width_um": 0.06,
    },
    "jw03535-c1005_t003_miri_ch4-long_s3d.fits": {
        "lines": ["OIV_25.89", "FeII_25.99"],
        "ratios": [("OIV_25.89", "FeII_25.99")],
        "half_width_um": 0.06,
    },
}

