def log_hdu_info(hdul, selected_hdu):
    print("📄 HDU summary:")
    for i, hdu in enumerate(hdul):
        shape = getattr(hdu.data, 'shape', None)
        dtype = getattr(hdu.data, 'dtype', None)
        prefix = "👉" if hdu == selected_hdu else "  "
        print(f"{prefix} HDU[{i}]: shape={shape}, dtype={dtype}")

