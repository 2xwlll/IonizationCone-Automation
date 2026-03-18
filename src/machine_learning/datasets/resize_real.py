def resize_spatial(cube, target_shape):
    from skimage.transform import resize
    resized = np.empty((cube.shape[0], target_shape[1], target_shape[2]), dtype=np.float32)
    for i in range(cube.shape[0]):
        resized[i] = resize(cube[i], (target_shape[1], target_shape[2]), mode='reflect', anti_aliasing=True)
    return resized

