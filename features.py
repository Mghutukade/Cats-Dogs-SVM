from skimage.feature import hog

def extract_hog_features(images, pixels_per_cell=(8, 8)):
    hog_features = []
    for img in images:
        hog_feat = hog(
            img.reshape(64, 64), 
            pixels_per_cell=pixels_per_cell,
            cells_per_block=(2, 2),
            orientations=9,
            block_norm='L2-Hys'
        )
        hog_features.append(hog_feat)
    return hog_features
