import numpy as np
from sklearn.datasets import fetch_olivetti_faces

def _avg_pool(img, block_size=4):
    """Helper function to downscale an image via average pooling."""
    h, w = img.shape
    img = img[:h - (h % block_size), :w - (w % block_size)]
    h2, w2 = img.shape[0] // block_size, img.shape[1] // block_size
    return img.reshape(h2, block_size, w2, block_size).mean(axis=(1, 3))

def prepare_olivetti_data(person_ids, pool_size):
    """Loads, filters, and prepares the Olivetti faces dataset."""
    print("1. Preparing data...")
    faces = fetch_olivetti_faces()
    
    pooled_images = np.array([_avg_pool(img, block_size=pool_size) for img in faces.images])
    
    mask = np.isin(faces.target, person_ids)
    X_selected, y_selected = pooled_images[mask], faces.target[mask]
    
    unique_ids = np.unique(y_selected)
    id_map = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
    y_true = np.vectorize(id_map.get)(y_selected)
    
    X = X_selected.reshape(X_selected.shape[0], -1)
    n_classes = len(unique_ids)
    
    print("Data preparation complete.")
    return X, y_true, n_classes