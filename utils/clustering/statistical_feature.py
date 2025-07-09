import numpy as np
import os
import scipy.stats
import logging
import datetime
import nibabel as nib
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def load_images_from_directory(input_dir):
    """
    Recursively load all NIfTI volumes from the input directory and its subdirectories.
    Files must be named by series_id.nii or series_id.nii.gz.
    """
    print(f"Scanning directory for NIfTI files: {input_dir}")
    volume_paths = {}
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                full_path = os.path.join(root, file)
                # Remove both .nii and .gz extensions for the series_id
                series_id = file
                if series_id.endswith(".nii.gz"):
                    series_id = series_id[:-7]
                elif series_id.endswith(".nii"):
                    series_id = series_id[:-4]
                volume_paths[series_id] = full_path
    print(f"✓ Found {len(volume_paths)} NIfTI files")
    return volume_paths

def extract_features(volume):
    volume = volume[volume > 0]  # exclude air/background
    mean = np.mean(volume)
    std = np.std(volume)
    min_val = np.min(volume)
    max_val = np.max(volume)
    skew = scipy.stats.skew(volume.flatten())
    kurt = scipy.stats.kurtosis(volume.flatten())
    entropy = scipy.stats.entropy(np.histogram(volume, bins=64, density=True)[0] + 1e-6)
    p10 = np.percentile(volume, 10)
    p50 = np.percentile(volume, 50)
    p90 = np.percentile(volume, 90)
    
    return [mean, std, min_val, max_val, skew, kurt, entropy, p10, p50, p90]

input_dir = "utils/debug/ncct_cect/vindr_ds/original_volumes/"
volume_paths = load_images_from_directory(input_dir)
volume_info = {key: path for key, path in volume_paths.items()}


# Apply feature extraction
feature_matrix = np.array([extract_features(nib.load(vol_path).get_fdata()) for key, vol_path in volume_info.items()])


scaler = StandardScaler()
X_scaled = scaler.fit_transform(feature_matrix)



X_tsne = TSNE(n_components=2, perplexity=10, random_state=42).fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.title("t-SNE of CT Volumes Based on Statistical Features")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.grid(True)
plt.show()



kmeans = KMeans(n_clusters=3, random_state=0)
cluster_labels = kmeans.fit_predict(X_scaled)

# Plot with cluster colors
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels, cmap='tab10')
plt.colorbar()
plt.title("t-SNE with KMeans Clustering")
plt.show()
