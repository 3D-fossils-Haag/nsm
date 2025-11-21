# Utility functions for fine-tuning optimization of novel meshes in trained models

from sklearn.decomposition import PCA
import numpy as np
import torch

# Initialize latent near PCA offset mean
def pca_initialize_latent(mean_latent, latent_codes, top_k=10):
    # Convert to numpy
    latent_np = latent_codes.detach().cpu().numpy()
    mean_np = mean_latent.detach().cpu().numpy().squeeze()
    pca = PCA(n_components=latent_np.shape[1])
    pca.fit(latent_np)
    # Sample along top-K PCs
    top_components = pca.components_[:top_k]  # (K, D)
    top_eigenvalues = pca.explained_variance_[:top_k]
    scale = 0.01  # tune this
    coeffs = np.random.randn(top_k) * np.sqrt(top_eigenvalues) * scale
    pca_offset = np.dot(coeffs, top_components)  # D
    init_latent = mean_np + pca_offset
    return torch.tensor(init_latent, dtype=torch.float32, device=latent_codes.device).unsqueeze(0)

# Get top k PCA's based on defined explained variance threshold
def get_top_k_pcs(latent_codes, threshold=0.90):
    latent_np = latent_codes.cpu().numpy()
    pca = PCA()
    pca.fit(latent_np)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    k = np.searchsorted(cum_var, threshold)
    print(f"Selected top {k+1} PCs to explain {threshold*100:.1f}% of variance")
    return pca, k + 1

# Find the top 5 most similar meshes from training data to novel/input mesh - uses L2 (euclidian) distance in latent space
def find_similar(latent_novel, latent_codes, top_k=5, n_std=2):
    dists = torch.norm(latent_codes - latent_novel, dim=1)
    mean_dist = dists.mean()
    std_dist = dists.std()
    threshold = mean_dist + n_std * std_dist
    within = dists <= threshold
    sorted_idx = torch.argsort(dists[within])[:top_k]
    similar_ids = torch.nonzero(within).squeeze()[sorted_idx]
    print(f"similar_ids shape: {similar_ids.shape}")
    print(f"similar_ids: {similar_ids}")
    return similar_ids.tolist(), dists[similar_ids].tolist()

# Find the top 5 most similar meshes from training data to novel/input mesh - uses cosine distance in latent space
def find_similar_cos(latent_novel, latent_codes, top_k=5, n_std=2, device='cuda'):
    # Compute cosine similarity between each latent code and the novel latent code
    cosine_similarities = F.cosine_similarity(latent_codes.to(device), latent_novel.to(device), dim=1)
    cosine_distances = 1 - cosine_similarities
    mean_dist = cosine_distances.mean()
    std_dist = cosine_distances.std()
    # Apply threshold (mean + n_std * std)
    threshold = mean_dist + n_std * std_dist
    within = cosine_distances <= threshold
    # Sort distances within the threshold and get top_k
    within_indices = torch.nonzero(within, as_tuple=False).squeeze()
    if within_indices.numel() == 0:
        print("No similar items within the threshold.")
        return [], []
    # If only one index remains, ensure it's a 1D tensor
    if within_indices.ndim == 0:
        within_indices = within_indices.unsqueeze(0)
    sorted_indices = torch.argsort(cosine_distances[within_indices])[:top_k]
    similar_ids = within_indices[sorted_indices]  # 1D: shape [top_k]
    print(f"similar_ids shape: {similar_ids.shape}")
    print(f"similar_ids: {similar_ids}")
    return similar_ids.tolist(), cosine_distances[similar_ids].tolist()