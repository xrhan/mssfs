import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch
import ot
from sklearn.manifold import TSNE
from utils import *


def compute_wasserstein_distance(X: np.ndarray, Y: np.ndarray) -> float:
    M = ot.dist(X, Y, metric='euclidean')
    a = np.ones((X.shape[0],)) / X.shape[0]
    b = np.ones((Y.shape[0],)) / Y.shape[0]
    W_distance = ot.emd2(a, b, M)
    print(f"Wasserstein distance: {W_distance}")
    return W_distance


def process_one_cycle_img(file_path: str, result_fusion: bool = False) -> dict[int, np.ndarray]:
    # Step 1: Read in image files
    with open(file_path, 'rb') as f:
        results = pickle.load(f)

    # Step 2: Transform normal prediction (patch) at multiple steps to global shape
    single_result_normal = {}
    end_index = len(results) - 1 if result_fusion else len(results)
    for i in range(end_index):
        idx = i
        batch_size, _, patch_size, _ = results[idx].shape
        side_num = int(math.sqrt(batch_size))
        global_normal = np.moveaxis(
            single_depatchify(results[idx], patch_size, side_num, side_num), 0, -1
        )
        global_normal = normalize_normals_torch(torch.tensor(global_normal)).numpy()
        single_result_normal[i] = global_normal
    
    return single_result_normal


def process_all_result(root_dir, img_id, seeds):
    # Step 3: process different random seeds
    all_results = {}
    for seed in seeds:
        file_name = f'{root_dir}img_{img_id}_scheduler_1_seed_{seed}_img.pkl'
        all_results[seed] = process_one_cycle_img(file_name)
    return all_results
        

def compute_tsne(
    normal_predictions: list[np.ndarray],
    resize: bool = True,
    target_size: int = 32,
    perplexity: int = 30
) -> tuple[np.ndarray, np.ndarray]:

    if resize:
        resized_predictions = np.array([
            cv2.resize(norm, (target_size, target_size)) for norm in normal_predictions
        ])
    else:
        resized_predictions = np.array(normal_predictions)

    reshaped_data = resized_predictions.reshape(len(normal_predictions), -1)
    tsne = TSNE(n_components=2, random_state=2, perplexity=perplexity)
    tsne_result = tsne.fit_transform(reshaped_data)
    return tsne_result, resized_predictions


def visualize_tsne(
    tsne_result: np.ndarray,
    labels: list,
    figure_name: str,
    save_path: str
) -> None:
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter, label='Labels')

    for i, label in enumerate(labels):
        plt.text(tsne_result[i, 0] + 0.3, tsne_result[i, 1], str(label), fontsize=10, va='center')

    plt.title(f't-SNE Visualization: {figure_name}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    save_file = f'{save_path}{figure_name}_tsne_vis.pdf'
    plt.savefig(save_file, format='pdf')
    plt.close()
    print(f"Saved t-SNE visualization to {save_file}")
    