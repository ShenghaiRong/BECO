import math
from typing import List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from umap import UMAP
import torch
from torch import Tensor


"""
All functions in this package should start with `plot`
"""

"""
t-SNE help: https://distill.pub/2016/misread-tsne/
UMAP Help: https://pair-code.github.io/understanding-umap/
"""


def plot_embeddings(embeddings: np.ndarray, labels: np.ndarray, algo: str='tsne', perplexity: int=50):
    """
    This function do t-SNE on embeddings and draws an 2-D plot

    NOTE: refer to "https://seaborn.pydata.org/tutorial/color_palettes.html"
        to choose the colormap you like

    Args:
        algo: choose from 'tsne' and 'umap'
        embeddings: 2-D ndarray of shape (N, C)
        labels: 1-D ndarray of shape N
    """
    if algo == 'tsne':
        fit = TSNE(
            n_components=2, init='pca', learning_rate='auto',
            perplexity=perplexity, n_iter=1000, n_jobs=8
        )
    elif algo == 'umap':
        fit = UMAP(
            n_components=2, n_neighbors=perplexity, min_dist=0.1,
            n_jobs=8
        )
    # pos has the shape of (N, 2)
    pos = fit.fit_transform(embeddings)
    data_frame = pd.DataFrame({
        'X': pos[:, 0],
        'Y': pos[:, 1],
        'lables': labels
    })
    sns.relplot(data=data_frame, palette="viridis", x='X', y='Y', hue='lables', height=8)
    return plt.gcf()


def plot_embeddings_3d(embeddings: np.ndarray, labels: np.ndarray, algo: str='tsne', perplexity: int=50):
    """
    This function do t-SNE on embeddings and draws an 3-D plot

    Args:
        embeddings: 2-D ndarray of shape (N, C)
        labels: 1-D ndarray of shape N
    """
    if algo == 'tsne':
        fit = TSNE(
            n_components=3, init='pca', learning_rate='auto',
            perplexity=perplexity, n_iter=1000, n_jobs=8
        )
    elif algo == 'umap':
        fit = UMAP(
            n_components=3, n_neighbors=perplexity, min_dist=0.1,
            n_jobs=8
        )
    # pos has the shape of (N, 3)
    pos = fit.fit_transform(embeddings)
    # If using umap, should shift them to have zero global mean
    if algo == 'umap':
        global_mean = np.mean(pos, axis=0)
        pos -= global_mean
    # scale the maximum value of each axis to 1.
    pos = pos / np.max(pos, axis=0)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')

    # Draw a transparent sphere
    phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
    x = 1.0 * np.sin(phi) * np.cos(theta)
    y = 1.0 * np.sin(phi) * np.sin(theta)
    z = 1.0 * np.cos(phi)
    ax.plot_surface(x, y, z,  rstride=1, cstride=1, color='w', alpha=0.3)
    # Draw embeddings
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=labels, s=20)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect((4, 4, 4))
    fig.tight_layout()
    
    return fig


def plot_confusion_matrix(matrix: np.ndarray, matrix_ref : np.ndarray=None, normalize_row: bool=True):
    """
    visualize confusion matrix

    Args:
        matrix: The confusion matrix
        matrix_ref: If given, a relative confusion matrix is given by `matrix - matrix_ref`
    """
    if matrix_ref is not None:
        matrix = matrix - matrix_ref
    # normalize each row of confusion matrix
    if normalize_row:
        gt_sum = matrix.sum(axis=1, keepdims=True)
        hist = matrix / gt_sum
    else:
        gt_max = matrix.max()
        hist = matrix / gt_max
    f, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(hist, cmap="YlGnBu", annot=True, fmt='.5f', annot_kws={'fontsize': 7})
    ax.set_ylabel("Ground Truth")
    ax.set_xlabel("Prediction")
    return plt.gcf()


def plot_out_prob_distribution(prob: Tensor):
    """
    visualize output probability distribution of an image using bar_plot

    Args:
        prob: the output probability, with the shape of C * H * W
    """
    prob = torch.sort(prob, dim=0, descending=True)
    prob = torch.mean(prob, dim=(1, 2))
    prob = prob.numpy()
    sns.relplot(data=prob, kind='line')
    return plt.gcf()


def plot_heatmap(logits: torch.Tensor):
    """
    This function computes the entropy of the output logits, and change it to
    a heatmap. Originally desinged for simseg networks
    
    Args:
        logits: H * W * C
    """
    def _entropy(logits: torch.Tensor):
        EPS = 1e-8
        C = logits.shape[0]
        prob = torch.softmax(logits, dim=0)
        log_prob = torch.log_softmax(logits, dim=0)
        return (-1.0 / math.log(C) + EPS) * torch.sum(prob * log_prob, dim=0)

    entropy_map = _entropy(logits).numpy()
    ax = sns.heatmap(entropy_map, cmap="YlGnBu")
    ax.set_xticks([])
    ax.set_yticks([])
    return plt.gcf()


def plot_accuracy_barplot(metric_name: str, metric: List[float]):
    """
    plot a metric list as a bar plot
    """
    num = len(metric)
    data = pd.DataFrame({
        'class_idx': np.linspace(0, num-1, num, dtype=np.int32),
        metric_name : metric
    })
    ax = sns.barplot(data=data, x='class_idx', y=metric_name, color='c')
    ax.bar_label(ax.containers[0], fontsize=7)
    return plt.gcf()