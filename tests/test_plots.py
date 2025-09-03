import matplotlib.pyplot as plt
from autoReport import pivot_heatmap, pivot_infografico_unitcircle
import numpy as np

def test_heatmap_runs():
    data = np.arange(360)
    fig, ax = pivot_heatmap(data)
    assert fig is not None
    plt.close(fig)

def test_unitcircle_runs():
    data = np.arange(12)
    fig, ax = pivot_infografico_unitcircle(data)
    assert fig is not None
    plt.close(fig)
