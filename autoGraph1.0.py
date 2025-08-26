import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap

# ========================================================
# Helpers
# ========================================================
def _try_cubic_periodic(x_deg, y, new_deg):
    """Cubic periodic interpolation if SciPy exists; else linear."""
    try:
        from scipy.interpolate import CubicSpline
        x_ext = np.r_[x_deg, x_deg[0] + 360.0]
        y_ext = np.r_[y, y[0]]
        cs = CubicSpline(x_ext, y_ext, bc_type='periodic')
        return cs(new_deg)
    except Exception:
        x_ext = np.r_[x_deg, x_deg[0] + 360.0]
        y_ext = np.r_[y, y[0]]
        return np.interp(new_deg, x_ext, y_ext)

def aggregate_by_bins(df, bin_deg):
    """Sum Percentimeter by angular bins of width bin_deg."""
    df = df.dropna(subset=["CurrentAngle"]).copy()
    df["CurrentAngle"] = df["CurrentAngle"].astype(float) % 360
    df["Percentimeter"] = pd.to_numeric(df["Percentimeter"], errors="coerce").fillna(0)
    edges = np.arange(0, 360 + bin_deg, bin_deg)
    labels = edges[:-1]
    idx = pd.cut(df["CurrentAngle"], bins=edges, labels=labels,
                 include_lowest=True, right=False)
    sums = df.groupby(idx)["Percentimeter"].sum().reindex(labels, fill_value=0).values
    centers = (labels + bin_deg / 2.0) % 360
    return centers.astype(float), sums.astype(float)

def interpolate_to_1deg(centers_deg, values):
    """Smooth circular interpolation from coarse bins to 1° resolution."""
    new_deg = np.arange(0, 360, 1.0)
    smoothed = _try_cubic_periodic(np.sort(centers_deg), values[np.argsort(centers_deg)], new_deg)
    return new_deg.astype(int), np.clip(smoothed, 0, None)


# ========================================================
# 1. HEATMAP FUNCTION (locked circumferential)
# ========================================================
def pivot_heatmap(laminas_mm, titulo="Distribuição da Lâmina - Heatmap"):
    laminas_mm = np.asarray(laminas_mm, dtype=float)
    if laminas_mm.shape[0] != 360:
        raise ValueError("pivot_heatmap expects 360 values (one per degree).")

    angles = np.deg2rad(np.arange(0, 360))
    n_rings = 100
    radii = np.linspace(0, 1.0, n_rings)

    theta, r = np.meshgrid(angles, radii)
    z = np.tile(laminas_mm, (n_rings, 1))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    cmap = LinearSegmentedColormap.from_list(
        "yl_to_bl",
        ["#f7fcb9", "#c7e9b4", "#7fcdbb", "#41b6c4", "#2c7fb8", "#253494"]
    )
    im = ax.pcolormesh(theta, r, z, cmap=cmap, shading='auto')

    ax.set_theta_zero_location('E')   # East = 0°
    ax.set_theta_direction(1)         # CCW
    ax.grid(False)
    ax.set_yticklabels([])

    ax.set_title(titulo, va='bottom', fontsize=14)
    cbar = fig.colorbar(im, ax=ax, pad=0.1)
    cbar.set_label('Lâmina (mm)', rotation=270, labelpad=20)
    return fig, ax


# ========================================================
# 2. INFOGRAPHIC FUNCTION (30° bins)
# ========================================================
def pivot_infografico_unitcircle(laminas_mm, setor_size=30,
                                 titulo="Lâmina acumulada por faixa angular"):
    laminas_mm = np.array(laminas_mm, dtype=float)
    n_bins = len(laminas_mm)
    theta = np.linspace(0, 2*np.pi, n_bins, endpoint=False)
    width = 2*np.pi / n_bins

    max_lamina = max(laminas_mm) if np.max(laminas_mm) > 0 else 1.0
    barras_altura = (laminas_mm / max_lamina) * 0.6

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    raio_interno = 0.35
    altura_anel = 0.25
    ax.bar(theta, altura_anel, width=width, bottom=raio_interno,
           align='edge', edgecolor='white', linewidth=1)
    ax.bar(theta, barras_altura, width=width*0.9,
           bottom=raio_interno+altura_anel+0.05,
           align='edge', alpha=0.9, edgecolor='black')

    def faixa_label(i):
        a0 = int(i*setor_size) % 360
        a1 = int((i+1)*setor_size) % 360
        return f"{a0}°–{a1 if a1!=0 else 360}°"
    labels = [faixa_label(i) for i in range(n_bins)]
    ax.set_xticks(theta + width/2)
    ax.set_xticklabels(labels, fontsize=8)

    # Cardinal points
    ax.text(0, 1.05, "L", ha='center', va='center', fontsize=12)
    ax.text(np.pi/2, 1.05, "N", ha='center', va='center', fontsize=12)
    ax.text(np.pi, 1.05, "O", ha='center', va='center', fontsize=12)
    ax.text(3*np.pi/2, 1.05, "S", ha='center', va='center', fontsize=12)

    for t, mm in zip(theta, laminas_mm):
        ang_centro = t + width/2
        r_text = raio_interno + altura_anel/2
        ax.text(ang_centro, r_text, f"{int(mm)} mm", ha='center', va='center',
                fontsize=9, rotation=np.rad2deg(ang_centro), rotation_mode='anchor')

    ax.set_ylim(0, 1.05)
    ax.set_yticklabels([])
    plt.title(titulo, pad=20)
    plt.tight_layout()
    return fig, ax


# ========================================================
# 3. MAIN PIPELINE
# ========================================================
if __name__ == "__main__":
    excel_file = "autoReport_clean.xlsx"
    setor_size = 30

    df = pd.read_excel(excel_file)

    # (A) Aggregate raw data by 30° bins
    centers30, sums30 = aggregate_by_bins(df, setor_size)

    # (B) Interpolate smoothly to 1° resolution
    degs1, sums1 = interpolate_to_1deg(centers30, sums30)

    # Plot infográfico (30°)
    fig1, ax1 = pivot_infografico_unitcircle(
        sums30,
        setor_size=setor_size,
        titulo=f"Distribuição da Lâmina de Água - setores de {setor_size}°"
    )
    filename_info = f"infografico_{setor_size}_{datetime.now().strftime('%d-%m-%Y--%H-%M-%S')}.png"
    fig1.savefig(filename_info, dpi=200, bbox_inches="tight")
    print("Infográfico salvo em:", filename_info)

    # Plot heatmap (1° smooth)
    fig2, ax2 = pivot_heatmap(
        sums1,
        titulo="Distribuição da Lâmina de Água - setores de 1°"
    )
    filename_heat = f"heatmap_1_{datetime.now().strftime('%d-%m-%Y--%H-%M-%S')}.png"
    fig2.savefig(filename_heat, dpi=200, bbox_inches="tight")
    print("Heatmap salvo em:", filename_heat)

    plt.show()
