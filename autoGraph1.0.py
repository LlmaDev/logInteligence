import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# ============================================
# 1. HEATMAP FUNCTION
# ============================================
def pivot_heatmap(laminas_mm, setor_size=360, titulo="Distribuição da Lâmina - Heatmap"):
    laminas_mm = np.array(laminas_mm, dtype=float)
    n_bins = len(laminas_mm)

    # Ângulos dos setores
    angles = np.linspace(0, 2*np.pi, n_bins, endpoint=False)

    # Criar raios concêntricos (rings)
    n_rings = 50
    radii = np.linspace(0, setor_size/2, n_rings)

    # Malha polar
    theta, r = np.meshgrid(angles, radii)

    # Circumferential heatmap → mesma cor em todo raio do setor
    z = np.tile(laminas_mm, (n_rings, 1))

    # Figura polar
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    im = ax.pcolormesh(theta, r, z, cmap='viridis_r', shading='auto')

    # Config polar
    ax.set_theta_zero_location('E')   # zero graus no Norte
    ax.set_theta_direction(1)        # sentido horário
    ax.grid(False)

    # Título e colorbar
    ax.set_title(titulo, va='bottom', fontsize=14)
    cbar = fig.colorbar(im, ax=ax, pad=0.1)
    cbar.set_label('Lâmina (mm)', rotation=270, labelpad=20)

    return fig, ax


# ============================================
# 2. INFOGRAPHIC FUNCTION
# ============================================
def pivot_infografico_unitcircle(laminas_mm, setor_size=30, titulo="Lâmina acumulada por faixa angular (base trigonométrica)"):
    laminas_mm = np.array(laminas_mm, dtype=float)
    n_bins = len(laminas_mm)

    # Ângulos dos setores
    theta = np.linspace(0, 2*np.pi, n_bins, endpoint=False)
    width = 2*np.pi / n_bins

    # Normalização
    max_lamina = max(laminas_mm) if np.max(laminas_mm) > 0 else 1.0
    barras_altura = (laminas_mm / max_lamina) * 0.6  # até 60% do raio

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    # Zero no Leste (trigonométrico)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    # Anel interno
    raio_interno = 0.35
    altura_anel = 0.25
    ax.bar(theta, altura_anel, width=width, bottom=raio_interno,
           align='edge', edgecolor='white', linewidth=1)

    # Barras externas
    ax.bar(theta, barras_altura, width=width*0.9, bottom=raio_interno+altura_anel+0.05,
           align='edge', alpha=0.9, edgecolor='black')

    # Labels dos setores
    def faixa_label(i):
        a0 = int(i*setor_size) % 360
        a1 = int((i+1)*setor_size) % 360
        return f"{a0}°–{a1 if a1!=0 else 360}°"

    labels = [faixa_label(i) for i in range(n_bins)]
    ax.set_xticks(theta + width/2)
    ax.set_xticklabels(labels, fontsize=8)

    # Pontos cardeais
    ax.text(0, 1.08, "L", ha='center', va='center', fontsize=12)
    ax.text(np.pi/2, 1.08, "N", ha='center', va='center', fontsize=12)
    ax.text(np.pi, 1.08, "O", ha='center', va='center', fontsize=12)
    ax.text(3*np.pi/2, 1.08, "S", ha='center', va='center', fontsize=12)

    # Texto dentro dos setores
    for i, (t, mm) in enumerate(zip(theta, laminas_mm)):
        ang_centro = t + width/2
        r_text = raio_interno + altura_anel/2
        ax.text(ang_centro, r_text, f"{int(mm)} mm", ha='center', va='center',
                fontsize=9, rotation=np.rad2deg(ang_centro), rotation_mode='anchor')

    ax.set_ylim(0, 1.05)
    ax.set_yticklabels([])

    plt.title(titulo, pad=20)
    plt.tight_layout()
    return fig, ax


# ============================================
# 3. MAIN PIPELINE
# ============================================
if __name__ == "__main__":
    # Configuração
    excel_file = "autoReport_clean.xlsx"
    setor_size = 30
    titulo = f"Distribuição da Lâmina de Água - setores de {setor_size}°"

    # Carregar dados
    df = pd.read_excel(excel_file)
    df = df.dropna(subset=["CurrentAngle"])
    df["CurrentAngle"] = df["CurrentAngle"].astype(float) % 360
    df["Percentimeter"] = pd.to_numeric(df["Percentimeter"], errors="coerce").fillna(0)

    # Criar setores
    bins = np.arange(0, 360+setor_size, setor_size)
    labels = range(len(bins)-1)
    df["setor"] = pd.cut(df["CurrentAngle"], bins=bins, labels=labels,
                         include_lowest=True, right=False)

    # Agregar lâmina
    laminas_mm = df.groupby("setor")["Percentimeter"].sum().reindex(labels, fill_value=0).tolist()
    print(laminas_mm)

    # Gerar **infográfico**
    fig1, ax1 = pivot_infografico_unitcircle(laminas_mm, setor_size=setor_size, titulo=titulo)
    filename_info = f"infografico_{setor_size}_{datetime.now().strftime('%d-%m-%Y--%H-%M-%S')}.png"
    fig1.savefig(filename_info, dpi=200, bbox_inches="tight")
    print("Infográfico salvo em:", filename_info)

    # Gerar **heatmap**
    fig2, ax2 = pivot_heatmap(laminas_mm, setor_size=setor_size, titulo=titulo)
    filename_heat = f"heatmap_{setor_size}_{datetime.now().strftime('%d-%m-%Y--%H-%M-%S')}.png"
    fig2.savefig(filename_heat, dpi=200, bbox_inches="tight")
    print("Heatmap salvo em:", filename_heat)

    plt.show()
