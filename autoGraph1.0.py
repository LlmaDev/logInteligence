import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

def pivot_heatmap(laminas_mm, setor_size=30, titulo="Heatmap de Lâmina de Água (polar)"):
    laminas_mm = np.array(laminas_mm, dtype=float)
    n_bins = len(laminas_mm)

    # Criar matriz radial (r, theta)
    theta_edges = np.linspace(0, 2*np.pi, n_bins+1)
    r_edges = np.linspace(0, 1, 100)  # resolução radial do heatmap

    # Expandir para grid
    Theta, R = np.meshgrid(theta_edges, r_edges)

    # Criar valores (copiar cada setor ao longo de r)
    Z = np.zeros_like(Theta)
    for i in range(n_bins):
        mask = (Theta >= theta_edges[i]) & (Theta < theta_edges[i+1])
        Z[mask] = laminas_mm[i]

    # Figura
    fig, ax = plt.subplots(subplot_kw={"projection":"polar"}, figsize=(8,8))
    c = ax.pcolormesh(Theta, R, Z, cmap="viridis")

    # Configurações de orientação
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    # Ticks cardeais
    ax.text(0, 1.05, "L", ha='center', va='center', fontsize=12)
    ax.text(np.pi/2, 1.05, "N", ha='center', va='center', fontsize=12)
    ax.text(np.pi, 1.05, "O", ha='center', va='center', fontsize=12)
    ax.text(3*np.pi/2, 1.05, "S", ha='center', va='center', fontsize=12)

    ax.set_yticklabels([])  # remove círculos concêntricos
    plt.title(titulo, pad=20)
    plt.colorbar(c, ax=ax, orientation="horizontal", pad=0.1, fraction=0.05, label="mm")

    plt.tight_layout()
    return fig, ax

def pivot_infografico_unitcircle(laminas_mm, setor_size=30, titulo="Lâmina acumulada por faixa angular (base trigonométrica)"):
    laminas_mm = np.array(laminas_mm, dtype=float)
    n_bins = len(laminas_mm)
    
    # Ângulos dos setores (centros)
    theta = np.linspace(0, 2*np.pi, n_bins, endpoint=False)
    width = 2*np.pi / n_bins
    
    # Normalização para altura das barras externas
    max_lamina = max(laminas_mm) if np.max(laminas_mm) > 0 else 1.0
    barras_altura = (laminas_mm / max_lamina) * 0.6  # 60% do raio
    
    # Figura polar
    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(111, polar=True)
    
    # Zero em Leste e sentido anti-horário (trigonométrico)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    
    # Anel interno
    raio_interno = 0.35
    altura_anel = 0.25
    ax.bar(theta, altura_anel, width=width, bottom=raio_interno, align='edge',
           edgecolor='white', linewidth=1)
    
    # Barras externas
    ax.bar(theta, barras_altura, width=width*0.9, bottom=raio_interno+altura_anel+0.05,
           align='edge', alpha=0.9, edgecolor='black')
    
    # Labels de faixa
    def faixa_label(i):
        a0 = int(i*setor_size) % 360
        a1 = int((i+1)*setor_size) % 360
        return f"{a0}°–{a1 if a1!=0 else 360}°"
    
    labels = [faixa_label(i) for i in range(n_bins)]
    
    # Anotações no anel interno
    for i, (t, mm) in enumerate(zip(theta, laminas_mm)):
        ang_centro = t + width/2
        r_text = raio_interno + altura_anel/2
        ax.text(ang_centro, r_text, f"{int(mm)} mm", ha='center', va='center',
                fontsize=9, rotation=np.rad2deg(ang_centro), rotation_mode='anchor')
    
    # Pontos cardeais
    ax.text(0, 1.08, "L", ha='center', va='center', fontsize=12)
    ax.text(np.pi/2, 1.08, "N", ha='center', va='center', fontsize=12)
    ax.text(np.pi, 1.08, "O", ha='center', va='center', fontsize=12)
    ax.text(3*np.pi/2, 1.08, "S", ha='center', va='center', fontsize=12)
    
    # Ajustes
    ax.set_ylim(0, 1.05)
    ax.set_yticklabels([])
    ax.set_xticks(theta + width/2)
    ax.set_xticklabels(labels, fontsize=8)
    
    plt.title(titulo, pad=20)
    plt.tight_layout()
    return fig, ax


# === MAIN PIPELINE ===
# Configuração
excel_file = "autoReport_clean.xlsx"
setor_size = 30  # graus
titulo = "Distribuição da Lâmina de Água - setores de {}°".format(setor_size)

# Carregar dados
df = pd.read_excel(excel_file)

# Assegurar colunas (ajuste se necessário)
# Supondo que a coluna "CurrentAngle" está em graus e "Percentimeter" ou "Value" em mm
df = df.dropna(subset=["CurrentAngle"])  # descartar linhas sem ângulo
df["CurrentAngle"] = df["CurrentAngle"].astype(float) % 360
df["Percentimeter"] = pd.to_numeric(df["Percentimeter"], errors="coerce").fillna(0)

# Criar setores fixos
bins = np.arange(0, 360+setor_size, setor_size)
labels = range(len(bins)-1)
df["setor"] = pd.cut(df["CurrentAngle"], bins=bins, labels=labels, include_lowest=True, right=False)

# Agregar lâmina por setor
laminas_mm = df.groupby("setor")["Percentimeter"].sum().reindex(labels, fill_value=0).tolist()

# Gerar gráfico
fig, ax = pivot_infografico_unitcircle(laminas_mm, setor_size=setor_size, titulo=titulo)

# Nome dinâmico do arquivo
filename = 'autoGraphAgroCangaia_2' + str(setor_size) + datetime.now().strftime('%d-%m-%Y--%H:%M:%S') + '.png'
plt.savefig(filename, dpi=200, bbox_inches="tight")
plt.show()

# Reuso dos dados já calculados
fig, ax = pivot_heatmap(laminas_mm, setor_size=setor_size,
                        titulo=f"Distribuição da Lâmina - Heatmap {setor_size}°")

# Salvar com mesmo padrão de nome
filename_heatmap = 'autoHeatmapAgroCangaia_2' + str(setor_size) + datetime.now().strftime('%d-%m-%Y--%H:%M:%S') + '.png'
plt.savefig(filename_heatmap, dpi=200, bbox_inches="tight")
plt.show()

print("Heatmap salvo em:", filename_heatmap)
print("Gráfico salvo em:", filename)
