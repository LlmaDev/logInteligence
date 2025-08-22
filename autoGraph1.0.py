import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ========================
# Parâmetros
# ========================
excel_file = "autoReport.xlsx"
pivotBlade = 4.4  # ajuste conforme necessário
n = 12  # número fixo de setores (ex: 12 setores = 30° cada)
setor_size = 360 / n

# ========================
# 1. Ler dados
# ========================
df = pd.read_excel(excel_file)

# ========================
# 2. Filtrar por DWP (2º dígito == "6")
# ========================
df = df[df["Command"].astype(str).str[1] == "6"]

# ========================
# 3. Remover inválidos (655, 0)
# ========================
df = df[(df["InitialAngle"] != 655) & (df["CurrentAngle"] != 655) & 
        (df["Percentimeter"] != 655) & (df["Percentimeter"] != 0)]

# ========================
# 4. Calcular lâmina (com tratamento de erros)
# ========================
# Converte para numérico, forçando inválidos para NaN
df["Percentimeter"] = pd.to_numeric(df["Percentimeter"], errors="coerce")

# Remove linhas com valores inválidos
df = df.dropna(subset=["Percentimeter"])

# Evita divisão por zero
df = df[df["Percentimeter"] != 0]

# Calcula a lâmina
df["Lamina"] = pivotBlade * 100 / df["Percentimeter"]

# ========================
# 5. Criar acumulador por setor
# ========================
acumulado = np.zeros(n)

for _, row in df.iterrows():
    start = int(row["InitialAngle"]) % 360
    end = int(row["CurrentAngle"]) % 360
    lamina = row["Lamina"]

    if end < start:  # arco que passa pelo 0°
        end += 360

    for s in range(n):
        setor_start = s * setor_size
        setor_end = (s + 1) * setor_size

        # Verifica se intervalo [start, end] cruza setor
        if not (end < setor_start or start > setor_end):
            acumulado[s % n] += lamina

# ========================
# 6. Plotar gráfico no formato híbrido (anel + barras)
# ========================
theta = np.linspace(0, 2*np.pi, n, endpoint=False)
width = 2*np.pi / n

max_lamina = max(acumulado) if np.max(acumulado) > 0 else 1.0
barras_altura = (acumulado / max_lamina) * 0.6  # 60% do raio externo

fig = plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

# Ciclo trigonométrico: 0° no Leste, CCW
ax.set_theta_zero_location("E")
ax.set_theta_direction(1)

# Anel interno (pizza de setores iguais)
raio_interno = 0.35
altura_anel = 0.25
ax.bar(theta, altura_anel, width=width, bottom=raio_interno,
       align='edge', edgecolor='black', linewidth=1)

# Barras externas (acumulado)
cmap = plt.cm.viridis
colors = (acumulado - acumulado.min()) / (acumulado.max() - acumulado.min())
ax.bar(theta, 0.6, width=width, bottom=0.4, color=cmap(colors))

# Labels das faixas angulares
labels = [f"{int(i*setor_size)}°-{int((i+1)*setor_size)}°" for i in range(n)]

#for i, (t, mm) in enumerate(zip(theta, acumulado)):
#    ang_centro = t + width/2
 #   r_text = raio_interno + altura_anel/2
  #  ax.text(ang_centro, r_text, f"{int(mm)} mm", ha='center', va='center',
   #         fontsize=9, rotation=np.rad2deg(ang_centro), rotation_mode='anchor')

# Marcas cardeais
ax.text(0, 1.08, "L", ha='center', va='center', fontsize=12)
ax.text(np.pi/2, 1.08, "N", ha='center', va='center', fontsize=12)
ax.text(np.pi, 1.08, "O", ha='center', va='center', fontsize=12)
ax.text(3*np.pi/2, 1.08, "S", ha='center', va='center', fontsize=12)

# Ajustes visuais
ax.set_ylim(0, 1.05)
ax.set_yticklabels([])
##ax.set_xticks(theta + width/2)
##ax.set_xticklabels(labels, fontsize=8)

plt.title("Distribuição da Lâmina de Água GrupoBB_1", pad=20)
plt.tight_layout()

# ========================
# 7. Salvar com timestamp
# ========================
filename = 'autoGraph' + str(setor_size) + datetime.now().strftime('%d-%m-%Y--%H:%M:%S') + '.png'
plt.savefig(filename, dpi=200, bbox_inches="tight")