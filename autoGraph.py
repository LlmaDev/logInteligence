import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, datetime

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
# 3. Remover inválidos (655)
# ========================
df = df[(df["InitialAngle"] != 655) & (df["CurrentAngle"] != 655) & (df["Percentimeter"] != 655) & (df["Percentimeter"] != 0)]

# ========================
# 4. Calcular lâmina
# ========================
df["Lamina"] = pivotBlade * 100 / df["Percentimeter"].astype(float)

# ========================
# 5. Criar acumulador por setor
# ========================
acumulado = np.zeros(n)

for _, row in df.iterrows():
    start = int(row["InitialAngle"]) % 360
    end = int(row["CurrentAngle"]) % 360
    lamina = row["Lamina"]

    # Caso arco "passe por 0°" (ex: 350° -> 20°)
    if end < start:
        end += 360

    # Iterar setores e acumular se o arco intercepta
    for s in range(n):
        setor_start = s * setor_size
        setor_end = (s + 1) * setor_size

        # expandir faixa em +360 para cobrir wraps
        if setor_end <= end and setor_start >= start:
            continue

        # Verifica se intervalo [start, end] cruza setor
        if not (end < setor_start or start > setor_end):
            acumulado[s % n] += lamina

# ========================
# 6. Plotar gráfico circular
# ========================
labels = [f"{int(i*setor_size)}°-{int((i+1)*setor_size)}°" for i in range(n)]

plt.figure(figsize=(8, 8))
plt.pie(acumulado, labels=labels, autopct="%.1f%%", startangle=90, counterclock=False)
plt.title("Distribuição da Lâmina de Água por Setores Angulares")
filename = 'autoGraph' + datetime.now().strftime('%d-%m-%Y--%H:%M:%S') + '.png'

# Save the figure with the updated filename
plt.savefig(filename)

