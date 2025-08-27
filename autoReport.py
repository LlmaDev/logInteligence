import click
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------- parser from before ----------------
def parse_message_line(line: str):
    line = line.strip()
    if "->" not in line:
        return None
    try:
        dt_part, details = line.split(" -> ", 1)
        dt_be, hour_be = dt_part.split(" ", 1)
        details = details.rstrip("$").lstrip("#")
        parts = details.split("-")
        if len(parts) != 7:
            return None
        Status, farm, Command, percent, init_angle, curr_angle, rtc = parts
        if not Command.isdigit() or len(Command) < 2 or Command[1] != "6":
            return None
        return {
            "DtBe": dt_be.strip(),
            "HourBe": hour_be.strip(),
            "Status": Status.strip(),
            "FarmName": farm.strip(),
            "Command": int(Command),
            "Percentimeter": percent.strip(),
            "InitialAngle": init_angle.strip(),
            "CurrentAngle": curr_angle.strip(),
            "RTC": rtc.strip()
        }
    except Exception:
        return None

def parse_all_logs(root: Path, pivots: list[str]) -> pd.DataFrame:
    rows = []
    for pivot in pivots:
        for msgfile in root.glob(f"{pivot}/*/*/MESSAGE.txt"):
            for line in msgfile.read_text(encoding="utf-8", errors="ignore").splitlines():
                parsed = parse_message_line(line)
                if parsed:
                    parsed["Pivot"] = pivot
                    rows.append(parsed)
    return pd.DataFrame(rows)

# ---------------- your helpers unchanged ----------------
def _try_cubic_periodic(x_deg, y, new_deg):
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
    df = df.dropna(subset=["CurrentAngle"]).copy()
    df["CurrentAngle"] = pd.to_numeric(df["CurrentAngle"], errors="coerce") % 360
    df["Percentimeter"] = pd.to_numeric(df["Percentimeter"], errors="coerce").fillna(0)
    edges = np.arange(0, 360 + bin_deg, bin_deg)
    labels = edges[:-1]
    idx = pd.cut(df["CurrentAngle"], bins=edges, labels=labels,
                 include_lowest=True, right=False)
    sums = df.groupby(idx)["Percentimeter"].sum().reindex(labels, fill_value=0).values
    centers = (labels + bin_deg / 2.0) % 360
    return centers.astype(float), sums.astype(float)

def interpolate_to_1deg(centers_deg, values):
    new_deg = np.arange(0, 360, 1.0)
    smoothed = _try_cubic_periodic(np.sort(centers_deg), values[np.argsort(centers_deg)], new_deg)
    return new_deg.astype(int), np.clip(smoothed, 0, None)

def pivot_heatmap(laminas_mm, titulo="Distribuição da Lâmina - Heatmap"):
    laminas_mm = np.asarray(laminas_mm, dtype=float)
    laminas_norm = (laminas_mm - np.min(laminas_mm))/(np.max(laminas_mm) - np.min(laminas_mm))
    angles = np.deg2rad(np.arange(0, 360))
    n_rings = 100
    radii = np.linspace(0, 1.0, n_rings)
    theta, r = np.meshgrid(angles, radii)
    z = np.tile(laminas_norm, (n_rings, 1))
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    cmap = LinearSegmentedColormap.from_list("yl_to_bl",
        ["#f7fcb9","#c7e9b4","#7fcdbb","#41b6c4","#2c7fb8","#253494"])
    im = ax.pcolormesh(theta, r, z, cmap='viridis_r', shading='auto', vmin=0, vmax=1)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.grid(False); ax.set_yticklabels([])
    ax.set_title(titulo, va='bottom', fontsize=14)
    cbar = fig.colorbar(im, ax=ax, pad=0.1)
    cbar.set_label('Irrigação Normalizada (0–1)', rotation=270, labelpad=20)
    return fig, ax

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
    ax.set_theta_zero_location("E"); ax.set_theta_direction(1)
    raio_interno = 0.35; altura_anel = 0.25
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
    ax.set_xticks(theta + width/2); ax.set_xticklabels(labels, fontsize=8)
    ax.text(0, 1.05, "L", ha='center'); ax.text(np.pi/2, 1.05, "N", ha='center')
    ax.text(np.pi, 1.05, "O", ha='center'); ax.text(3*np.pi/2, 1.05, "S", ha='center')
    for t, mm in zip(theta, laminas_mm):
        ang_centro = t + width/2; r_text = raio_interno + altura_anel/2
        ax.text(ang_centro, r_text, f"{int(mm)} mm", ha='center',
                va='center', fontsize=9, rotation=np.rad2deg(ang_centro),
                rotation_mode='anchor')
    ax.set_ylim(0, 1.05); ax.set_yticklabels([])
    plt.title(titulo, pad=20); plt.tight_layout()
    return fig, ax

# ---------------- CLI ----------------
@click.command()
@click.option("--root", default="./resources/logs", type=click.Path(path_type=Path))
@click.option("--pivots", multiple=True, default=["Pivo2"], help="Which pivots to parse")
@click.option("--export-csv", type=click.Path(path_type=Path), help="Save parsed data to CSV")
@click.option("--setor-size", default=30, show_default=True, help="Bin size in degrees")
def main(root: Path, pivots: list[str], export_csv: Path, setor_size: int):
    df = parse_all_logs(root, pivots)
    if df.empty:
        raise SystemExit("No valid rows parsed.")
    if export_csv:
        df.to_csv(export_csv, index=False)
        print("Parsed data saved to:", export_csv)

    centers30, sums30 = aggregate_by_bins(df, setor_size)
    degs1, sums1 = interpolate_to_1deg(centers30, sums30)

    fig1, _ = pivot_infografico_unitcircle(
        sums30, setor_size=setor_size,
        titulo=f"Distribuição da Lâmina de Água - setores de {setor_size}°")
    fname1 = f"infografico_{setor_size}_{datetime.now().strftime('%d-%m-%Y--%H-%M-%S')}.png"
    fig1.savefig(fname1, dpi=200, bbox_inches="tight")
    print("Infográfico salvo em:", fname1)

    fig2, _ = pivot_heatmap(sums1, titulo="Distribuição da Lâmina de Água - setores de 1°")
    fname2 = f"heatmap_1_{datetime.now().strftime('%d-%m-%Y--%H-%M-%S')}.png"
    fig2.savefig(fname2, dpi=200, bbox_inches="tight")
    print("Heatmap salvo em:", fname2)

    plt.show()

if __name__ == "__main__":
    main()
