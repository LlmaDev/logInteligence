#!/usr/bin/env python3
import click
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

# ---------------- CONFIGURABLE: pivot blades (editable) ----------------
pivot_blades = {
    "Pivo2": 5.46,
    "Pivo4": 6.18,
    "Pivo13": 6.18,
    "Pivo15": 4.60,
    # adicione outros pivôs aqui conforme necessário
}

# fixed sectors list requested
SECTORS_LIST = [30, 15, 10, 5]


# ---------------- parsing single line ----------------
def parse_message_line(line: str):
    """
    Parse a single log line. Return dict or None.
    Keeps original string fields for later cleaning.
    Filters Command by second digit == '6' (as requested).
    """
    line = line.strip()
    if "->" not in line:
        return None
    try:
        dt_part, details = line.split(" -> ", 1)
        # dt_part normally "21-08-2025 10:19:38"
        # split first space
        dt_be, hour_be = dt_part.split(" ", 1)
        details = details.rstrip("$").lstrip("#")
        parts = details.split("-")
        if len(parts) != 7:
            return None
        Status, farm, Command, percent, init_angle, curr_angle, rtc = parts

        # filter Command: second written digit == "6"
        if not Command.isdigit() or len(Command) < 2 or Command[1] != "6":
            return None

        # Build a base dict
        return {
            "DtBe": dt_be.strip(),
            "HourBe": hour_be.strip(),
            "Status": Status.strip(),
            "FarmName": farm.strip(),
            "Command": Command.strip(),
            "Percentimeter_raw": percent.strip(),
            "InitialAngle_raw": init_angle.strip(),
            "CurrentAngle_raw": curr_angle.strip(),
            "RTC": rtc.strip()
        }
    except Exception:
        return None


# ---------------- gather files and parse in order ----------------
def parse_all_logs(root: Path, pivots: list[str]) -> pd.DataFrame:
    """
    Walk root/pivot/*/*/MESSAGE.txt, parse lines in chronological file order.
    Return DataFrame with parsed rows and a combined timestamp for sorting.
    Also applies the Percentimeter==0 handling using sequential logic per pivot:
      - if Percentimeter is 0/invalid AND CurrentAngle didn't change vs last -> discard
      - if Percentimeter is 0/invalid AND CurrentAngle changed -> use last_percentimeter (if exists)
    """
    all_rows = []

    for pivot in pivots:
        pivot_path = root / pivot
        if not pivot_path.exists():
            print(f"[warn] pivot folder not found: {pivot_path} (skipping)")
            continue

        # collect matching MESSAGE.txt paths under pivot/*/*/MESSAGE.txt
        msg_paths = sorted(pivot_path.glob("*/*/MESSAGE.txt"))

        last_percent = None
        last_current_angle = None

        for p in msg_paths:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for line in text.splitlines():
                parsed = parse_message_line(line)
                if not parsed:
                    continue

                # try to parse datetime from DtBe + HourBe, fallback to file mtime
                try:
                    ts = datetime.strptime(parsed["DtBe"] + " " + parsed["HourBe"], "%d-%m-%Y %H:%M:%S")
                except Exception:
                    ts = datetime.fromtimestamp(p.stat().st_mtime)

                # coerce numeric fields carefully
                # clean percent
                try:
                    percent_val = float(parsed["Percentimeter_raw"])
                except Exception:
                    percent_val = np.nan

                try:
                    curr_angle_val = float(parsed["CurrentAngle_raw"]) % 360
                except Exception:
                    curr_angle_val = np.nan

                try:
                    init_angle_val = float(parsed["InitialAngle_raw"]) % 360
                except Exception:
                    init_angle_val = np.nan

                # Apply percent==0 / nan rule:
                # if percent == 0 or nan:
                #   - if last_current_angle is None: we don't have context -> skip line
                #   - if curr_angle_val == last_current_angle (no movement) -> discard
                #   - else (movement) and last_percent available -> use last_percent
                use_percent = percent_val
                if np.isnan(percent_val) or percent_val == 0:
                    if last_percent is None:
                        # no previous percent to fallback, drop this line
                        continue
                    # if current angle is NaN -> can't decide -> drop
                    if np.isnan(curr_angle_val):
                        continue
                    # compare with last_current_angle (if last_current_angle==None then drop)
                    if last_current_angle is None:
                        continue
                    if np.isclose(curr_angle_val, last_current_angle, atol=1e-6):
                        # no movement -> discard
                        continue
                    else:
                        # movement -> use last_percent
                        use_percent = last_percent
                else:
                    # valid percent -> update last_percent
                    last_percent = percent_val

                # update last_current_angle if curr angle valid
                if not np.isnan(curr_angle_val):
                    last_current_angle = curr_angle_val

                row = {
                    "Timestamp": ts,
                    "DtBe": parsed["DtBe"],
                    "HourBe": parsed["HourBe"],
                    "Status": parsed["Status"],
                    "FarmName": parsed["FarmName"],
                    "Command": parsed["Command"],
                    "Percentimeter": float(use_percent),
                    "InitialAngle": float(init_angle_val) if not np.isnan(init_angle_val) else np.nan,
                    "CurrentAngle": float(curr_angle_val) if not np.isnan(curr_angle_val) else np.nan,
                    "RTC": parsed["RTC"],
                    "Pivot": pivot
                }
                all_rows.append(row)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    # sort by timestamp to preserve chronological order across files
    df.sort_values("Timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------- calc lamina per pivot (vectorized) ----------------
def calculate_lamina(df: pd.DataFrame, blades: dict) -> pd.DataFrame:
    """
    Adds 'Lamina' column to df using pivot_blades mapping.
    Drops rows for which pivot has no blade configured.
    Lamina = pivotBlade * 100 / Percentimeter
    """
    df = df.copy()
    lamina_vals = []
    keep_mask = []

    for pivot_name, group in df.groupby("Pivot"):
        blade = blades.get(pivot_name)
        if blade is None:
            print(f"[warn] no pivotBlade for {pivot_name}, dropping its rows")
            df.drop(df[df["Pivot"] == pivot_name].index, inplace=True)
            continue
        # compute lamina safely
        mask = df["Pivot"] == pivot_name
        pvals = df.loc[mask, "Percentimeter"].astype(float)
        # avoid division by zero (shouldn't happen due to earlier handling) but protect anyway
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = (float(blade) * 100.0) / pvals.replace(0, np.nan)
        df.loc[mask, "Lamina"] = lam
    # drop potential infinite or NaN lamina
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Lamina", "CurrentAngle"])
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------- aggregation & interpolation (unchanged) ----------------
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
        return np.interp(new_deg, x_ext, new_deg)  # fallback linear


def aggregate_by_bins(df, bin_deg, value_col="Percentimeter"):
    """
    Sum value_col by angular bins of width bin_deg.
    Returns: centers_deg (float array), sums (float array)
    """
    df = df.dropna(subset=["CurrentAngle"]).copy()
    df["CurrentAngle"] = pd.to_numeric(df["CurrentAngle"], errors="coerce") % 360
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)
    edges = np.arange(0, 360 + bin_deg, bin_deg)
    labels = edges[:-1]
    idx = pd.cut(df["CurrentAngle"], bins=edges, labels=labels,
                 include_lowest=True, right=False)
    sums = df.groupby(idx)[value_col].sum().reindex(labels, fill_value=0).values
    centers = (labels + bin_deg / 2.0) % 360
    return centers.astype(float), sums.astype(float)


def interpolate_to_1deg(centers_deg, values):
    new_deg = np.arange(0, 360, 1.0)
    # ensure sorting
    order = np.argsort(centers_deg)
    try:
        smoothed = _try_cubic_periodic(centers_deg[order], values[order], new_deg)
    except Exception:
        smoothed = np.interp(new_deg, centers_deg[order], values[order])
    return new_deg.astype(int), np.clip(smoothed, 0, None)


# ---------------- plotting (slight fixes: uses defined cmap) ----------------
def pivot_heatmap(laminas_mm, titulo="Distribuição da Lâmina - Heatmap"):
    laminas_mm = np.asarray(laminas_mm, dtype=float)
    if laminas_mm.shape[0] != 360:
        raise ValueError("pivot_heatmap expects 360 values (one per degree).")
    # normalize to 0-1
    mn, mx = np.min(laminas_mm), np.max(laminas_mm)
    if np.isclose(mx, mn):
        laminas_norm = np.zeros_like(laminas_mm)
    else:
        laminas_norm = (laminas_mm - mn) / (mx - mn)
    angles = np.deg2rad(np.arange(0, 360))
    n_rings = 100
    radii = np.linspace(0, 1.0, n_rings)
    theta, r = np.meshgrid(angles, radii)
    z = np.tile(laminas_norm, (n_rings, 1))
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    cmap = LinearSegmentedColormap.from_list("yl_to_bl",
                                            ["#f7fcb9", "#c7e9b4", "#7fcdbb", "#41b6c4", "#2c7fb8", "#253494"])
    im = ax.pcolormesh(theta, r, z, cmap=cmap, shading='auto', vmin=0, vmax=1)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_title(titulo, va='bottom', fontsize=14)
    cbar = fig.colorbar(im, ax=ax, pad=0.1)
    cbar.set_label('Irrigação Normalizada (0–1)', rotation=270, labelpad=20)
    return fig, ax


def pivot_infografico_unitcircle(laminas_mm, setor_size=30,
                                 titulo="Lâmina acumulada por faixa angular"):
    laminas_mm = np.array(laminas_mm, dtype=float)
    n_bins = len(laminas_mm)
    theta = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
    width = 2 * np.pi / n_bins
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
    ax.bar(theta, barras_altura, width=width * 0.9,
           bottom=raio_interno + altura_anel + 0.05,
           align='edge', alpha=0.9, edgecolor='black')
    def faixa_label(i):
        a0 = int(i * setor_size) % 360
        a1 = int((i + 1) * setor_size) % 360
        return f"{a0}°–{a1 if a1 != 0 else 360}°"
    labels = [faixa_label(i) for i in range(n_bins)]
    ax.set_xticks(theta + width / 2)
    ax.set_xticklabels(labels, fontsize=8)
    # Cardinal points
    ax.text(0, 1.05, "L", ha='center'); ax.text(np.pi / 2, 1.05, "N", ha='center')
    ax.text(np.pi, 1.05, "O", ha='center'); ax.text(3 * np.pi / 2, 1.05, "S", ha='center')
    for t, mm in zip(theta, laminas_mm):
        ang_centro = t + width / 2
        r_text = raio_interno + altura_anel / 2
        ax.text(ang_centro, r_text, f"{int(mm)} mm", ha='center',
                va='center', fontsize=9, rotation=np.rad2deg(ang_centro),
                rotation_mode='anchor')
    ax.set_ylim(0, 1.05)
    ax.set_yticklabels([])
    plt.title(titulo, pad=20)
    plt.tight_layout()
    return fig, ax


# ---------------- generate graphs per pivot ----------------
def generate_graphs_for_pivot(df_pivot: pd.DataFrame, pivot_name: str, out_dir: Path):
    """
    Given parsed+lamina DataFrame already filtered for pivot, generate:
      - infografico for each sector in SECTORS_LIST (aggregating using Percentimeter or Lamina?)
      - heatmap 1° based on Lamina aggregated to 1°
    We'll aggregate by Percentimeter -> convert to Lamina sum per angular bin.
    The plotting functions expect arrays of accumulated lamina per degree/bin.
    """
    timestamp = datetime.now().strftime("%d-%m-%Y--%H-%M-%S")
    # ensure out_dir exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV verification
    csv_path = out_dir / f"parsed_{pivot_name}_{timestamp}.csv"
    df_pivot.to_csv(csv_path, index=False)
    print(f"[info] parsed CSV saved: {csv_path}")

    # We will aggregate using Lamina (mm) per record.
    df_pivot = df_pivot.dropna(subset=["Lamina", "CurrentAngle"]).copy()

    # For each sector size -> infografico
    for setor in SECTORS_LIST:
        centers, sums = aggregate_by_bins(df_pivot, setor, value_col="Lamina")
        # sums correspond to accumulated lamina per bin
        fig, ax = pivot_infografico_unitcircle(sums, setor_size=setor,
                                               titulo=f"{pivot_name} - Lâmina acumulada ({setor}°)")
        fname = out_dir / f"infografico_{pivot_name}_{setor}_{timestamp}.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[info] Infográfico salvo em: {fname}")

    # Heatmap 1°: aggregate to 1° bins first (using Lamina)
    centers1, sums1 = aggregate_by_bins(df_pivot, 1, value_col="Lamina")
    # ensure length 360
    if sums1.shape[0] != 360:
        # if aggregate_by_bins returns centers of size 360 that's fine; else reindex
        # but aggregate_by_bins with bin_deg=1 should return 360 values
        sums1 = np.resize(sums1, 360)
    fig2, ax2 = pivot_heatmap(sums1, titulo=f"{pivot_name} - Heatmap 1°")
    fname2 = out_dir / f"heatMap_{pivot_name}_{timestamp}.png"
    fig2.savefig(fname2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"[info] Heatmap salvo em: {fname2}")


# ---------------- CLI ----------------
@click.command()
@click.option("--root", default="./resources/logs", type=click.Path(path_type=Path),
              help="Raiz das pastas de logs (ex: ./resources/logs)")
@click.option("--pivots", multiple=True, default=tuple(pivot_blades.keys()),
              help="Lista de pivots para processar (ex: --pivots Pivo2 Pivo4)")
@click.option("--export-csv/--no-export-csv", default=True,
              help="Salvar CSV de parsed por pivot (padrão: salvar)")
def main(root: Path, pivots: list[str], export_csv: bool):
    root = Path(root)
    pivots_list = list(pivots)
    print("[start] Parsing logs from:", root)
    df = parse_all_logs(root, pivots_list)
    if df.empty:
        print("[error] Nenhuma linha válida encontrada. Verifique os logs e o padrão do parser.")
        raise SystemExit(1)

    # calculate lamina using pivot_blades mapping
    df = calculate_lamina(df, pivot_blades)
    if df.empty:
        print("[error] Após cálculo de lâmina não há dados válidos.")
        raise SystemExit(1)

    # process per pivot
    for pivot_name in df["Pivot"].unique():
        df_pivot = df[df["Pivot"] == pivot_name].copy()
        out_dir = Path("./resources/imgs") / pivot_name
        print(f"[info] Processando pivot {pivot_name}: {len(df_pivot)} linhas -> saída em {out_dir}")
        # optionally save CSV (already done inside generate)
        generate_graphs_for_pivot(df_pivot, pivot_name, out_dir)

    print("[done] Todos os pivôs processados.")


if __name__ == "__main__":
    main()
