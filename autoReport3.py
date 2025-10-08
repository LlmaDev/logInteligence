'''
Goals:
    - lamina >= pivot_blade exclude
    - verify percent extraction
    - verify percent/lamina conversion
    - generate grpahs from data and BD
    
'''    
#!/usr/bin/env python3
import click
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import Counter
import re
from database import IrrigationDatabase


# ---------------- Core parsing functions ----------------
def parse_message_line(line: str):
    """Parse a single log line into structured data"""
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
        
        return {
            "DtBe": dt_be.strip(),
            "HourBe": hour_be.strip(),
            "Status": Status.strip(),
            "FarmName": farm.strip(),
            "Command": int(Command) if str(Command).isdigit() else Command,
            "Percentimeter": percent.strip(),
            "InitialAngle": init_angle.strip(),
            "CurrentAngle": curr_angle.strip(),
            "RTC": rtc.strip()
        }
    except Exception:
        return None

def parse_all_logs(root: Path, pivots: list[str]) -> pd.DataFrame:
    """Parse all MESSAGE.txt files for given pivots"""
    rows = []
    for pivot in pivots:
        for msgfile in sorted(root.glob(f"{pivot}/*/*/MESSAGE.txt"), key=lambda p: str(p)):
            for line in msgfile.read_text(encoding="utf-8", errors="ignore").splitlines():
                parsed = parse_message_line(line)
                if parsed:
                    parsed["Pivot"] = pivot
                    rows.append(parsed)
    return pd.DataFrame(rows)

def safe_parse_timestamp(dt_str: str, hour_str: str):
    """Parse timestamp with fallbacks"""
    s = f"{dt_str} {hour_str}"
    try:
        return datetime.strptime(s, "%d-%m-%Y %H:%M:%S")
    except Exception:
        try:
            return pd.to_datetime(s, dayfirst=True)
        except Exception:
            return pd.NaT

# ---------------- Cycle detection ----------------
def find_cycles_for_pivot(df_p: pd.DataFrame):
    """
    Enhanced cycle detection with 7-command handling, 655 filtering, and duration filtering.
    
    Rules:
    - Cycle starts when 2nd digit = '6', ends when 2nd digit = '5'
    - When encountering 2nd digit = '7' during cycle, look ahead through all consecutive 7s
        - If first non-7 command matches last command before 7s → continue cycle
        - If different → break into two cycles
    - Stop command (2nd digit = '5') is NOT included in cycle data
    - 655 values in percentimeter are filtered out and replaced with cycle median
    - Cycles shorter than 5 minutes are discarded
    """
    cycles = []
    in_cycle = False
    start_idx = None
    percent_values = []
    direction = None
    start_angle = None
    last_valid_angle = None
    indices_in_cycle = []
    last_command_before_7 = None

    def _digits_of_command(val):
        s = "" if val is None else str(val)
        return re.sub(r"\D", "", s)

    def _get_command_value(val):
        """Get numeric command value for comparison"""
        try:
            return int(val) if str(val).isdigit() else None
        except:
            return None

    def _get_filtered_percentimeter(val):
        """Filter out 655 values, return None for replacement later"""
        try:
            pval = float(val or 0.0)
            return None if pval == 655.0 else pval
        except:
            return 0.0

    def _replace_655_with_median(percent_list):
        """Replace None values (originally 655) with median of valid values"""
        valid_values = [p for p in percent_list if p is not None]
        if not valid_values:
            return [0.0] * len(percent_list)  # fallback if all were 655
        
        median_val = np.median(valid_values)
        return [median_val if p is None else p for p in percent_list]

    def _get_cycle_duration_minutes(start_idx, end_idx):
        """Calculate cycle duration in minutes using timestamps"""
        try:
            start_row = df_p.loc[start_idx]
            end_row = df_p.loc[end_idx]
            
            start_ts = safe_parse_timestamp(start_row.get("DtBe", ""), start_row.get("HourBe", ""))
            end_ts = safe_parse_timestamp(end_row.get("DtBe", ""), end_row.get("HourBe", ""))
            
            if pd.isna(start_ts) or pd.isna(end_ts):
                return 0  # Invalid timestamps - will be filtered out
            
            duration = (end_ts - start_ts).total_seconds() / 60.0  # Convert to minutes
            return max(0, duration)  # Ensure non-negative
        except Exception:
            return 0  # Error calculating duration - will be filtered out

    def _save_cycle_if_valid(start_idx, end_idx, start_angle, end_angle, direction, percent_values, indices_in_cycle, warning=None):
        """Save cycle only if it meets duration requirements"""
        if not indices_in_cycle:
            return False
        
        # Calculate duration
        duration_min = _get_cycle_duration_minutes(start_idx, end_idx)
        
        # Filter out cycles shorter than 5 minutes
        if duration_min < 5.0:
            print(f"[DEBUG] Discarding cycle {start_idx}->{end_idx}: duration {duration_min:.1f} min < 5 min")
            return False
        
        # Clean percentimeter values
        percent_values_clean = _replace_655_with_median(percent_values)
        
        cycle_data = {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "start_angle": start_angle if start_angle is not None else last_valid_angle,
            "end_angle": end_angle,
            "direction": direction,
            "percent_list": percent_values_clean,
            "indices": indices_in_cycle.copy(),
            "duration_minutes": duration_min
        }
        
        if warning:
            cycle_data["warning"] = warning
            
        cycles.append(cycle_data)
        print(f"[DEBUG] Saved cycle {start_idx}->{end_idx}: duration {duration_min:.1f} min")
        return True

    # Convert to list for easier lookahead
    rows = list(df_p.itertuples())
    
    i = 0
    while i < len(rows):
        row = rows[i]
        orig_idx = row.Index
        cmd_digits = _digits_of_command(getattr(row, "Command", "") or "")
        first_digit = cmd_digits[0] if len(cmd_digits) >= 1 else None
        second_digit = cmd_digits[1] if len(cmd_digits) >= 2 else None
        current_command = _get_command_value(getattr(row, "Command", ""))

        # Filter percentimeter (None if 655, actual value otherwise)
        pval = _get_filtered_percentimeter(getattr(row, "Percentimeter", 0.0))

        try:
            ang = int(float(getattr(row, "CurrentAngle", None)))
            # Also filter 655 from angles
            if ang == 655:
                ang = None
            else:
                last_valid_angle = ang
        except Exception:
            ang = None

        # START condition
        if (not in_cycle) and (second_digit == "6"):
            in_cycle = True
            start_idx = orig_idx
            direction = first_digit if first_digit is not None else "3"
            start_angle = ang if ang is not None else last_valid_angle
            percent_values = [pval]
            indices_in_cycle = [orig_idx]
            last_command_before_7 = current_command
            i += 1
            continue

        # CLOSE condition - BEFORE processing the row data
        if in_cycle and (second_digit == "5"):
            # End the cycle at the PREVIOUS row (don't include this stop command)
            end_idx = indices_in_cycle[-1] if indices_in_cycle else start_idx
            end_angle = last_valid_angle
            
            # Save cycle only if it meets duration requirements
            _save_cycle_if_valid(start_idx, end_idx, start_angle, end_angle, direction, percent_values, indices_in_cycle)

            # Reset state
            in_cycle = False
            start_idx = None
            percent_values = []
            direction = None
            start_angle = None
            indices_in_cycle = []
            last_command_before_7 = None
            i += 1
            continue

        # HANDLE 7-COMMAND SEQUENCES INSIDE CYCLE
        if in_cycle and (second_digit == "7"):
            # Look ahead through all consecutive 7s
            j = i
            seven_indices = []
            seven_percent_values = []
            
            # Collect all consecutive 7-commands
            while j < len(rows):
                current_row = rows[j]
                current_cmd_digits = _digits_of_command(getattr(current_row, "Command", "") or "")
                current_second_digit = current_cmd_digits[1] if len(current_cmd_digits) >= 2 else None
                
                if current_second_digit == "7":
                    seven_indices.append(current_row.Index)
                    seven_pval = _get_filtered_percentimeter(getattr(current_row, "Percentimeter", 0.0))
                    seven_percent_values.append(seven_pval)
                    
                    # Update angle if valid (filter 655)
                    try:
                        seven_ang = int(float(getattr(current_row, "CurrentAngle", None)))
                        if seven_ang != 655 and seven_ang is not None:
                            last_valid_angle = seven_ang
                    except:
                        pass
                    
                    j += 1
                else:
                    break  # Found first non-7 command
            
            # Check what comes after the 7s
            if j < len(rows):
                next_row = rows[j]
                next_command = _get_command_value(getattr(next_row, "Command", ""))
                
                # Compare with last command before 7s
                if next_command == last_command_before_7:
                    # CONTINUE CYCLE: Include 7s and continue
                    percent_values.extend(seven_percent_values)
                    indices_in_cycle.extend(seven_indices)
                    # Don't update last_command_before_7 - keep the original
                    i = j  # Move to the first non-7 command to process it normally
                    continue
                else:
                    # BREAK CYCLE: End current cycle before 7s, start new cycle after 7s
                    
                    # End current cycle (before the 7s) - check duration
                    end_idx = indices_in_cycle[-1] if indices_in_cycle else start_idx
                    _save_cycle_if_valid(start_idx, end_idx, start_angle, last_valid_angle, direction, percent_values, indices_in_cycle)
                    
                    # Start new cycle from first non-7 command
                    next_row = rows[j]
                    next_cmd_digits = _digits_of_command(getattr(next_row, "Command", "") or "")
                    next_first_digit = next_cmd_digits[0] if len(next_cmd_digits) >= 1 else None
                    
                    try:
                        next_ang = int(float(getattr(next_row, "CurrentAngle", None)))
                        if next_ang != 655 and next_ang is not None:
                            last_valid_angle = next_ang
                    except:
                        pass
                    
                    next_pval = _get_filtered_percentimeter(getattr(next_row, "Percentimeter", 0.0))
                    
                    # Reset for new cycle
                    start_idx = next_row.Index
                    direction = next_first_digit if next_first_digit is not None else direction
                    start_angle = last_valid_angle
                    percent_values = [next_pval]
                    indices_in_cycle = [next_row.Index]
                    last_command_before_7 = next_command
                    
                    i = j + 1  # Skip the first non-7 command since we already processed it
                    continue
            else:
                # Reached end of file during 7s - include them in current cycle
                percent_values.extend(seven_percent_values)
                indices_in_cycle.extend(seven_indices)
                i = j  # This will end the while loop
                continue

        # NORMAL PROCESSING INSIDE CYCLE
        if in_cycle:
            percent_values.append(pval)
            indices_in_cycle.append(orig_idx)
            if ang is not None:
                last_valid_angle = ang
            
            # Update last_command_before_7 for non-7 commands
            if second_digit != "7":
                last_command_before_7 = current_command

        i += 1

    # Handle EOF while in cycle - check duration
    if in_cycle:
        end_idx = indices_in_cycle[-1] if indices_in_cycle else None
        if end_idx:
            _save_cycle_if_valid(start_idx, end_idx, start_angle, last_valid_angle, direction, percent_values, indices_in_cycle, warning="closed_on_eof")

    return cycles

def _degrees_range_inclusive(start: int, stop: int, direction: str):
    """Generate degree sequence based on direction"""
    start = int(start) % 360
    stop = int(stop) % 360
    seq = []
    if direction == "3":  # forward => decreasing
        cur = start
        seq.append(cur)
        while cur != stop:
            cur = (cur - 1) % 360
            seq.append(cur)
            if len(seq) > 720:  # safety
                break
    else:  # reverse => increasing
        cur = start
        seq.append(cur)
        while cur != stop:
            cur = (cur + 1) % 360
            seq.append(cur)
            if len(seq) > 720:
                break
    return seq



# ---------------- Data processing ----------------

def _get_degrees_between_angles(start_angle: int, end_angle: int, direction: str):
    """Get all degrees from start_angle to end_angle following direction"""
    return _degrees_range_inclusive(start_angle, end_angle, direction)

# Update the cycle processing in main() function
def process_cycle_data_correctly(df, pivot_blade):
    """Replacement for the old cycle processing logic"""
    # Make sure we have the required columns as numeric
    df["CurrentAngle"] = pd.to_numeric(df["CurrentAngle"], errors="coerce")
    df["Percentimeter"] = pd.to_numeric(df["Percentimeter"], errors="coerce")
    
    # Process cycles with correct angle-percentimeter mapping
    lamina_360, cycle_rows = process_cycles_to_accumulators(df, pivot_blade)
    
    return lamina_360, cycle_rows

def process_cycles_to_accumulators(df, pivot_blade: float):
    """Process cycles with individual angle-percentimeter mapping"""
    lamina_acc = np.zeros(360, dtype=float)
    cycle_rows = set()
    
    for pivot_val in df["Pivot"].unique():
        df_p = df[df["Pivot"] == pivot_val]
        cycles = find_cycles_for_pivot(df_p)
        print(f"[DEBUG] Pivot '{pivot_val}' -> found {len(cycles)} cycles")

        for c in cycles:
            # Collect cycle row indices
            cycle_rows.update(c.get("indices", []))
            
            # Get actual angle-percentimeter data from the cycle
            angle_percent_data = []
            for idx in c.get("indices", []):
                try:
                    row = df_p.loc[idx]
                    angle = pd.to_numeric(row.get("CurrentAngle", None), errors="coerce")
                    percent = pd.to_numeric(row.get("Percentimeter", None), errors="coerce")
                    
                    # Filter out 655 values and invalid data
                    if pd.notna(angle) and pd.notna(percent) and percent != 655 and angle != 655:
                        angle_percent_data.append((int(angle % 360), float(percent)))
                except Exception:
                    continue
            
            if not angle_percent_data:
                print(f"[WARN] Cycle {c.get('start_idx')} -> {c.get('end_idx')} has no valid angle-percentimeter data")
                continue
            
            # Get cycle direction
            direction = c.get("direction", "3")
            
            # Apply percentimeter to individual angles based on ranges
            for i, (current_angle, current_percent) in enumerate(angle_percent_data):
                
                if i < len(angle_percent_data) - 1:
                    # Get the next angle
                    next_angle = angle_percent_data[i + 1][0]
                    
                    # Get angles that should receive this percentimeter
                    # (from current_angle up to but NOT including next_angle)
                    affected_angles = _get_angles_up_to_next(current_angle, next_angle, direction)
                else:
                    # Last reading - only applies to its own angle
                    affected_angles = [current_angle]
                
                # Calculate lamina for this percentimeter
                if current_percent > 0:
                    lam_per_deg = (pivot_blade * 100.0) / current_percent 
                else:
                    lam_per_deg = 0.0
                
                # Apply lamina to each individual angle
                for angle in affected_angles:
                    lamina_acc[angle] += lam_per_deg
                
                print(f"[DEBUG] {current_angle}° (percent={current_percent}) -> angles {affected_angles[:3]}{'...' if len(affected_angles)>3 else ''} [{len(affected_angles)} total] with lamina {lam_per_deg:.2f}")

    return lamina_acc, cycle_rows

def save_cycles_to_database(df, pivot_name, pivot_blade, db_config=None):
    """
    Salva cada ciclo individualmente no banco de dados
    """
    if db_config is None:
        db_config = {
            'host': 'localhost',
            'database': 'irrigation_db',
            'user': 'postgres',
            'password': 'admin'  
        }
    
    db = IrrigationDatabase(**db_config)
    
    if not db.connect():
        print("[ERRO] Não foi possível conectar ao banco")
        return
    
    try:
        # Processar cada pivô
        for pivot_val in df["Pivot"].unique():
            df_p = df[df["Pivot"] == pivot_val]
            cycles = find_cycles_for_pivot(df_p)
            
            print(f"\n[DB] Salvando {len(cycles)} ciclos para {pivot_val}")
            
            for ci, cycle in enumerate(cycles, start=1):
                # Calcular array de lâmina para este ciclo específico
                lamina_cycle = calculate_cycle_lamina_array(df_p, cycle, pivot_blade)
                
                # Obter timestamps
                start_ts = get_timestamp_for_index(df_p, cycle['start_idx'])
                end_ts = get_timestamp_for_index(df_p, cycle['end_idx'])
                
                # Gerar cycle_id único
                cycle_id = f"{pivot_val}_{start_ts.strftime('%Y%m%d_%H%M%S')}_C{ci:03d}"
                
                cycle_data = {
                    'cycle_id': cycle_id,
                    'pivo_id': pivot_val,
                    'start_date': start_ts,
                    'end_date': end_ts,
                    'blade_factor': pivot_blade,
                    'duration_minutes': cycle.get('duration_minutes', 0),
                    'lamina_360': lamina_cycle
                }
                
                db.insert_cycle_data(cycle_data)
        
    finally:
        db.disconnect()


def calculate_cycle_lamina_array(df_p, cycle, pivot_blade):
    """
    Calcula array de lâmina 360° para um ciclo específico
    """
    lamina_acc = np.zeros(360, dtype=float)
    
    # Obter dados angle-percentimeter do ciclo
    angle_percent_data = []
    for idx in cycle.get("indices", []):
        try:
            row = df_p.loc[idx]
            angle = pd.to_numeric(row.get("CurrentAngle", None), errors="coerce")
            percent = pd.to_numeric(row.get("Percentimeter", None), errors="coerce")
            
            if pd.notna(angle) and pd.notna(percent) and percent != 655 and angle != 655:
                angle_percent_data.append((int(angle % 360), float(percent)))
        except Exception:
            continue
    
    if not angle_percent_data:
        return lamina_acc
    
    direction = cycle.get("direction", "3")
    
    # Aplicar percentímetro aos ângulos
    for i, (current_angle, current_percent) in enumerate(angle_percent_data):
        if i < len(angle_percent_data) - 1:
            next_angle = angle_percent_data[i + 1][0]
            affected_angles = _get_angles_up_to_next(current_angle, next_angle, direction)
        else:
            affected_angles = [current_angle]
        
        if current_percent > 0:
            lam_per_deg = (pivot_blade * 100.0) / current_percent 
        else:
            lam_per_deg = 0.0
        
        for angle in affected_angles:
            lamina_acc[angle] += lam_per_deg
    
    return lamina_acc


def get_timestamp_for_index(df_p, idx):
    """Obtém timestamp para um índice do dataframe"""
    try:
        row = df_p.loc[idx]
        return safe_parse_timestamp(row.get("DtBe", ""), row.get("HourBe", ""))
    except:
        return datetime.now()

def _get_angles_up_to_next(start_angle: int, end_angle: int, direction: str):
    """
    Get all angles from start_angle up to (but NOT including) end_angle.
    
    Examples:
    - Direction 4 (increasing): 30° to 40° -> [30, 31, 32, ..., 39] (40 excluded)  
    - Direction 3 (decreasing): 40° to 30° -> [40, 39, 38, ..., 31] (30 excluded)
    """
    start = start_angle % 360
    end = end_angle % 360
    angles = []
    
    if direction == "4":  # increasing
        if start <= end:
            # Normal case: 30 to 40 -> [30, 31, 32, ..., 39]
            angles = list(range(start, end))
        else:
            # Wrap case: 350 to 20 -> [350, 351, ..., 359, 0, 1, ..., 19]
            angles = list(range(start, 360)) + list(range(0, end))
    
    else:  # direction == "3", decreasing
        if start >= end:
            # Normal case: 40 to 30 -> [40, 39, 38, ..., 31]
            angles = list(range(start, end, -1))
        else:
            # Wrap case: 30 to 350 -> [30, 29, 28, ..., 1, 0, 359, 358, ..., 351]
            angles = list(range(start, -1, -1)) + list(range(359, end, -1))
    
    return angles
# ---------------- Visualization ----------------
def pivot_heatmap(laminas_mm, titulo="Distribuição da Lâmina - Heatmap"):
    """Generate polar heatmap"""
    laminas_mm = np.asarray(laminas_mm, dtype=float)
    laminas_norm = (laminas_mm - np.min(laminas_mm)) / (np.max(laminas_mm) - np.min(laminas_mm) + 1e-12)
    angles = np.deg2rad(np.arange(0, 360))
    n_rings = 100
    radii = np.linspace(0.3, 1.0, n_rings)
    theta, r = np.meshgrid(angles, radii)
    z = np.tile(laminas_norm, (n_rings, 1))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    
    im = ax.pcolormesh(theta, r, z, cmap=plt.cm.viridis_r, shading="auto", vmin=0, vmax=1)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.grid(True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(titulo, va="bottom", fontsize=14, weight="bold")

    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis_r,
                               norm=plt.Normalize(vmin=np.min(laminas_mm), vmax=np.max(laminas_mm)))
    cbar = fig.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label("Lâmina acumulada (mm)", rotation=270, labelpad=20)
    return fig, ax

def pivot_bar_chart(laminas_mm, titulo="Lâmina acumulada por faixa angular", 
                   bottom=0.3, bar_scale="linear"):
    """Generate polar bar chart with optional scaling"""
    laminas_mm = np.array(laminas_mm, dtype=float)
    
    # Apply scaling transformation
    if bar_scale == "sqrt" and np.max(laminas_mm) > 0:
        laminas_display = np.sqrt(laminas_mm / np.max(laminas_mm)) * 0.6
    elif bar_scale == "log" and np.max(laminas_mm) > 0:
        laminas_display = np.log1p(laminas_mm) / np.log1p(np.max(laminas_mm)) * 0.6
    else:  # linear or none
        max_val = np.max(laminas_mm) if np.max(laminas_mm) > 0 else 1.0
        laminas_display = (laminas_mm / max_val) * 0.6

    n_bins = len(laminas_mm)
    theta = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
    width = 2 * np.pi / n_bins

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    bars = ax.bar(theta, laminas_display, width=width * 0.95, bottom=bottom,
                  edgecolor="white", linewidth=0.3, alpha=0.9)

    ax.grid(True)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{titulo} (scale: {bar_scale})", pad=20, fontsize=14, weight="bold")
    plt.tight_layout()
    return fig, ax

# REMOVED: transform_for_bars - integrated into pivot_bar_chart
# REMOVED: pivot_infografico_unitcircle - renamed to pivot_bar_chart

# ---------------- Export functions ----------------
def export_to_csv(df_raw, binned_df, pivot_name: str, out_dir: Path):
    """Export raw and binned data to CSV"""
    timestamp = datetime.now().strftime('%d-%m-%Y--%H-%M-%S')
    
    raw_out = out_dir / f"{pivot_name}_raw_{timestamp}.csv"
    df_raw.to_csv(raw_out, index=False)
    print(f"[OK] Raw data saved to {raw_out}")

    binned_out = out_dir / f"{pivot_name}_binned_{timestamp}.csv"
    binned_df.to_csv(binned_out, index=False)
    print(f"[OK] Binned data saved to {binned_out}")

def export_to_excel(df_raw, binned_df, cycle_rows: set, pivot_name: str, out_dir: Path):
    """Export to Excel with cycle row formatting"""
    timestamp = datetime.now().strftime('%d-%m-%Y--%H-%M-%S')
    xlsx_path = out_dir / f"{pivot_name}_excel_{timestamp}.xlsx"

    try:
        import xlsxwriter
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
            df_raw.to_excel(writer, sheet_name="Raw", index=False)
            binned_df.to_excel(writer, sheet_name="Binned", index=False)

            # Bold cycle rows
            if cycle_rows:
                workbook = writer.book
                worksheet = writer.sheets["Raw"]
                bold_fmt = workbook.add_format({"bold": True})
                
                bolded_count = 0
                for orig_idx in sorted(cycle_rows):
                    if orig_idx in df_raw.index:
                        pos = df_raw.index.get_loc(orig_idx)
                        excel_row = pos + 1  # +1 for header
                        worksheet.set_row(excel_row, None, bold_fmt)
                        bolded_count += 1
                
                print(f"[OK] Excel saved to {xlsx_path} with {bolded_count} cycle rows bolded")
    except ImportError:
        print("[WARN] xlsxwriter not installed; skipping Excel export")
    except Exception as e:
        print(f"[WARN] Excel export failed: {e}")

def save_cycle_csvs(df, cycles_dir: Path, pivot_blade: float):
    """Save individual cycle CSV files"""
    for pivot_val in df["Pivot"].unique():
        df_p = df[df["Pivot"] == pivot_val]
        cycles = find_cycles_for_pivot(df_p)
        
        if cycles:
            pivot_folder = cycles_dir / str(pivot_val)
            pivot_folder.mkdir(parents=True, exist_ok=True)
            
            for ci, c in enumerate(cycles, start=1):
                # Calculate cycle data (similar to main processing)
                p_list = [float(x) for x in c["percent_list"] if x is not None and not math.isnan(float(x))]
                p_cycle = np.median(p_list) if p_list else 0.0
                lam_per_deg = (pivot_blade * 100) / p_cycle if p_cycle > 0 else 0.0 
                
                sa = int(c["start_angle"]) if c.get("start_angle") is not None else None
                ea = int(c["end_angle"]) if c.get("end_angle") is not None else None
                
                if sa is not None and ea is not None:
                    degs = _degrees_range_inclusive(sa % 360, ea % 360, str(c.get("direction", "3")))
                    
                    cycle_df = pd.DataFrame({
                        "AngleDeg": degs,
                        "LaminaPerDegree": [lam_per_deg] * len(degs),
                        "PercentimeterCycle": [p_cycle] * len(degs),
                        "CycleIndex": [ci] * len(degs),
                    })
                    
                    fname = pivot_folder / f"{pivot_val}_cycle{ci}.csv"
                    cycle_df.to_csv(fname, index=False)

                    

# ---------------- Main CLI ----------------
@click.command()
@click.option("--root", default="./resources/logs", type=click.Path(path_type=Path))
@click.option("--pivots", multiple=True, default=["agrocangaia2"], help="Which pivots to parse")
@click.option("--csvfile", type=click.Path(path_type=Path), 
              default=Path("./resources/logs/logsAgroCangaiaCyclesAcumulado.csv"))
@click.option("--export-csv", type=click.Path(path_type=Path), default=None,
              flag_value=Path("./resources/outputCSV"))
@click.option("--export-excel", type=click.Path(path_type=Path), default=None,
              flag_value=Path("./resources/outputCSV"))
@click.option("--export-cycles", type=click.Path(path_type=Path), default=None,
              flag_value=Path("./resources/outputCSV"))
@click.option("--source", type=click.Choice(["csv", "logs"]), default="logs")
@click.option("--pivot-blade", default=5.46, type=float)
@click.option("--bar-scale", type=click.Choice(["linear", "sqrt", "log"]), default="linear")
@click.option("--save-dir", type=click.Path(path_type=Path), default=Path("./resources/imgs"))
@click.option("--save-database", is_flag=True, help="Save cycles to PostgreSQL database")

def main(root, pivots, csvfile, export_csv, export_excel, export_cycles, 
         source, pivot_blade, bar_scale, save_dir, save_database):
    
    if source == "csv" and csvfile and csvfile.exists():
        print(f"\n🔹 Loading data from CSV: {csvfile}")
        df = pd.read_csv(csvfile, sep=",", encoding="utf-8", engine="python")
        df = df.rename(columns=lambda c: c.strip().lower())
        
        if "grau" not in df.columns or "lamina acumulada" not in df.columns:
            raise SystemExit("CSV must contain 'grau' and 'lamina acumulada' columns")
            
        # Simple visualization for CSV source
        angles = pd.to_numeric(df["grau"], errors="coerce") % 360
        laminas = pd.to_numeric(df["lamina acumulada"], errors="coerce").fillna(0)
        
        # Create 360-degree array
        lamina_360 = np.zeros(360)
        for angle, lamina in zip(angles, laminas):
            if not np.isnan(angle):
                lamina_360[int(angle)] = lamina
                
        pivot_name = "CSV_Data"

        if save_database:
            print("\n[DB] Salvando ciclos no banco de dados........")
            save_cycles_to_database(df, pivot_name, pivot_blade)
        
    elif source == "logs":
        print(f"\n🔹 Loading data from logs: {root}")
        df = parse_all_logs(root, pivots)
        if df.empty:
            raise SystemExit("No valid rows parsed from logs")

        # Process data
        df["Percentimeter"] = pd.to_numeric(df["Percentimeter"], errors="coerce").fillna(0)
        df["CurrentAngle"] = pd.to_numeric(df["CurrentAngle"], errors="coerce")
        df["Timestamp"] = df.apply(lambda r: safe_parse_timestamp(r.get("DtBe", ""), r.get("HourBe", "")), axis=1)
        df = df.sort_values(["Timestamp"]).reset_index(drop=True)

        pivot_name = str(df["Pivot"].iloc[0])
        
        # Process cycles
        lamina_360, cycle_rows = process_cycles_to_accumulators(df, pivot_blade)
        
        # Prepare export data
        df_raw = df.copy()
        cmd_digits = (df_raw["Command"].astype(str).fillna("").str.replace(r"\D", "", regex=True)
                     .str.zfill(3).str.slice(0, 3))
        
        df_raw["Direction"] = pd.to_numeric(cmd_digits.str[0], errors="coerce").fillna(0).astype(int)
        df_raw["Water"] = pd.to_numeric(cmd_digits.str[1], errors="coerce").fillna(0).astype(int)
        df_raw["Power"] = pd.to_numeric(cmd_digits.str[2], errors="coerce").fillna(0).astype(int)

        df_raw["Direction"] = df_raw["Direction"].map({8: "OFF", 3: "Forward", 4: "Reverse"})
        df_raw["Water"] = df_raw["Water"].map({5: "OFF", 7: "Pressuring", 6: "ON"})
        df_raw["Power"] = df_raw["Power"].map({1: "ON", 2: "OFF"})

        binned_df = pd.DataFrame({
            "AngleDeg": np.arange(0, 360, dtype=int),
            "Lamina": lamina_360
        })

        # Export data
        if export_csv:
            export_to_csv(df_raw, binned_df, pivot_name, Path(export_csv))
        if export_excel:
            export_to_excel(df_raw, binned_df, cycle_rows, pivot_name, Path(export_excel))
        if export_cycles:
            save_cycle_csvs(df, Path(export_cycles), pivot_blade)
        if save_database:
            print("\n[DB] Salvando ciclos no banco de dados........")
            save_cycles_to_database(df, pivot_name, pivot_blade)
    else:
        raise SystemExit("Invalid source selected")

    # Generate visualizations
    out_folder = Path(save_dir) / pivot_name
    out_folder.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%d-%m-%Y--%H-%M-%S')
    
    # Summary stats
    print(f"\nSummary for '{pivot_name}':")
    print(f"  min = {np.min(lamina_360):.2f} mm")
    print(f"  max = {np.max(lamina_360):.2f} mm") 
    print(f"  mean= {np.mean(lamina_360):.2f} mm")

    # Bar chart
    fig1, _ = pivot_bar_chart(lamina_360, bar_scale=bar_scale)
    fname1 = out_folder / f"bar_chart_{pivot_name}_{source}_{bar_scale}_{timestamp}.png"
    fig1.savefig(fname1, dpi=200, bbox_inches="tight")
    print(f"Bar chart saved: {fname1}")

    # Heatmap  
    fig2, _ = pivot_heatmap(lamina_360)
    fname2 = out_folder / f"heatmap_{pivot_name}_{source}_{timestamp}.png"
    fig2.savefig(fname2, dpi=200, bbox_inches="tight")
    print(f"Heatmap saved: {fname2}")

    plt.show()

if __name__ == "__main__":
    main()