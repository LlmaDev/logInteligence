import click
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import re
from database import IrrigationDatabase
from typing import List, Dict

"""
agrocangaia2: 5,46
agrocangaia3: 3.87
agrocangaia4: 6.18
agrocangaia13: 6.12
agrocangaia15: 4.6
"""

# ============================================================================
# STEP 1: PARSE RAW LOGS
# Purpose: Read MESSAGE.txt files and extract structured data
# Logging: Uncomment df.to_csv() line to save for troubleshooting
# ============================================================================

def parse_message_line(line: str):
    """Parse single MESSAGE.txt line. Format: DD-MM-YYYY HH:MM:SS -> #Status-Farm-Command-...$"""
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
    """Parse all MESSAGE.txt files for specified pivots."""
    rows = []
    for pivot in pivots:
        for msgfile in sorted(root.glob(f"{pivot}/*/*/MESSAGE.txt"), key=lambda p: str(p)):
            for line in msgfile.read_text(encoding="utf-8", errors="ignore").splitlines():
                parsed = parse_message_line(line)
                if parsed:
                    parsed["Pivot"] = pivot
                    rows.append(parsed)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["Percentimeter"] = pd.to_numeric(df["Percentimeter"], errors="coerce")
    df["CurrentAngle"] = pd.to_numeric(df["CurrentAngle"], errors="coerce")
    # 🔍 LOGGING: df.to_csv("debug_01_raw_parsed.csv", index=False)
    return df

# ============================================================================
# STEP 2: FILTER ANGLE GLITCHES
# Purpose: Remove sensor glitches BEFORE cycle detection
# Pattern: [110°, 190°, 113°] → removes 190° (spike that immediately reverts)
# Logic: jump_in >= 40° AND jump_out >= 40° AND bypass < 40° = GLITCH
# ============================================================================

def filter_angle_glitches(df: pd.DataFrame, direction: str, threshold: float = 40.0) -> bool:
    """
    Check if a cycle has angle glitches by examining ALL consecutive reading pairs.
    
    This function validates an ENTIRE CYCLE by checking:
    1. Three-way spikes: [A → B → A] where B is anomalous
    2. Direction violations: movement opposite to expected direction
    3. Large jumps: sudden angle changes > threshold
    
    Args:
        df: DataFrame containing cycle readings with 'CurrentAngle' column
        direction: Expected direction ("3"=DECREMENT/down, "4"=INCREMENT/up)
        threshold: Maximum allowed angle change in degrees (default: 40.0)
        
    Returns:
        True if cycle is VALID (no glitches found)
        False if cycle is INVALID (glitches detected - should be discarded)
    """
    
    # ============================================================================
    # STEP 1: INITIAL VALIDATION
    # ============================================================================
    if df.empty or len(df) < 3:
        print(f"[GLITCH_CHECK] ⚠️  Insufficient data: {len(df)} readings")
        return True  # Too small to validate, assume clean
    
    print(f"\n{'='*70}")
    print(f"[GLITCH_CHECK] Analyzing cycle: {len(df)} readings")
    
    # Normalize direction to string for consistent comparison
    direction_code = str(direction).strip()
    
    # Validate direction code
    if direction_code not in ("3", "4"):
        print(f"[GLITCH_CHECK] ⚠️  Unknown direction '{direction_code}', assuming valid")
        return True
    
    print(f"[GLITCH_CHECK] Expected direction: {direction_code} ({'DECREMENT ↓' if direction_code=='3' else 'INCREMENT ↑'})")
    print(f"[GLITCH_CHECK] Jump threshold: {threshold}°")
    print(f"{'='*70}")
    
    # ============================================================================
    # STEP 2: EXTRACT AND CLEAN ANGLE SEQUENCE
    # ============================================================================
    angles = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        try:
            angle = row["CurrentAngle"]
            
            # Skip invalid readings (655 = sensor error code)
            if pd.isna(angle) or angle == 655:
                continue
                
            # Normalize to 0-360 range
            angle_norm = float(angle) % 360
            angles.append(angle_norm)
            valid_indices.append(idx)
            
        except (ValueError, KeyError, TypeError) as e:
            print(f"[GLITCH_CHECK] ⚠️  Skipping invalid row {idx}: {e}")
            continue
    
    # Need at least 3 valid readings for meaningful validation
    if len(angles) < 3:
        print(f"[GLITCH_CHECK] ⚠️  Only {len(angles)} valid angles after cleaning")
        return True
    
    print(f"[GLITCH_CHECK] Valid readings: {len(angles)}/{len(df)}")
    print(f"[GLITCH_CHECK] Angle range: {min(angles):.1f}° to {max(angles):.1f}°")
    
    # ============================================================================
    # STEP 3: SLIDING WINDOW VALIDATION (consecutive pairs)
    # ============================================================================
    glitch_count = 0
    direction_violations = 0
    large_jumps = []
    
    for i in range(len(angles) - 1):  # Check ALL consecutive pairs
        
        prev_angle = angles[i]
        curr_angle = angles[i + 1]
        
        # ---------------------------------------------------------------------
        # Calculate angular difference (handling 360° wrap-around)
        # ---------------------------------------------------------------------
        diff = curr_angle - prev_angle
        
        # Normalize to -180 to +180 range
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        
        abs_diff = abs(diff)
        
        # ---------------------------------------------------------------------
        # CHECK 1: Large Jump Detection
        # ---------------------------------------------------------------------
        if abs_diff > threshold:
            large_jumps.append((i, prev_angle, curr_angle, abs_diff))
            print(f"[GLITCH] 🚨 Large jump at reading {i+1}: {prev_angle:.1f}° → {curr_angle:.1f}° (Δ={abs_diff:.1f}°)")
            glitch_count += 1
            
            # If jump is large, check if it's a three-way spike
            if i + 2 < len(angles):
                next_angle = angles[i + 2]
                
                # Calculate bypass (direct prev → next)
                bypass = next_angle - prev_angle
                if bypass > 180:
                    bypass -= 360
                elif bypass < -180:
                    bypass += 360
                bypass_abs = abs(bypass)
                
                # Three-way spike pattern: jump out, jump back, small bypass
                jump_out = abs(angles[i + 2] - curr_angle)
                if jump_out > 180:
                    jump_out = 360 - jump_out
                
                if jump_out >= threshold and bypass_abs < threshold:
                    print(f"           └→ Three-way spike detected: {prev_angle:.1f}° → {curr_angle:.1f}° → {next_angle:.1f}°")
                    print(f"              Bypass = {bypass_abs:.1f}° (should be < {threshold}°)")
                    print(f"[GLITCH_CHECK] ❌ INVALID CYCLE - Three-way spike (false 360° coverage)")
                    print(f"{'='*70}\n")
                    return False  # INVALID CYCLE - discard immediately
        
        # ---------------------------------------------------------------------
        # CHECK 2: Direction Violation Detection
        # ---------------------------------------------------------------------
        if direction_code == "3":  # DECREMENT - angle should DECREASE
            # In decrement mode, diff should be negative or small positive (noise)
            # Large positive movement = wrong direction
            if diff > threshold:
                print(f"[GLITCH] 🚨 Direction violation (DECR): {prev_angle:.1f}° → {curr_angle:.1f}° (+{diff:.1f}°)")
                print(f"           Expected: angle should DECREASE, but it INCREASED by {diff:.1f}°")
                direction_violations += 1
                
        elif direction_code == "4":  # INCREMENT - angle should INCREASE
            # In increment mode, diff should be positive or small negative (noise)
            # Large negative movement = wrong direction
            if diff < -threshold:
                print(f"[GLITCH] 🚨 Direction violation (INCR): {prev_angle:.1f}° → {curr_angle:.1f}° ({diff:.1f}°)")
                print(f"           Expected: angle should INCREASE, but it DECREASED by {abs(diff):.1f}°")
                direction_violations += 1
    
    # ============================================================================
    # STEP 4: FINAL VERDICT
    # ============================================================================
    total_issues = glitch_count + direction_violations
    
    print(f"\n[GLITCH_CHECK] Summary:")
    print(f"  - Large jumps: {glitch_count}")
    print(f"  - Direction violations: {direction_violations}")
    print(f"  - Total issues: {total_issues}")
    
    if total_issues > 0:
        # 🔴 CRITICAL: Stricter tolerance to prevent false 360° coverage
        # Allow up to 1 minor issue in very long cycles only
        tolerance = 1 if len(angles) > 100 else 0
        
        if total_issues > tolerance:
            print(f"[GLITCH_CHECK] ❌ INVALID CYCLE - {total_issues} issues exceed tolerance ({tolerance})")
            print(f"               This prevents false 360° coverage from glitchy data")
            print(f"{'='*70}\n")
            return False
        else:
            print(f"[GLITCH_CHECK] ⚠️  ACCEPTED with {total_issues} minor issue (within tolerance for {len(angles)} readings)")
            print(f"{'='*70}\n")
            return True
    else:
        print(f"[GLITCH_CHECK] ✅ VALID CYCLE - No glitches detected")
        print(f"{'='*70}\n")
        return True


def safe_parse_timestamp(dt_str: str, hour_str: str):
    """Parse timestamp with fallback."""
    s = f"{dt_str} {hour_str}"
    try:
        return datetime.strptime(s, "%d-%m-%Y %H:%M:%S")
    except Exception:
        try:
            return pd.to_datetime(s, dayfirst=True)
        except Exception:
            return pd.NaT

# ============================================================================
# STEP 3: DETECT CYCLES
# Purpose: Identify irrigation cycles from command sequences
# Start: Command Y="6" (Water ON), End: Y="5" (Water OFF)
# Special: Y="7" (Pressurizing) - check if temporary or cycle break
# ============================================================================

def find_cycles_for_pivot(df_p: pd.DataFrame):
    """Detect cycles from command sequences."""
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
        try:
            return int(val) if str(val).isdigit() else None
        except:
            return None
    
    def _get_filtered_percentimeter(val):
        try:
            pval = float(val or 0.0)
            return None if pval == 655.0 else pval
        except:
            return 0.0
    
    def _replace_655_with_median(percent_list):
        valid_values = [p for p in percent_list if p is not None]
        if not valid_values:
            return [0.0] * len(percent_list)
        m = np.median(valid_values)
        return [m if p is None else p for p in percent_list]
    
    def _get_cycle_duration_minutes(start_idx, end_idx):
        try:
            start_row = df_p.loc[start_idx]
            end_row = df_p.loc[end_idx]
            start_ts = safe_parse_timestamp(start_row.get("DtBe", ""), start_row.get("HourBe", ""))
            end_ts = safe_parse_timestamp(end_row.get("DtBe", ""), end_row.get("HourBe", ""))
            if pd.isna(start_ts) or pd.isna(end_ts):
                return 0
            return max(0, (end_ts - start_ts).total_seconds() / 60.0)
        except Exception:
            return 0
    
    def _save_cycle_if_valid(start_idx, end_idx, start_angle, end_angle, direction, percent_values, indices_in_cycle, warning=None): ##Validação aqui
        if not indices_in_cycle:
            return False
        duration_min = _get_cycle_duration_minutes(start_idx, end_idx)
        if duration_min < 5.0:
            return False  # Discard cycles < 5min
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
        return True
    
    rows = list(df_p.itertuples())
    i = 0
    
    while i < len(rows):
        row = rows[i]
        orig_idx = row.Index
        cmd_digits = _digits_of_command(getattr(row, "Command", "") or "")
        first_digit = cmd_digits[0] if len(cmd_digits) >= 1 else None
        second_digit = cmd_digits[1] if len(cmd_digits) >= 2 else None
        current_command = _get_command_value(getattr(row, "Command", ""))
        pval = _get_filtered_percentimeter(getattr(row, "Percentimeter", 0.0))
        
        try:
            ang = int(float(getattr(row, "CurrentAngle", None)))
            if ang == 655:
                ang = None
            else:
                last_valid_angle = ang
        except Exception:
            ang = None
        
        # 🔴 FIXED: Added missing continue statements!
        # Cycle start
        if (not in_cycle) and (second_digit == "6"):
            in_cycle = True
            start_idx = orig_idx
            direction = first_digit if first_digit is not None else "3"
            start_angle = ang if ang is not None else last_valid_angle
            percent_values = [pval]
            indices_in_cycle = [orig_idx]
            last_command_before_7 = current_command
            i += 1
            continue  # 🔴 FIXED: Was missing!
        
        # Cycle end
        if in_cycle and (second_digit == "5"):
            end_idx = indices_in_cycle[-1] if indices_in_cycle else start_idx
            end_angle = last_valid_angle
            _save_cycle_if_valid(start_idx, end_idx, start_angle, end_angle, direction, percent_values, indices_in_cycle)
            in_cycle = False
            start_idx = None
            percent_values = []
            direction = None
            start_angle = None
            indices_in_cycle = []
            last_command_before_7 = None
            i += 1
            continue  # 🔴 FIXED: Was missing!
        
        # Handle "7" sequences
        if in_cycle and (second_digit == "7"):
            j = i
            seven_indices = []
            seven_percent_values = []
            while j < len(rows):
                current_row = rows[j]
                current_cmd_digits = _digits_of_command(getattr(current_row, "Command", "") or "")
                current_second_digit = current_cmd_digits[1] if len(current_cmd_digits) >= 2 else None
                if current_second_digit == "7":
                    seven_indices.append(current_row.Index)
                    seven_pval = _get_filtered_percentimeter(getattr(current_row, "Percentimeter", 0.0))
                    seven_percent_values.append(seven_pval)
                    try:
                        seven_ang = int(float(getattr(current_row, "CurrentAngle", None)))
                        if seven_ang != 655 and seven_ang is not None:
                            last_valid_angle = seven_ang
                    except:
                        pass
                    j += 1
                else:
                    break
            
            if j < len(rows):
                next_row = rows[j]
                next_command = _get_command_value(getattr(next_row, "Command", ""))
                if next_command == last_command_before_7:
                    percent_values.extend(seven_percent_values)
                    indices_in_cycle.extend(seven_indices)
                    i = j
                    continue  # 🔴 FIXED: Was missing!
                else:
                    end_idx = indices_in_cycle[-1] if indices_in_cycle else start_idx
                    _save_cycle_if_valid(start_idx, end_idx, start_angle, last_valid_angle, direction, percent_values, indices_in_cycle)
                    next_cmd_digits = _digits_of_command(getattr(next_row, "Command", "") or "")
                    next_first_digit = next_cmd_digits[0] if len(next_cmd_digits) >= 1 else None
                    try:
                        next_ang = int(float(getattr(next_row, "CurrentAngle", None)))
                        if next_ang != 655 and next_ang is not None:
                            last_valid_angle = next_ang
                    except:
                        pass
                    next_pval = _get_filtered_percentimeter(getattr(next_row, "Percentimeter", 0.0))
                    start_idx = next_row.Index
                    direction = next_first_digit if next_first_digit is not None else direction
                    start_angle = last_valid_angle
                    percent_values = [next_pval]
                    indices_in_cycle = [next_row.Index]
                    last_command_before_7 = next_command
                    i = j + 1
                    continue  # 🔴 FIXED: Was missing!
            else:
                percent_values.extend(seven_percent_values)
                indices_in_cycle.extend(seven_indices)
                i = j
                continue  # 🔴 FIXED: Was missing!
        
        # Normal reading
        if in_cycle:
            percent_values.append(pval)
            indices_in_cycle.append(orig_idx)
            if ang is not None:
                last_valid_angle = ang
            if second_digit != "7":
                last_command_before_7 = current_command
        
        i += 1
    
    # Handle EOF
    if in_cycle:
        end_idx = indices_in_cycle[-1] if indices_in_cycle else None
        if end_idx:
            _save_cycle_if_valid(start_idx, end_idx, start_angle, last_valid_angle, direction, percent_values, indices_in_cycle, warning="closed_on_eof")
    
    return cycles

# ============================================================================
# STEP 4: GENERATE ANGLES BETWEEN READINGS
# Purpose: Fill gaps between consecutive sensor readings
# Direction "3"=DECREMENT, "4"=INCREMENT
# Returns angles from START (inclusive) to END (exclusive)
# ============================================================================

def minimal_signed_diff(a: int, b: int) -> int:
    """Return minimal signed angular difference from a -> b in range [-180, 179].

    Normalizes inputs to [0,360) then computes the shortest signed rotation
    (positive = rotate forward/increment, negative = rotate backward/decrement).
    """
    a %= 360
    b %= 360
    # (b - a + 540) % 360 moves raw diff into [0,360), then -180 shifts to [-180,179]
    d = (b - a + 540) % 360 - 180
    return d


def get_angles_between_readings(start_angle: int, end_angle: int, direction: str, glitch_threshold: int) -> List[int]:
    """Generate angles FROM start (inclusive) TO end (exclusive).
       Skip if jump > glitch_threshold or transient spike."""
    def minimal_signed_diff(a: int, b: int) -> int:
        a %= 360; b %= 360
        return (b - a + 540) % 360 - 180

    start = int(start_angle) % 360
    end = int(end_angle) % 360
    angles: List[int] = []

    # --- Skip identical ---
    if start == end:
        return angles

    d = minimal_signed_diff(start, end)
    abs_d = abs(d)
def get_angles_between_readings(start_angle: int, end_angle: int, direction: str, glitch_threshold: int) -> List[int]:
    """Generate discrete angles from start (inclusive) to end (exclusive) along the minimal path.

    - Uses minimal_signed_diff to discover shortest rotation direction and magnitude.
    - If difference is 0 -> returns [] (nothing to fill).
    - If abs(diff) > glitch_threshold -> treat as a jump/glitch and skip filling -> returns [].
    - Otherwise step = +1 if diff>0 else -1 and yield each angle (mod 360) until end is reached.
    - Raises RuntimeError if loop unexpectedly exceeds 360 iterations (safety).
    - `direction` param is accepted for compatibility but not required: function infers direction from values.
    """
    start = int(start_angle) % 360
    end = int(end_angle) % 360
    d = minimal_signed_diff(start, end)

    if d == 0:
        return []  # identical readings -> nothing between them
    if abs(d) > int(glitch_threshold):
        return []  # treat as an invalid big jump -> skip filling
    step = 1 if d > 0 else -1
    angles: List[int] = []
    cur = start
    # iterate until we reach end (exclusive), appending each intermediate reading
    while cur != end:
        angles.append(cur)
        cur = (cur + step) % 360
        if len(angles) > 360:
            raise RuntimeError("infinite loop")
    return angles
    
    # --- Spike detection: large delta = glitch ---
    if abs_d > glitch_threshold:
        return angles  # skip filling on spike/jump

    # --- Direction inference / enforcement ---
    step = 1 if (direction == "4" or d > 0) else -1
    if (step == 1 and d < 0) or (step == -1 and d > 0):
        return angles  # mismatch between given direction and diff

    cur = start
    while cur != end:
        angles.append(cur)
        cur = (cur + step) % 360
        if len(angles) > 360:
            raise RuntimeError("infinite loop")

    return angles


def remove_spike_runs(readings: List[int], glitch_threshold: int) -> List[int]:
    """Remove runs of identical readings that look like spikes.

    A run of value V is considered a spike (and removed) if:
      - there exists a previous value P (immediately before the run) and a next value N (immediately after the run),
      - P and N are close to each other (abs(minimal_signed_diff(P, N)) <= glitch_threshold),
      - and both P and N are far from V (abs(minimal_signed_diff(V, P)) > glitch_threshold and abs(minimal_signed_diff(V, N)) > glitch_threshold).

    Rationale: this removes intermediate incorrect values (like a short run of 170 between 36 and 31)
    when the surrounding readings match and are close to each other, indicating the middle run is a spike.
    The entire consecutive run of V is removed (not just one sample).
    """
    n = len(readings)
    i = 0
    out: List[int] = []

    while i < n:
        v = readings[i]
        # find end of run of same value v
        j = i
        while j < n and readings[j] == v:
            j += 1
        prev_idx = i - 1
        next_idx = j if j < n else None

        # If we have both prev and next, test the spike criteria
        if prev_idx >= 0 and next_idx is not None:
            prev = readings[prev_idx]
            nxt = readings[next_idx]
            if (abs(minimal_signed_diff(prev, nxt)) <= glitch_threshold and
                abs(minimal_signed_diff(v, prev)) > glitch_threshold and
                abs(minimal_signed_diff(v, nxt)) > glitch_threshold):
                # it's a spike run -> skip the entire run by advancing i to j
                i = j
                continue

        # otherwise keep the run as-is
        out.extend(readings[i:j])
        i = j
    return out

# ============================================================================
# STEP 5: CALCULATE LAMINA PER CYCLE  
# Purpose: Compute lamina distribution for each degree
# Formula: lamina_per_deg = (pivot_blade × 100) / percentimeter 
# 🔴 CRITICAL: Last reading must NOT wrap to create false 360° coverage!
# ============================================================================

def calculate_cycle_lamina_array(df_p: pd.DataFrame, cycle: dict, pivot_blade: float, glitch_threshold ) -> np.ndarray:
    """
    Calculate lamina distribution for a cycle using the MEDIAN percentimeter
    for the whole cycle (replaces per-reading percentimeter).

    🔴 FIXED: Last reading does NOT wrap - prevents false 360° coverage
    """
    lamina_acc = np.zeros(360, dtype=float)
    angle_percent_data = []
    
    for idx in cycle.get("indices", []):
        try:
            row = df_p.loc[idx]
            angle = pd.to_numeric(row.get("CurrentAngle", None), errors="coerce")
            percent = pd.to_numeric(row.get("Percentimeter", None), errors="coerce")
            
            if pd.notna(angle) and pd.notna(percent) and angle != 655 and percent != 655:
                angle_clean = int(angle % 360)
                percent_clean = float(percent)
                
                if percent_clean <= 0 or percent_clean > 200:
                    print(f"[WARN] Invalid percentimeter {percent_clean} at angle {angle_clean}")
                    continue
                
                angle_percent_data.append((angle_clean, percent_clean))
        except Exception as e:
            print(f"[WARN] Error at index {idx}: {e}")
            continue
    
    if not angle_percent_data:
        print("[WARN] No valid data - returning zeros")
        return lamina_acc

    # compute MEDIAN percentimeter for the entire cycle
    percents = np.array([p for _, p in angle_percent_data], dtype=float)
    median_percent = float(np.median(percents))
    if median_percent <= 0 or median_percent > 10000:  # sanity upper bound
        print(f"[WARN] Bad median percentimeter: {median_percent} - aborting")
        return lamina_acc
    print(f"[DEBUG] Using MEDIAN percentimeter for cycle: {median_percent:.4f}")

    direction = cycle.get("direction", "3")
    print(f"[DEBUG] Direction: {direction} ({'DECREMENT' if direction=='3' else 'INCREMENT'})")
    
    # 🔴 CRITICAL: Calculate actual angle span
    start_angle = angle_percent_data[0][0]
    end_angle = angle_percent_data[-1][0]
    
    if direction == "3":  # DECREMENT
        if start_angle >= end_angle:
            span = start_angle - end_angle
        else:
            span = start_angle + (360 - end_angle)
    else:  # INCREMENT
        if end_angle >= start_angle:
            span = end_angle - start_angle
        else:
            span = end_angle + (360 - start_angle)
    
    print(f"[DEBUG] Angle span: {start_angle}° → {end_angle}° = {span}° coverage")
    
    # Process each reading — use median_percent for all lamina calculations
    total_angles_affected = 0
    for i, (current_angle, current_percent) in enumerate(angle_percent_data):
        
        if i < len(angle_percent_data) - 1:
            next_angle = angle_percent_data[i + 1][0]
            affected_angles = get_angles_between_readings(current_angle, next_angle, direction, glitch_threshold)
            if affected_angles is None:
                continue

            if i < 3:
                print(f"[FILL {i+1}] {current_angle}° → {next_angle}° | {len(affected_angles)} angles")
        else:
            # 🔴 CRITICAL FIX: Last reading NEVER wraps!
            affected_angles = [current_angle]
            print(f"[FILL {i+1}] LAST: {current_angle}° | Only self (NO WRAP)")
        
        total_angles_affected += len(affected_angles)
        
        # Calculate lamina using median_percent for whole cycle
        lamina_per_deg = (pivot_blade * 100.0) / median_percent
        
        if i == 0:
            print(f"[CALC] ({pivot_blade} * 100) / {median_percent:.4f} = {lamina_per_deg:.4f} mm/deg (median used)")
        
        for angle in affected_angles:
            lamina_acc[angle] += lamina_per_deg
    
    # Validation
    actual_coverage = np.sum(lamina_acc > 0.01)
    print(f"[DEBUG] Total angles filled: {total_angles_affected}")
    print(f"[DEBUG] Actual coverage: {actual_coverage}° (span was {span}°)")
    print(f"[DEBUG] Lamina: min={np.min(lamina_acc):.2f}, max={np.max(lamina_acc):.2f}")
    
    # 🔴 WARNING: Check for impossible coverage
    if actual_coverage >= 350 and cycle.get("duration_minutes", 0) < 600:
        print(f"[WARNING] ⚠️  Suspicious: {actual_coverage}° in {cycle.get('duration_minutes', 0):.0f} min!")
    
    return lamina_acc


# ============================================================================
# STEP 6: AGGREGATE ALL CYCLES FOR VISUALIZATION
# Purpose: Process all pivots, calculate lamina for each cycle independently,
#          then SUM for visualization purposes ONLY
# 
# CRITICAL: Each cycle's lamina is calculated and stored SEPARATELY in database
#           The sum (lamina_total) is ONLY used for generating heatmap/bar chart
#           showing total accumulated irrigation across all cycles
# ============================================================================

def compute_all_cycles_and_lamina(df: pd.DataFrame, pivot_blade: float, glitch_threshold):
    """
    Detect all cycles and calculate lamina for each independently.
    Only processes cycles that pass glitch detection.
    
    🔴 CRITICAL: This function prevents false 360° coverage by:
    1. Validating each cycle's angle readings for glitches
    2. Rejecting cycles with direction violations
    3. Rejecting cycles with large angle jumps
    """
    lamina_total_for_viz = np.zeros(360, dtype=float)
    cycles_info = []
    
    print(f"\n{'='*70}")
    print(f"CYCLE DETECTION AND LAMINA CALCULATION")
    print(f"Using pivot_blade = {pivot_blade} mm")
    print(f"Will skip cycles with angle glitches")
    print(f"{'='*70}")
    
    for pivot_val in df["Pivot"].unique():
        df_p = df[df["Pivot"] == pivot_val]
        
        cycles = find_cycles_for_pivot(df_p)
        print(f"\n[INFO] Pivot '{pivot_val}' → {len(cycles)} cycles detected")
        
        valid_cycles = 0
        skipped_cycles = 0
        
        for ci, c in enumerate(cycles, start=1):
            print(f"\n{'='*70}")
            print(f"CYCLE {ci}/{len(cycles)} - CHECKING FOR GLITCHES")
            print(f"{'='*70}")
            
            # Get the cycle direction (default to "3" if missing)
            cycle_direction = c.get("direction")  
            print(f"[CYCLE] Direction from cycle dict: '{cycle_direction}'")
            
            # Extract cycle data using the indices
            cycle_indices = c.get("indices", [])
            if not cycle_indices:
                print("🚫 [CYCLE] No indices found - skipping")
                skipped_cycles += 1
                continue  
                
            df_cycle = df_p.loc[cycle_indices]
            
            # 🔍 DEBUG: Show what we're checking
            print(f"\n[DEBUG] Cycle {ci} info:")
            print(f"  - Direction: {cycle_direction} ({'DECREMENT' if cycle_direction=='3' else 'INCREMENT' if cycle_direction=='4' else 'UNKNOWN'})")
            print(f"  - Total readings: {len(df_cycle)}")
            print(f"  - First 3 angles: {df_cycle['CurrentAngle'].head(3).tolist()}")
            print(f"  - Last 3 angles: {df_cycle['CurrentAngle'].tail(3).tolist()}")
            print(f"  - Duration: {c.get('duration_minutes'):.1f} minutes")
            
            # 🔴 Check for glitches in this cycle
            # is_clean = filter_angle_glitches(df_cycle, cycle_direction)
            
            # if not is_clean:
            #     print(f"🚫 [CYCLE] Cycle {ci} has glitches - SKIPPING")
            #     skipped_cycles += 1
            #     continue  # 🔴 FIXED: Was outside the if block!
            
            # If we get here, the cycle is clean - calculate lamina
            print(f"✅ [CYCLE] Cycle {ci} is clean - calculating lamina")
            lamina_cycle = calculate_cycle_lamina_array(df_p, c, pivot_blade, glitch_threshold)
            lamina_total_for_viz+=lamina_cycle
            
            # Parse timestamps
            start_ts = safe_parse_timestamp(
                df_p.loc[c["start_idx"]].get("DtBe", ""), 
                df_p.loc[c["start_idx"]].get("HourBe", "")
            )
            end_ts = safe_parse_timestamp(
                df_p.loc[c["end_idx"]].get("DtBe", ""), 
                df_p.loc[c["end_idx"]].get("HourBe", "")
            )
            
            cycle_id = f"{pivot_val}_{start_ts.strftime('%Y%m%d_%H%M%S')}_C{ci:03d}"
            
            cycles_info.append({
                "cycle_id": cycle_id,
                "pivo_id": pivot_val,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "duration_minutes": c.get("duration_minutes"),
                "lamina_360": lamina_cycle,
                "start_idx": c.get("start_idx"),
                "end_idx": c.get("end_idx"),
                "indices": c.get("indices"),
                "direction": c.get("direction"),
                "blade_factor": pivot_blade,
                "validated": True  # Mark as validated (passed glitch check)
            })
            valid_cycles += 1
        
        print(f"\n[SUMMARY] Pivot '{pivot_val}': {valid_cycles} valid cycles, {skipped_cycles} skipped due to glitches")
    
    print(f"\n{'='*70}")
    print(f"AGGREGATION COMPLETE")
    print(f"Total valid cycles: {len(cycles_info)}")
    if cycles_info:
        print(f"Lamina range: min={np.min(lamina_total_for_viz):.2f}, max={np.max(lamina_total_for_viz):.2f} mm/deg")
    else:
        print("⚠️  No valid cycles found after glitch filtering!")
    print(f"{'='*70}\n")
    
    return lamina_total_for_viz, cycles_info

# ============================================================================
# STEP 7-8: EXPORT & DATABASE (unchanged from original)
# ============================================================================

def export_to_csv(df_raw, binned_df, pivot_name: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%d-%m-%Y--%H-%M-%S')
    raw_out = out_dir / f"{pivot_name}_raw_{timestamp}.csv"
    df_raw.to_csv(raw_out, index=False)
    print(f"[OK] Raw: {raw_out}")
    binned_out = out_dir / f"{pivot_name}_binned_{timestamp}.csv"
    binned_df.to_csv(binned_out, index=False)
    print(f"[OK] Binned: {binned_out}")

def export_to_excel(df_raw, binned_df, cycle_rows: set, pivot_name: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%d-%m-%Y--%H-%M-%S')
    xlsx_path = out_dir / f"{pivot_name}_excel_{timestamp}.xlsx"
    try:
        import xlsxwriter
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
            df_raw.to_excel(writer, sheet_name="Raw", index=False)
            binned_df.to_excel(writer, sheet_name="Binned", index=False)
            if cycle_rows:
                workbook = writer.book
                worksheet = writer.sheets["Raw"]
                bold_fmt = workbook.add_format({"bold": True})
                for orig_idx in sorted(cycle_rows):
                    if orig_idx in df_raw.index:
                        pos = df_raw.index.get_loc(orig_idx)
                        worksheet.set_row(pos + 1, None, bold_fmt)
        print(f"[OK] Excel: {xlsx_path}")
    except Exception as e:
        print(f"[WARN] Excel failed: {e}")

def save_cycle_csvs_from_info(cycles_info, cycles_dir: Path, pivot_blade: float):
    cycles_dir.mkdir(parents=True, exist_ok=True)
    by_pivot = {}
    for c in cycles_info:
        by_pivot.setdefault(c["pivo_id"], []).append(c)
    for pivot_val, cycles in by_pivot.items():
        pivot_folder = cycles_dir / str(pivot_val)
        pivot_folder.mkdir(parents=True, exist_ok=True)
        for idx, c in enumerate(cycles, start=1):
            df_cycle = pd.DataFrame({
                "AngleDeg": np.arange(0, 360),
                "LaminaPerDegree": c["lamina_360"],
                "CycleIndex": [idx] * 360
            })
            fname = pivot_folder / f"{pivot_val}_cycle{idx}_{c['cycle_id']}.csv"
            df_cycle.to_csv(fname, index=False)

# ============================================================================
# STEP 7-8: DATABASE SAVE
# Purpose: Save INDIVIDUAL cycle data to PostgreSQL
# 
# CRITICAL: Each cycle is saved as a SEPARATE row with its OWN lamina_360 array
#           We are NOT saving accumulated/summed data
#           Each cycle's lamina_360 represents irrigation for THAT CYCLE ONLY
# ============================================================================

def save_cycles_to_database_from_info(cycles_info: list, db_config: dict = None, replace: bool = False):
    """
    Save individual cycles to PostgreSQL database.
    
    IMPORTANT: 
    - Each cycle is saved as a separate database row
    - lamina_360 contains THIS CYCLE'S data ONLY (not accumulated)
    - blade_factor is the actual value used for calculations (not hardcoded)
    
    Args:
        cycles_info: List of cycle dicts, each with independent lamina_360 array
        db_config: Database connection configuration
    """
    if db_config is None:
        db_config = {
            'host': 'localhost', 
            'database': 'irrigation_db', 
            'user': 'postgres', 
            'password': 'admin'
        }
    
    # Connect to database
    db = IrrigationDatabase(**db_config)
    if not db.connect():
        print("[ERROR] ❌ Database connection failed")
        return
    
    print(f"\n{'='*70}")
    print(f"DATABASE SAVE: {len(cycles_info)} cycles")
    print(f"{'='*70}")
    
    try:
        if replace:
            try:
                if hasattr(db, "run_query"):
                    db.run_query("DELETE FROM cycle_lamina_data;")
                elif hasattr(db, "conn"):
                    with db.conn.cursor() as cur:
                        cur.execute("DELETE FROM cycle_lamina_data;")
                        db.conn.commit()
                print("[DB] ⚠️ Cleared existing cycle data (replace mode)")
            except Exception as e:
                print(f"[DB] ⚠️ Could not clear table: {e}")


        for i, c in enumerate(cycles_info, 1):
            # -------------------------------------------------------------------------
            # PREPARE CYCLE DATA FOR DATABASE
            # Each cycle gets its own row with:
            # - Unique cycle_id
            # - Start/end timestamps
            # - Duration
            # - blade_factor (actual value used, NOT hardcoded)
            # - lamina_360: 360-element array with lamina for THIS CYCLE ONLY
            # -------------------------------------------------------------------------
            cycle_data = {
                'cycle_id': c['cycle_id'],              # Unique ID
                'pivo_id': c['pivo_id'],                # Pivot name
                'start_date': c['start_ts'],            # Cycle start time
                'end_date': c['end_ts'],                # Cycle end time
                'blade_factor': c['blade_factor'],      # Actual blade_factor used (not hardcoded!)
                'duration_minutes': int(c['duration_minutes']),  # Duration
                'lamina_360': c['lamina_360']           # THIS CYCLE'S lamina array (NOT accumulated!)
            }
            #print(cycle_data)
            #{', '.join([f'lamina_at_{i:03d} = EXCLUDED.lamina_at_{i:03d}' for i in range(360)])}
            if (int(cycle_data['duration_minutes'] == 360) and c['duration_minutes'] < 693):
                print("Escape")
                continue
                    
            # Insert/update this cycle in database
            db.insert_cycle_data(cycle_data)
            
             #Show sample for first cycle
            # if i == 1:
            #     print(f"\n[DB] Sample cycle saved:")
            #     print(f"  ID: {c['cycle_id']}")
            #     print(f"  Blade factor: {c['blade_factor']} mm")
            #     print(f"  Duration: {c['duration_minutes']:.0f} minutes")
            #     print(f"  Lamina array shape: {c['lamina_360'].shape}")
            #     print(f"  Lamina range: [{np.min(c['lamina_360']):.2f}, {np.max(c['lamina_360']):.2f}] mm/deg")
            #     print(f"  Total lamina in cycle: {np.sum(c['lamina_360']):.2f} mm")
        
        print(f"\n[DB] ✅ Successfully saved {len(cycles_info)} cycles")
        print(f"[DB] Each cycle stored independently (NOT accumulated)")
    
    except Exception as e:
        print(f"[DB ERROR] ❌ {e}")
    
    finally:
        db.disconnect()
        print(f"{'='*70}\n")

# ============================================================================
# STEP 9: VISUALIZATION - ⚠️ NEVER CHANGE THESE FUNCTIONS! ⚠️
# ============================================================================

def pivot_heatmap(laminas_mm, titulo="Distribuição da Lâmina - Heatmap"):
    laminas_mm = np.asarray(laminas_mm, dtype=float)  # garante array float
    laminas_norm = (laminas_mm - np.min(laminas_mm)) / (np.max(laminas_mm) - np.min(laminas_mm) + 1e-12)  # normaliza
    angles = np.deg2rad(np.arange(0, 360))  # ângulos em rad
    n_rings = 100  # número de anéis para visual
    radii = np.linspace(0.3, 1.0, n_rings)  # raio mínimo->máximo
    theta, r = np.meshgrid(angles, radii)
    z = np.tile(laminas_norm, (n_rings, 1))  # repete linha normalizada
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    im = ax.pcolormesh(theta, r, z, cmap=plt.cm.viridis_r, shading="auto", vmin=0, vmax=1)
    ax.set_theta_zero_location("E")  # zero na direita (east)
    ax.set_theta_direction(1)  # sentido horário
    ax.grid(True)
    ax.set_xticks([])  # sem ticks
    ax.set_yticks([])
    ax.set_title(titulo, va="bottom", fontsize=14, weight="bold")
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis_r, norm=plt.Normalize(vmin=np.min(laminas_mm), vmax=np.max(laminas_mm)))
    cbar = fig.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label("Lâmina acumulada (mm)", rotation=270, labelpad=20)
    return fig, ax

def pivot_bar_chart(laminas_mm, titulo="Lâmina acumulada por faixa angular", bottom=0.3, bar_scale="linear"):
    laminas_mm = np.array(laminas_mm, dtype=float)
    if bar_scale == "sqrt" and np.max(laminas_mm) > 0:
        laminas_display = np.sqrt(laminas_mm / np.max(laminas_mm)) * 0.6  # sqrt scaling
    elif bar_scale == "log" and np.max(laminas_mm) > 0:
        laminas_display = np.log1p(laminas_mm) / np.log1p(np.max(laminas_mm)) * 0.6  # log scaling
    else:
        max_val = np.max(laminas_mm) if np.max(laminas_mm) > 0 else 1.0
        laminas_display = (laminas_mm / max_val) * 0.6  # linear scaling
    n_bins = len(laminas_mm)
    theta = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
    width = 2 * np.pi / n_bins
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    bars = ax.bar(theta, laminas_display, width=width * 0.95, bottom=bottom, edgecolor="white", linewidth=0.3, alpha=0.9)
    ax.grid(True)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{titulo} (scale: {bar_scale})", pad=20, fontsize=14, weight="bold")
    plt.tight_layout()
    return fig, ax

# Add this new function after the safe_parse_timestamp function (around line 250)

def filter_cycles_by_daterange(cycles_info: list, start_date: str = None, end_date: str = None) -> list:
    """
    Filter cycles based on start and end date range.
    
    Args:
        cycles_info: List of cycle dictionaries
        start_date: Start date string in format 'DD-MM-YYYY' or 'YYYY-MM-DD'
        end_date: End date string in format 'DD-MM-YYYY' or 'YYYY-MM-DD'
    
    Returns:
        Filtered list of cycles within the date range
    """
    if not cycles_info:
        return []
    
    # Parse date strings
    start_dt = None
    end_dt = None
    
    if start_date:
        try:
            # Try DD-MM-YYYY format first
            start_dt = datetime.strptime(start_date, "%d-%m-%Y")
        except ValueError:
            try:
                # Try YYYY-MM-DD format
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                print(f"[WARN] Invalid start_date format: {start_date}. Use DD-MM-YYYY or YYYY-MM-DD")
                return cycles_info
    
    if end_date:
        try:
            # Try DD-MM-YYYY format first
            end_dt = datetime.strptime(end_date, "%d-%m-%Y")
            # Set to end of day
            end_dt = end_dt.replace(hour=23, minute=59, second=59)
        except ValueError:
            try:
                # Try YYYY-MM-DD format
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                end_dt = end_dt.replace(hour=23, minute=59, second=59)
            except ValueError:
                print(f"[WARN] Invalid end_date format: {end_date}. Use DD-MM-YYYY or YYYY-MM-DD")
                return cycles_info
    
    # Filter cycles
    filtered = []
    total_cycles = len(cycles_info)
    
    for cycle in cycles_info:
        cycle_start = cycle.get('start_ts')
        
        # Skip cycles with invalid timestamps
        if pd.isna(cycle_start):
            continue
        
        # Check if cycle is within date range
        if start_dt and cycle_start < start_dt:
            continue
        if end_dt and cycle_start > end_dt:
            continue
        
        filtered.append(cycle)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"DATE RANGE FILTERING")
    print(f"{'='*70}")
    if start_dt:
        print(f"Start date: {start_dt.strftime('%d-%m-%Y %H:%M:%S')}")
    else:
        print(f"Start date: No limit (all cycles from beginning)")
    
    if end_dt:
        print(f"End date:   {end_dt.strftime('%d-%m-%Y %H:%M:%S')}")
    else:
        print(f"End date:   No limit (all cycles until end)")
    
    print(f"\nTotal cycles before filter: {total_cycles}")
    print(f"Cycles within date range:   {len(filtered)}")
    print(f"Cycles excluded:            {total_cycles - len(filtered)}")
    print(f"{'='*70}\n")
    
    return filtered

# ============================================================================
# STEP 10: CLI MAIN
# Purpose: Command-line interface with Click
# ============================================================================

@click.command()
@click.option('--replace/--no-replace', default=False,
              help="--replace: delete all cycles and insert new (fresh). "
                   "--no-replace: keep update/insert behavior (default).")
@click.option("--root", default="./resources/logs", type=click.Path(path_type=Path))
@click.option("--pivots", multiple=True, default=["agrocangaia2"], help="Which pivots to parse")
@click.option("--csvfile", type=click.Path(path_type=Path), default=Path("./resources/logs/logsAgroCangaiaCyclesAcumulado.csv"))
@click.option("--export-csv", type=click.Path(path_type=Path), default=None, flag_value=Path("./resources/outputCSV"))
@click.option("--export-excel", type=click.Path(path_type=Path), default=None, flag_value=Path("./resources/outputCSV"))
@click.option("--export-cycles", type=click.Path(path_type=Path), default=None, flag_value=Path("./resources/outputCSV"))
@click.option("--source", type=click.Choice(["csv", "logs"]), default="logs")
@click.option("--glitch-threshold", default=40.0, type=float, help="Angle glitch detection threshold")
@click.option("--pivot-blade", default=5.46, type=float, help="Blade factor in mm (default: 5.46)")
@click.option("--bar-scale", type=click.Choice(["linear", "sqrt", "log"]), default="linear")
@click.option("--save-dir", type=click.Path(path_type=Path), default=Path("./resources/imgs"))
@click.option("--save-database", is_flag=True, help="Save cycles to PostgreSQL database")
@click.option("--start-date", type=str, default=None, 
              help="Start date for filtering cycles (format: DD-MM-YYYY or YYYY-MM-DD). Only cycles starting on or after this date will be saved to database.")
@click.option("--end-date", type=str, default=None,
              help="End date for filtering cycles (format: DD-MM-YYYY or YYYY-MM-DD). Only cycles starting on or before this date will be saved to database.")

def main(replace, root, pivots, csvfile, export_csv, export_excel, export_cycles, source, pivot_blade, bar_scale, save_dir, save_database, glitch_threshold, start_date, end_date):
    """
    Main entry point for irrigation analysis.
    
    Flow:
    1. Parse logs or load CSV
    2. Filter glitches
    3. Sort by timestamp
    4. Detect cycles
    5. Calculate lamina
    6. Export results
    7. Save to database (optional)
    8. Generate visualizations
    """
    
    # STEP 1-2: Load and filter data
    if source == "csv" and csvfile and csvfile.exists():
        print(f"Loading CSV: {csvfile}")
        df = pd.read_csv(csvfile, sep=",", encoding="utf-8", engine="python")
        df = df.rename(columns=lambda c: c.strip().lower())
        if "grau" not in df.columns or "lamina acumulada" not in df.columns:
            raise SystemExit("CSV must contain 'grau' and 'lamina acumulada' columns")
        angles = pd.to_numeric(df["grau"], errors="coerce") % 360
        laminas = pd.to_numeric(df["lamina acumulada"], errors="coerce").fillna(0)
        lamina_360 = np.zeros(360)
        for angle, lamina in zip(angles, laminas):
            if not np.isnan(angle):
                lamina_360[int(angle)] = lamina
        pivot_name = "CSV_Data"
        cycles_info = []
    
    elif source == "logs":
        print(f"Loading logs from: {root}")
        df = parse_all_logs(Path(root), list(pivots))
        if df.empty:
            raise SystemExit("No valid rows parsed from logs")
        
        # STEP 3: Sort by timestamp
        df["Timestamp"] = df.apply(lambda r: safe_parse_timestamp(r.get("DtBe", ""), r.get("HourBe", "")), axis=1)
        df = df.sort_values(["Timestamp"]).reset_index(drop=True)
        
        # 🔍 LOGGING: df.to_csv("debug_04_sorted.csv", index=False)
        
        pivot_name = str(df["Pivot"].iloc[0])
        
        # STEP 4-5: Detect cycles and calculate lamina
        lamina_360, cycles_info = compute_all_cycles_and_lamina(df, pivot_blade, glitch_threshold)
    else:
        raise SystemExit("Invalid source selected")
    
    # STEP 6: Prepare export dataframes
    if source == "logs":
        df_raw = df.copy()
        cmd_digits = (df_raw["Command"].astype(str).fillna("").str.replace(r"\D", "", regex=True).str.zfill(3).str.slice(0, 3))
        df_raw["Direction"] = pd.to_numeric(cmd_digits.str[0], errors="coerce").fillna(0).astype(int)
        df_raw["Water"] = pd.to_numeric(cmd_digits.str[1], errors="coerce").fillna(0).astype(int)
        df_raw["Power"] = pd.to_numeric(cmd_digits.str[2], errors="coerce").fillna(0).astype(int)
        df_raw["Direction"] = df_raw["Direction"].map({8: "OFF", 3: "Forward", 4: "Reverse"})
        df_raw["Water"] = df_raw["Water"].map({5: "OFF", 7: "Pressuring", 6: "ON"})
        df_raw["Power"] = df_raw["Power"].map({1: "ON", 2: "OFF"})
        binned_df = pd.DataFrame({"AngleDeg": np.arange(0, 360, dtype=int), "Lamina": lamina_360})
    else:
        df_raw = pd.DataFrame()
        binned_df = pd.DataFrame({"AngleDeg": np.arange(0, 360, dtype=int), "Lamina": lamina_360})
    
    # STEP 7: Export files
    if export_csv:
        export_to_csv(df_raw, binned_df, pivot_name, Path(export_csv))
    if export_excel:
        cycle_rows = set()
        for c in cycles_info:
            cycle_rows.update(c.get("indices", []))
        export_to_excel(df_raw, binned_df, cycle_rows, pivot_name, Path(export_excel))
    if export_cycles:
        save_cycle_csvs_from_info(cycles_info, Path(export_cycles), pivot_blade)
    
    # # STEP 8: Save to database
    # if save_database and cycles_info:
    #     print("[DB] Saving cycles to database...")
    #     db_config = {'host': 'localhost', 'database': 'irrigation_db', 'user': 'postgres', 'password': 'admin'}
    #     save_cycles_to_database_from_info(cycles_info, db_config, replace)
    
    # STEP 8: Save to database (with optional date filtering)
    if save_database and cycles_info:
        print("[DB] Preparing to save cycles to database...")
        
        # Filter by date range if specified
        cycles_to_save = cycles_info
        if start_date or end_date:
            cycles_to_save = filter_cycles_by_daterange(cycles_info, start_date, end_date)
            
            if not cycles_to_save:
                print("[DB] ⚠️  No cycles found in specified date range. Skipping database save.")
            else:
                print(f"[DB] Saving {len(cycles_to_save)} cycles (filtered by date range)...")
                db_config = {'host': 'localhost', 'database': 'irrigation_db', 'user': 'postgres', 'password': 'admin'}
                save_cycles_to_database_from_info(cycles_to_save, db_config, replace)
        else:
            print(f"[DB] Saving all {len(cycles_to_save)} cycles (no date filter applied)...")
            db_config = {'host': 'localhost', 'database': 'irrigation_db', 'user': 'postgres', 'password': 'admin'}
            save_cycles_to_database_from_info(cycles_to_save, db_config, replace)

    # STEP 9: Generate visualizations
    out_folder = Path(save_dir) / pivot_name
    out_folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%d-%m-%Y--%H-%M-%S')
    
    print(f"\nSummary: min={np.min(lamina_360):.2f}, max={np.max(lamina_360):.2f}, mean={np.mean(lamina_360):.2f}")
    
    fig1, _ = pivot_bar_chart(lamina_360, bar_scale=bar_scale)
    fname1 = out_folder / f"bar_chart_{pivot_name}_{source}_{bar_scale}_{timestamp}.png"
    fig1.savefig(fname1, dpi=200, bbox_inches="tight")
    print(f"Bar chart saved: {fname1}")
    
    fig2, _ = pivot_heatmap(lamina_360)
    fname2 = out_folder / f"heatmap_{pivot_name}_{source}_{timestamp}.png"
    fig2.savefig(fname2, dpi=200, bbox_inches="tight")
    print(f"Heatmap saved: {fname2}")
    
    plt.show()

if __name__ == "__main__":
    main()