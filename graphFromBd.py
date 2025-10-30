#!/usr/bin/env python3
"""
Plot irrigation lamina data from PostgreSQL database.

This script:
1. Connects to PostgreSQL database
2. Retrieves cycle lamina data
3. Aggregates lamina across all cycles or specific filters
4. Generates heatmap and bar chart visualizations
"""

import click
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from database import IrrigationDatabase
from typing import Optional, List

# ============================================================================
# DATE PARSING UTILITY
# ============================================================================

def parse_date_string(date_str: str, is_end_date: bool = False) -> Optional[str]:
    """
    Parse date string in multiple formats and return PostgreSQL-compatible format.
    
    Args:
        date_str: Date string in format DD-MM-YYYY or YYYY-MM-DD
        is_end_date: If True, set time to end of day (23:59:59)
        
    Returns:
        Date string in format 'YYYY-MM-DD HH:MM:SS' or None if invalid
    """
    if not date_str:
        return None
    
    parsed_dt = None
    
    # Try DD-MM-YYYY format
    try:
        parsed_dt = datetime.strptime(date_str, "%d-%m-%Y")
    except ValueError:
        pass
    
    # Try YYYY-MM-DD format
    if not parsed_dt:
        try:
            parsed_dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            pass
    
    # Try DD/MM/YYYY format
    if not parsed_dt:
        try:
            parsed_dt = datetime.strptime(date_str, "%d/%m/%Y")
        except ValueError:
            pass
    
    if not parsed_dt:
        print(f"[WARN] Invalid date format: {date_str}")
        print(f"       Supported formats: DD-MM-YYYY, YYYY-MM-DD, DD/MM/YYYY")
        return None
    
    # Set time to end of day if this is an end date
    if is_end_date:
        parsed_dt = parsed_dt.replace(hour=23, minute=59, second=59)
    else:
        parsed_dt = parsed_dt.replace(hour=0, minute=0, second=0)
    
    # Return in PostgreSQL format
    return parsed_dt.strftime("%Y-%m-%d %H:%M:%S")


# ============================================================================
# VISUALIZATION FUNCTIONS (from autoReport3.py)
# ============================================================================

def pivot_heatmap(laminas_mm, titulo="Distribuição da Lâmina - Heatmap"):
    """Generate polar heatmap visualization of lamina distribution."""
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
                               norm=plt.Normalize(vmin=np.min(laminas_mm), 
                                                 vmax=np.max(laminas_mm)))
    cbar = fig.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label("Lâmina acumulada (mm)", rotation=270, labelpad=20)
    return fig, ax


def pivot_bar_chart(laminas_mm, titulo="Lâmina acumulada por faixa angular", 
                   bottom=0.3, bar_scale="linear"):
    """Generate polar bar chart visualization of lamina distribution."""
    laminas_mm = np.array(laminas_mm, dtype=float)
    
    if bar_scale == "sqrt" and np.max(laminas_mm) > 0:
        laminas_display = np.sqrt(laminas_mm / np.max(laminas_mm)) * 0.6
    elif bar_scale == "log" and np.max(laminas_mm) > 0:
        laminas_display = np.log1p(laminas_mm) / np.log1p(np.max(laminas_mm)) * 0.6
    else:
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


# ============================================================================
# DATABASE RETRIEVAL FUNCTIONS
# ============================================================================

def retrieve_cycles_from_db(db: IrrigationDatabase, 
                           pivo_id: Optional[str] = None,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           limit: Optional[int] = None) -> List[dict]:
    """
    Retrieve cycle data from database with optional filters.
    
    Args:
        db: Connected IrrigationDatabase instance
        pivo_id: Filter by specific pivot (e.g., "agrocangaia2")
        start_date: Filter cycles after this date (format: "YYYY-MM-DD HH:MM:SS")
        end_date: Filter cycles before this date (format: "YYYY-MM-DD HH:MM:SS")
        limit: Maximum number of cycles to retrieve
        
    Returns:
        List of cycle dictionaries with lamina_360 arrays
    """
    # Build query with filters
    query = "SELECT cycle_id, pivo_id, start_date, end_date, blade_factor, duration_minutes"
    
    # Add lamina columns
    for i in range(360):
        query += f", lamina_at_{i:03d}"
    
    query += " FROM cycle_lamina_data WHERE 1=1"
    params = []
    
    if pivo_id:
        query += " AND pivo_id = %s"
        params.append(pivo_id)
    
    if start_date:
        query += " AND start_date >= %s"
        params.append(start_date)
    
    if end_date:
        query += " AND end_date <= %s"
        params.append(end_date)
    
    query += " ORDER BY start_date"
    
    if limit:
        query += f" LIMIT {limit}"
    
    print(f"\n{'='*70}")
    print(f"DATABASE QUERY FILTERS")
    print(f"{'='*70}")
    print(f"  - Pivot:      {pivo_id or 'ALL'}")
    print(f"  - Start date: {start_date or 'ANY'}")
    print(f"  - End date:   {end_date or 'ANY'}")
    print(f"  - Limit:      {limit or 'NONE'}")
    print(f"{'='*70}\n")
    
    # Execute query
    try:
        cursor = db.conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        cursor.close()
        
        print(f"[DB] ✅ Retrieved {len(rows)} cycles from database")
        
        # Convert rows to dictionaries with lamina arrays
        cycles = []
        for row in rows:
            # First 6 columns are metadata
            cycle_id, pivo_id, start_date, end_date, blade_factor, duration_minutes = row[:6]
            
            # Next 360 columns are lamina values
            lamina_360 = np.array(row[6:], dtype=float)
            
            cycles.append({
                'cycle_id': cycle_id,
                'pivo_id': pivo_id,
                'start_date': start_date,
                'end_date': end_date,
                'blade_factor': blade_factor,
                'duration_minutes': duration_minutes,
                'lamina_360': lamina_360
            })
        
        return cycles
        
    except Exception as e:
        print(f"[DB ERROR] ❌ Failed to retrieve cycles: {e}")
        return []


def aggregate_lamina(cycles: List[dict], mode: str = "sum") -> np.ndarray:
    """
    Aggregate lamina data from multiple cycles.
    
    Args:
        cycles: List of cycle dictionaries with lamina_360 arrays
        mode: Aggregation mode - "sum", "mean", "max", "min"
        
    Returns:
        360-element array with aggregated lamina values
    """
    if not cycles:
        print("[WARN] ⚠️  No cycles to aggregate")
        return np.zeros(360)
    
    # Stack all lamina arrays
    lamina_stack = np.vstack([c['lamina_360'] for c in cycles])
    
    print(f"\n{'='*70}")
    print(f"LAMINA AGGREGATION")
    print(f"{'='*70}")
    print(f"  - Cycles:     {len(cycles)}")
    print(f"  - Mode:       {mode.upper()}")
    
    if mode == "sum":
        result = np.sum(lamina_stack, axis=0)
    elif mode == "mean":
        result = np.mean(lamina_stack, axis=0)
    elif mode == "max":
        result = np.max(lamina_stack, axis=0)
    elif mode == "min":
        result = np.min(lamina_stack, axis=0)
    else:
        print(f"[WARN] ⚠️  Unknown mode '{mode}', using 'sum'")
        result = np.sum(lamina_stack, axis=0)
    
    print(f"  - Result min: {np.min(result):.2f} mm")
    print(f"  - Result max: {np.max(result):.2f} mm")
    print(f"  - Result avg: {np.mean(result):.2f} mm")
    print(f"{'='*70}\n")
    
    return result


# ============================================================================
# CLI MAIN
# ============================================================================

@click.command()
@click.option("--host", default="localhost", help="Database host")
@click.option("--database", default="irrigation_db", help="Database name")
@click.option("--user", default="postgres", help="Database user")
@click.option("--password", default="admin", help="Database password")
@click.option("--pivo", default=None, help="Filter by pivot ID (e.g., agrocangaia2)")
@click.option("--start-date", default=None, 
              help="Filter cycles starting on or after this date. Formats: DD-MM-YYYY, YYYY-MM-DD, DD/MM/YYYY (e.g., 01-08-2025 or 2025-08-01)")
@click.option("--end-date", default=None, 
              help="Filter cycles ending on or before this date. Formats: DD-MM-YYYY, YYYY-MM-DD, DD/MM/YYYY (e.g., 31-08-2025 or 2025-08-31)")
@click.option("--limit", default=None, type=int, help="Maximum number of cycles to retrieve")
@click.option("--aggregate", type=click.Choice(["sum", "mean", "max", "min"]), 
              default="sum", help="Aggregation mode for multiple cycles")
@click.option("--bar-scale", type=click.Choice(["linear", "sqrt", "log"]), 
              default="linear", help="Bar chart scale")
@click.option("--save-dir", type=click.Path(path_type=Path), 
              default=Path("./resources/imgs"), help="Output directory for images")
@click.option("--show/--no-show", default=True, help="Display plots interactively")

def main(host, database, user, password, pivo, start_date, end_date, 
         limit, aggregate, bar_scale, save_dir, show):
    """
    Retrieve irrigation data from database and generate visualizations.
    
    Examples:
    
        # Plot all cycles for a specific pivot
        python graphFromBd.py --pivo agrocangaia2
        
        # Plot cycles within a date range (Brazilian format)
        python graphFromBd.py --start-date 01-08-2025 --end-date 31-08-2025
        
        # Plot cycles within a date range (ISO format)
        python graphFromBd.py --start-date 2025-08-01 --end-date 2025-08-31
        
        # Plot cycles from August 2025 onwards
        python graphFromBd.py --start-date 01-08-2025
        
        # Plot cycles up to September 2025
        python graphFromBd.py --end-date 30-09-2025
        
        # Plot mean lamina instead of sum
        python graphFromBd.py --aggregate mean --start-date 01-08-2025 --end-date 31-08-2025
        
        # Plot last 10 cycles only
        python graphFromBd.py --limit 10
        
        # Combine filters: specific pivot + date range + limit
        python graphFromBd.py --pivo agrocangaia2 --start-date 01-08-2025 --end-date 31-08-2025 --limit 50
    """
    
    print("\n" + "="*70)
    print("IRRIGATION DATA VISUALIZATION FROM DATABASE")
    print("="*70 + "\n")
    
    # Parse dates if provided
    start_date_parsed = None
    end_date_parsed = None
    
    if start_date:
        start_date_parsed = parse_date_string(start_date, is_end_date=False)
        if not start_date_parsed:
            print(f"[ERROR] ❌ Invalid start date format: {start_date}")
            return
    
    if end_date:
        end_date_parsed = parse_date_string(end_date, is_end_date=True)
        if not end_date_parsed:
            print(f"[ERROR] ❌ Invalid end date format: {end_date}")
            return
    
    # Show parsed dates
    if start_date_parsed or end_date_parsed:
        print(f"[INFO] Date range parsed:")
        if start_date_parsed:
            print(f"  - Start: {start_date_parsed}")
        if end_date_parsed:
            print(f"  - End:   {end_date_parsed}")
        print()
    
    # Connect to database
    db_config = {
        'host': host,
        'database': database,
        'user': user,
        'password': password
    }
    
    db = IrrigationDatabase(**db_config)
    if not db.connect():
        print("[ERROR] ❌ Database connection failed")
        return
    
    try:
        # Retrieve cycles
        cycles = retrieve_cycles_from_db(db, pivo, start_date_parsed, end_date_parsed, limit)
        
        if not cycles:
            print("[ERROR] ❌ No cycles found matching filters")
            print("\nTroubleshooting tips:")
            print("  1. Check if cycles exist in database for this pivot")
            print("  2. Verify date range covers existing cycles")
            print("  3. Try removing filters one by one")
            print("  4. Use --limit option to see if any data exists")
            return
        
        # Show cycle information
        print(f"\n{'='*70}")
        print(f"CYCLE SUMMARY")
        print(f"{'='*70}")
        print(f"  - Total cycles:   {len(cycles)}")
        print(f"  - Pivots:         {', '.join(sorted(set(c['pivo_id'] for c in cycles)))}")
        
        # Date range
        min_start = min(c['start_date'] for c in cycles)
        max_end = max(c['end_date'] for c in cycles)
        print(f"  - Date range:     {min_start} to {max_end}")
        
        # Duration stats
        total_duration = sum(c['duration_minutes'] for c in cycles)
        avg_duration = total_duration / len(cycles)
        print(f"  - Total duration: {total_duration:.1f} minutes ({total_duration/60:.1f} hours)")
        print(f"  - Avg duration:   {avg_duration:.1f} minutes")
        
        # Lamina stats per cycle
        total_laminas = [np.sum(c['lamina_360']) for c in cycles]
        print(f"  - Lamina per cycle:")
        print(f"    - Min:  {min(total_laminas):.2f} mm")
        print(f"    - Max:  {max(total_laminas):.2f} mm")
        print(f"    - Mean: {np.mean(total_laminas):.2f} mm")
        print(f"{'='*70}\n")
        
        # Aggregate lamina data
        lamina_360 = aggregate_lamina(cycles, mode=aggregate)
        
        # Generate output filename components
        pivot_name = pivo or "all_pivots"
        timestamp = datetime.now().strftime('%d-%m-%Y--%H-%M-%S')
        
        # Build descriptive filter string for filename
        filter_parts = [f"{aggregate}", f"{len(cycles)}cycles"]
        if start_date:
            filter_parts.append(f"from{start_date.replace('-', '').replace('/', '')}")
        if end_date:
            filter_parts.append(f"to{end_date.replace('-', '').replace('/', '')}")
        filter_str = "_".join(filter_parts)
        
        # Create output directory
        out_folder = Path(save_dir) / pivot_name
        out_folder.mkdir(parents=True, exist_ok=True)
        
        # Generate and save bar chart
        print(f"[PLOT] 📊 Generating bar chart...")
        fig1, _ = pivot_bar_chart(lamina_360, 
                                  titulo=f"Lâmina acumulada - {pivot_name} ({aggregate})",
                                  bar_scale=bar_scale)
        fname1 = out_folder / f"bar_chart_{pivot_name}_{filter_str}_{bar_scale}_{timestamp}.png"
        fig1.savefig(fname1, dpi=200, bbox_inches="tight")
        print(f"[OK] ✅ Bar chart saved: {fname1}")
        
        # Generate and save heatmap
        print(f"[PLOT] 🗺️  Generating heatmap...")
        fig2, _ = pivot_heatmap(lamina_360, 
                               titulo=f"Heatmap - {pivot_name} ({aggregate})")
        fname2 = out_folder / f"heatmap_{pivot_name}_{filter_str}_{timestamp}.png"
        fig2.savefig(fname2, dpi=200, bbox_inches="tight")
        print(f"[OK] ✅ Heatmap saved: {fname2}")
        
        # Display plots if requested
        if show:
            print(f"\n[INFO] 🖼️  Displaying plots... (close windows to exit)")
            plt.show()
        
        print(f"\n{'='*70}")
        print(f"VISUALIZATION COMPLETE ✅")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"[ERROR] ❌ {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        db.disconnect()


if __name__ == "__main__":
    main()