"""
QRC Visualization Script
========================
Author: Marcin Plodzien

This script generates standard analysis plots from the summarized results (`analysis_summary.pkl`).
It is the final step in the pipeline: Simulation -> Analysis -> Visualization.

Plots Generated:
----------------
1.  **Total Capacity vs Time Evolution** (`Capacity_vs_Time.png`):
    -   X-axis: $t_{evol}$ (Log scale).
    -   Y-axis: $\sum C(\\tau)$ (Total Memory Capacity).
    -   Hue: Data Input Type / h_mag.
    -   Style: Topology.
    -   Purpose: Identifies the optimal timescale for the reservoir.

2.  **Capacity Profiles** (`Profile_*.png`):
    -   X-axis: Lag $\\tau$.
    -   Y-axis: $C(\\tau)$.
    -   Curves: Different $t_{evol}$.
    -   Purpose: Inspects the memory fading profile (short-term vs long-term memory).

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
 
import sys
import importlib.util

import sys
import importlib.util

# ==============================================================================
# SPYDER / IDE CONFIGURATION
# ==============================================================================
# Set this to a path string (or config file) to run directly without CLI args
# MANUAL_PATH = None 
# Example: 
MANUAL_PATH = "results/QRC_TFIM_ZZ_X_N_rails_3_L_2"

# ==============================================================================
# LOAD PATH Logic
# ==============================================================================
# 1. CLI Argument -> 2. Manual Path -> 3. Default Config
arg = sys.argv[1] if len(sys.argv) > 1 else (MANUAL_PATH or "01_config_qrc_ladder.py")

if arg.endswith('.py'):
    print(f"Loading Configuration from: {arg}")
    spec = importlib.util.spec_from_file_location("config", arg)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    RESULTS_DIR = config.RESULTS_DIR
else:
    # Assume it is a direct directory path
    RESULTS_DIR = arg

print(f"Plotting targeting root: {RESULTS_DIR}")

ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis_results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)

PLOTS_DIR = FIGURES_DIR # Output for plots


def find_summary_files(analysis_dir):
    """Finds all analysis_summary_*.pkl files in the directory."""
    files = []
    if not os.path.exists(analysis_dir):
        return []
        
    for f in os.listdir(analysis_dir):
        if f.startswith("analysis_summary_") and f.endswith(".pkl"):
             files.append(os.path.join(analysis_dir, f))
             
    # Fallback to standard if no specific ones found
    if not files and os.path.exists(os.path.join(analysis_dir, "analysis_summary.pkl")):
         files.append(os.path.join(analysis_dir, "analysis_summary.pkl"))
         
    return sorted(files)

def standardize_columns(df):
    """Maps columns from Standalone format to Analysis format."""
    # Standalone -> Analysis
    mapping = {
        'HamType': 'topology',
        'T_evol': 't_evol',
        'h_mag': 'h_mag',
        'Data': 'data_input_type',
        'State': 'input_state_type', # Added mapping
        # 'Tau': 'Tau',
        # 'Capacity': 'Capacity'
    }
    df = df.rename(columns=mapping)
    # Ensure meas_config exists
    if 'meas_config' not in df.columns:
        df['meas_config'] = 'All'
    return df

def plot_total_capacity_vs_t_evol(df, suffix=""):
    """
    Plots Total Memory Capacity vs Evolution Time.
    Handles both Precomputed (Long) and Standard (Wide) formats.
    Facets by Measurement Config.
    """
    print("Plotting Total Capacity vs t_evol...")
    
    # Check Measure Configs
    meas_configs = sorted(df['meas_config'].unique())
    print(f"Configs found: {meas_configs}")
    
    # 1. Compute Total Capacity if not present
    if 'total_capacity' not in df.columns:
        if 'Capacity' in df.columns:
            # Aggregate Long Format: Sum(Capacity) per Group
            grp_cols = ['topology', 't_evol', 'h_mag', 'data_input_type', 'input_state_type', 'field_disorder', 'meas_config']
            grp_cols = [c for c in grp_cols if c in df.columns]
            df_agg = df.groupby(grp_cols)['Capacity'].sum().reset_index()
            df_agg = df_agg.rename(columns={'Capacity': 'total_capacity'})
        else:
            print("Error: DataFrame missing 'total_capacity' or 'Capacity' column.")
            return
    else:
        df_agg = df

    # Data Prep
    df_agg['t_evol'] = pd.to_numeric(df_agg['t_evol'])
    df_agg = df_agg.sort_values('t_evol')
    
    # Create Composite Label
    if 'input_state_type' not in df_agg.columns: df_agg['input_state_type'] = 'Unknown'
    if 'h_mag' not in df_agg.columns: df_agg['h_mag'] = 0.0
    if 'j_rail_left_name' not in df_agg.columns: df_agg['j_rail_left_name'] = '?'
    
    if 'field_disorder' not in df_agg.columns: df_agg['field_disorder'] = True
    df_agg['DisorderType'] = df_agg['field_disorder'].apply(lambda x: 'Random' if x else 'Uniform')

    # Condition: Input State + Topology Variant + Measurement Config
    # If we have multiple meas configs, we should probably create separate plots or facet rows/cols.
    # Let's facet columns by Meas Config, Rows by Data Input Type
    
    if 'data_input_type' not in df_agg.columns: df_agg['data_input_type'] = 'Unknown'
    dtypes = sorted(df_agg['data_input_type'].unique())
    
    # ==========================================================================
    # 1. TOTAL CAPACITY VS TIME
    # ==========================================================================
    # User Request: Single panel with all measurement lines.
    
    # Filter for valid data
    df_agg = df.dropna(subset=['total_capacity'])
    
    # We might have multiple DataInputTypes or H_mags. 
    # Let's facet by DataInputType (Rows) and H_mag (Cols) if multiple exist.
    # But usually these are fixed.
    
    # Unique conditions to Facet By (excluding t_evol and meas_config)
    
    # Setup Figure
    # Simplest: FacetGrid
    
    g = sns.FacetGrid(df_agg, row='data_input_type', col='h_mag', height=6, aspect=1.5, sharey=True)
    
    g.map_dataframe(
        sns.lineplot, 
        x='t_evol', 
        y='total_capacity', 
        hue='meas_config', # Lines by Measurement Config
        style='meas_config', # Differentiate by style too
        markers=True, 
        dashes=False,
        err_style='band' # If std exists (seaborn handles it if replicates exist, else we rely on precomputed)
    )
    
    # Manual error bars if precomputed std exists in single row per condition
    # Seaborn lineplot aggregates raw data. If we passed aggregated data, it just plots points.
    # If we have 'total_capacity_std', we might need custom mapping.
    # But for now, simple lineplot is sufficient.
    
    g.set(xscale='log')
    g.set_axis_labels("Evolution Time ($t_{evol}$)", "Total Memory Capacity $\sum C(\\tau)$")
    g.add_legend(title="Measurement")
    g.fig.suptitle("Total Capacity vs Time (All Measurements)", y=1.02)
    
    out_path = f"{PLOTS_DIR}/Capacity_vs_Time_Combined{suffix}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close()

def plot_capacity_profiles(df, suffix=""):
    """
    Plots Capacity Profile C(tau).
    Facets by t_evol (Panels).
    Hue by meas_config.
    """
    print("Plotting Capacity Profiles...")
    
    if 'caps_per_tau' not in df.columns:
        print("Skipping profiles: 'caps_per_tau' not found.")
        return

    # Expand 'caps_per_tau' (list) into rows for Seaborn
    # Need columns: t_evol, meas_config, tau, capacity
    
    expanded_rows = []
    
    # Iterate over aggregated rows
    for _, row in df.iterrows():
        caps = row['caps_per_tau']
        if isinstance(caps, (list, np.ndarray)):
            for tau, val in enumerate(caps):
                expanded_rows.append({
                    't_evol': row['t_evol'],
                    'meas_config': row['meas_config'],
                    'h_mag': row['h_mag'],
                    'tau': tau,
                    'capacity': val
                })
                
    if not expanded_rows:
        return
        
    long_df = pd.DataFrame(expanded_rows)
    long_df['meas_label'] = long_df['meas_config'].apply(lambda x: f"M{x}" if str(x).startswith('_') else str(x))
    
    # Facet by t_evol
    # If many t_evols, use col_wrap
    t_vals = sorted(long_df['t_evol'].unique())
    n_t = len(t_vals)
    
    g = sns.FacetGrid(long_df, col='t_evol', col_wrap=4, height=3, aspect=1.2, sharex=True, sharey=True, hue='meas_label')
    
    g.map_dataframe(
        sns.lineplot,
        x='tau',
        y='capacity'
    )
    
    g.set_axis_labels("Lag $\\tau$", "Capacity $C(\\tau)$")
    g.add_legend(title="Measurement")
    g.fig.suptitle("Capacity Profiles by Evolution Time", y=1.02)
    
    out_path = f"{PLOTS_DIR}/Capacity_Profiles_Grid{suffix}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close()


def plot_prediction_grid(df, suffix=""):
    """
    Plots Grid of Prediction Traces (True vs Predicted).
    Rows: Selected t_evol (up to 4).
    Cols: Single vs Batch Data Input.
    """
    print("Plotting Prediction Grids...")
    
    # Check if we have prediction traces
    if 'example_y_test' not in df.columns:
        print("No prediction traces found (example_y_test). Skipping grid plot.")
        return

    # Grouping (excluding data_input_type which determines columns)
    grouping_cols = ['input_state_type', 'h_mag', 'topology', 
                     'j_rail_left_name', 'j_rail_right_name', 'j_rung_name', 'field_disorder']
    
    # Ensure cols exist
    for c in grouping_cols:
        if c not in df.columns: df[c] = '?'
        
    unique_groups = df[grouping_cols].drop_duplicates()
    
    # Define Data Input Types (Columns)
    dtypes = ['single_data_input', 'batch_data_input']
    
    for _, row in unique_groups.iterrows():
        s_type = row['input_state_type']
        h_mag = row['h_mag']
        topo = row['topology']
        jrl = row['j_rail_left_name']
        jrr = row['j_rail_right_name']
        jrung = row['j_rung_name']
        is_disordered = row['field_disorder']
        dis_str = "Random" if is_disordered else "Uniform"
        
        # Filter for this group
        sub_group = df[(df['input_state_type'] == s_type) &
                       (df['h_mag'] == h_mag) & 
                       (df['topology'] == topo) &
                       (df['j_rail_left_name'] == jrl) &
                       (df['j_rail_right_name'] == jrr) &
                       (df['j_rung_name'] == jrung) &
                       (df['field_disorder'] == is_disordered)]
                       
        if sub_group.empty: continue
        
        # Select 4 representative t_evols
        all_ts = sorted(sub_group['t_evol'].unique())
        if not all_ts: continue
        
        # Selection logic: First, Last, and 2 evenly spaced in between
        if len(all_ts) <= 4:
            selected_ts = all_ts
        else:
            indices = np.linspace(0, len(all_ts)-1, 4, dtype=int)
            selected_ts = [all_ts[i] for i in indices]
            
        n_rows = len(selected_ts)
        n_cols = len(dtypes)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows), sharex=True, sharey=True, squeeze=False)
        
        for r_idx, t_val in enumerate(selected_ts):
            for c_idx, dtype in enumerate(dtypes):
                ax = axes[r_idx, c_idx]
                
                # Get Row Data
                record = sub_group[(sub_group['t_evol'] == t_val) & (sub_group['data_input_type'] == dtype)]
                
                if record.empty:
                    ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                    continue
                
                # Extract Traces (Assume scalar/single row per combination)
                rec = record.iloc[0]
                y_true = rec.get('example_y_test')
                y_pred = rec.get('example_y_pred')
                tau = rec.get('example_trace_tau', '?')
                
                if y_true is None or y_pred is None:
                     ax.text(0.5, 0.5, "No Traces", ha='center', va='center')
                     continue
                     
                # Plot Snippet
                limit = 50 # Reduced limit to make markers visible
                y_std = rec.get('example_y_pred_std')
                
                ax.plot(y_true[:limit], 'k-', marker='o', markersize=4, alpha=0.6, label='True')
                ax.plot(y_pred[:limit], 'r--', marker='x', markersize=4, alpha=0.8, label='Pred')
                
                if y_std is not None:
                     ax.fill_between(
                        range(len(y_pred[:limit])),
                        (y_pred - y_std)[:limit],
                        (y_pred + y_std)[:limit],
                        color='r',
                        alpha=0.2,
                        label='Std'
                     )
                
                if r_idx == 0:
                    ax.set_title(f"{dtype}")
                if c_idx == 0:
                    ax.set_ylabel(f"t_evol={t_val}")
                
                if r_idx == n_rows - 1 and c_idx == n_cols - 1:
                    ax.legend(loc='upper right', fontsize='small')
                    
        fig.suptitle(f"Prediction (tau={tau}) | {s_type} | h={h_mag} ({dis_str}) | {topo}\nL:{jrl}", fontsize=14)
        plt.tight_layout()
        
        fname = f"Prediction_Grid_{s_type}_h{h_mag}_{dis_str}_{topo}_L{jrl}{suffix}.png"
        plt.savefig(f"{PLOTS_DIR}/{fname}", dpi=300)
        print(f"Saved {PLOTS_DIR}/{fname}")
        plt.close()

def main():
    summary_files = find_summary_files(ANALYSIS_DIR)
    
    if not summary_files:
        print(f"No summary files found in {ANALYSIS_DIR}")
        return
        
    print(f"Found {len(summary_files)} summary files to plot: {[os.path.basename(f) for f in summary_files]}")
    
    for summary_path in summary_files:
        print(f"\n--- Plotting for: {os.path.basename(summary_path)} ---")
        try:
            df = pd.read_pickle(summary_path)
        except Exception as e:
            print(f"Error loading {summary_path}: {e}")
            continue
            
        if df.empty:
            print("DataFrame is empty. Skipping.")
            continue
            
        # Extract Config Name from filename
        # Format: analysis_summary_{CONF_NAME}.pkl
        basename = os.path.basename(summary_path)
        if basename == "analysis_summary.pkl":
            conf_suffix = ""
        else:
            # Strip prefix and extension
            conf_suffix = "_" + basename.replace("analysis_summary_", "").replace(".pkl", "")
            
        df = standardize_columns(df)
        
        # Pass suffix to plot functions to append to filename
        # Note: We need to update plot functions to accept suffix if not already handled.
        # Check plot functions first?
        # Let's assume user wants me to update them too, but for now just pass to logic.
        # Actually, plot functions hardcode output name. 
        # I MUST update them to accept suffix.
        
        plot_total_capacity_vs_t_evol(df, suffix=conf_suffix)
        plot_capacity_profiles(df, suffix=conf_suffix)
        plot_prediction_grid(df, suffix=conf_suffix)

if __name__ == "__main__":
    main()
