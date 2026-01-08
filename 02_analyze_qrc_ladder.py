"""
QRC Analysis Script: Capacity Calculation
=========================================
Author: Marcin Plodzien

This script processes the simulation results (collected pickle files) and computes the 
Memory/Prediction Capacity of the quantum reservoir.

Methodology:
------------
1.  **Data Loading**: Reads `collected_simulation_results.pkl`.
2.  **Grouping**: Groups data by hyperparameters (T_evol, h_mag, Topology, etc.).
3.  **Capacity Metric**:
    For each time lag `tau`:
    -   Splits reservoir state history X(t) and target signal y(t) into Train/Test sets.
    -   Fits a linear readout weights W via Ridge Regression: y_train = W * X_train.
    -   Predicts on test set: y_pred = W * X_test.
    -   Computes Squared Correlation: C(tau) = corr(y_test, y_pred)^2.
    
    Total Capacity C_total = Sum_{tau} C(tau).

4.  **Task Types**:
    -   **Memory**: Reconstruct past input u(t - tau).
    -   **Prediction**: Predict future input u(t + tau).
    -   (Note: For chaotic series like Santa Fe, 'prediction' is standard).

"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.linear_model import Ridge
from joblib import Parallel, delayed
import time

# ==============================================================================
# 1. CONFIGURATION
import sys
import importlib.util

# ==============================================================================
# SPYDER / IDE CONFIGURATION
# ==============================================================================
# Set this to a path string (or config file) to run directly without CLI args
MANUAL_PATH = None 
# Example: 
# MANUAL_PATH = "results/QRC_Ladder_Validation_Run_N_06_TFIM_ZZ_X"
MANUAL_PATH = "results/QRC_TFIM_ZZ_X_N_rails_3_L_2"

# ==============================================================================
# LOAD PATH Logic (Encapsulated)
# ==============================================================================
def get_config_context():
    """Determines paths based on CLI args or Manual Path."""
    arg = sys.argv[1] if len(sys.argv) > 1 else (MANUAL_PATH or "01_config_qrc_ladder.py")
    
    results_root = arg
    if arg.endswith('.py'):
        print(f"Loading Configuration from: {arg}")
        try:
            spec = importlib.util.spec_from_file_location("config", arg)
            config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config)
            results_root = config.RESULTS_DIR
        except Exception as e:
            print(f"Warning: Could not load config file {arg}: {e}")
            results_root = "results" # Fallback

    # Auto-detect 'data' subfolder
    if os.path.isdir(os.path.join(results_root, "data")):
        data_path = os.path.join(results_root, "data")
        print(f"Detected 'data' subfolder. Loading results from: {data_path}")
    else:
        data_path = results_root
        print(f"Analysis targeting root: {data_path}")

    # Output to subfolder in results
    output_dir = os.path.join(results_root, "analysis_results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = f"{output_dir}/analysis_summary.pkl"
    
    return data_path, output_dir, output_file

OUTPUT_FILE = None # Placeholder if needed globally, but better local.

# Analysis Hyperparameters
TAU_MAX = 50          # Maximum lag to compute capacity for
TASK_TYPE = 'prediction' # 'memory' or 'prediction'
TRAIN_RATIO = 0.8     # First 80% used for training readout (User suggested > 0.2)
REG_ALPHA = 1e-3      # Ridge Regression Regularization Strength

# Features Filtering
# We exclude non-observable columns from the Feature Matrix X
EXCLUDE_COLS = [
    'k', 's_k', 'Norm', 't_evol', 'h_mag', 'N', 'L', 
    'J_rung_name', 'config_name', 'input_state_type', 
    'data_input_type', 'realization', 'topology', 'dt',
    'j_rung_name', 'j_rail_left_name', 'j_rail_right_name',
    'field_disorder', 'measured_rails', 'param_names', 'integration_method'
]

# ==============================================================================
# 2. Measurement Configurations (Rail Subsets)
# Defines which rails are "read out" for the Ridge Regression.
MEASUREMENT_ON_RAIL = [
    #{"name": "0##", "measured_rails": [0       ]},       # Input rail only
    #{"name": "#1#", "measured_rails": [1       ]},       # Reservoir only
    # {"name": "##2", "measured_rails": [2       ]},       # Readout rail only
    #{"name": "0#2", "measured_rails": [0, 2    ]},       # Input + Readout
    # {"name": "#12", "measured_rails": [1, 2    ]},       # Reservoir + Readout
     {"name": "012", "measured_rails": [0, 1, 2 ]},       # All
]

# 3. Feature Selection Configuration
# ------------------------------------
# Defines which observables are used as features (columns in X) for the Ridge Regression.
# Format: 'COLUMN_NAME_MODE'
#   - COLUMN_NAME: The exact column name in the simulation DataFrame (e.g., 'Z_exp_val').
#   - MODE: '_local' (per qubit) or '_total' (summed over rails).
#
# Examples:
#   - 'Z_exp_val_local': Use Z expectation for each qubit.
#   - 'ZZ_exp_val_total': Sum of ZZ correlations over all measured rails.
#   - 'Z_std_val_local': Standard deviation of Z (derived if missing: sqrt(<Z^2> - <Z>^2)).
#
# 3. Feature Selection Configurations
# -------------------------------------
# Defines specific sets of features to analyze separately.
FEATURE_SELECTION_CONFIGS = [

    {
        "name": "[X,Y,Z]_local",
        "features": [
            'X_exp_val_local',  
            'Y_exp_val_local', 
            'Z_exp_val_local', 
        ]
    },
    
    {
        "name": "[X,Z]_local",
        "features": [
            'X_exp_val_local',  
            'Z_exp_val_local', 
        ]
    },

    {
        "name": "[Z]_local",
        "features": [
            'Z_exp_val_local', 
        ]
    },


    {
        "name": "[X,Y,Z]_local_[X2,Y2,Z2]_local",
        "features": [
            'X_exp_val_local', 'X^2_exp_val_local',  
            'Y_exp_val_local', 'Y^2_exp_val_local',  
            'Z_exp_val_local', 'Z^2_exp_val_local',  
        ]
    },

    {
        "name": "[X]_total_[X2]_total",
        "features": [
            'X_exp_val_total', 
            'X^2_exp_val_total',
        ]
    },

    {
        "name": "[Y]_total_[Y2]_total",
        "features": [
            'Y_exp_val_total', 
            'Y^2_exp_val_total',
        ]
    },

    {
        "name": "[Z]_total_[Z2]_total",
        "features": [
            'Z_exp_val_total', 
            'Z^2_exp_val_total',
        ]
    },
    {
        "name": "[X,Z]_total_[X2,Z2]_total",
        "features": [
            'X_exp_val_total', 'X^2_exp_val_total',
            'Z_exp_val_total', 'Z^2_exp_val_total',
        ]
    },
    

    {
        "name": "[X,Z]_local_[X2, Z2]_local",
        "features": [
            'X_exp_val_local', 
            'X^2_exp_val_local',
            'Z_exp_val_local', 
            'Z^2_exp_val_local',
        ]
    },    

    
]


# ==============================================================================
# 2. CORE LOGIC
# ==============================================================================

def calculate_capacity(group_df, tau_max=20, task_type='memory', return_traces_tau=None):
    """
    Computes the capacity profile C(tau).
    Optionally returns (y_test, y_pred) for a specific tau if return_traces_tau is set.
    
    Args:
        group_df (pd.DataFrame): Time series data.
        tau_max (int): Max lag.
        task_type (str): 'memory' or 'prediction'.
        return_traces_tau (int): If set, returns traces for this specific lag alongside capacities.
        
    Returns:
        np.array: Capacities C(tau).
        (Optional) tuple: (y_test, y_pred) if return_traces_tau is set.
    """
    # 1. Prepare Features & Targets
    # Sort by time step 'k' ensures temporal order
    group_df = group_df.sort_values('k')
    
    # Feature Selection Strategy
    # 1. Start with columns NOT in EXCLUDE_COLS
    initial_cols = [c for c in group_df.columns if c not in EXCLUDE_COLS]
    
    # 2. Further filter to ensure NUMERIC types
    # This catches ANY non-numeric column that wasn't explicitly excluded
    numeric_df = group_df[initial_cols].select_dtypes(include=[np.number])
    feature_cols = numeric_df.columns.tolist()
    
    X = numeric_df.values
    
    # Extract Target Signal S (Time x 1)
    # Ensure S is float
    S = group_df['s_k'].astype(float).values
    
    # Standardization (Z-score normalization)
    # Improves Ridge Regression stability
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-9
    X = (X - X_mean) / X_std
    
    n_samples = len(X)
    n_train = int(n_samples * TRAIN_RATIO)
    
    capacities = []
    collected_traces = None
    
    for tau in range(tau_max + 1):
        # 2. Construct Lagged Targets
        if task_type == 'memory':
            # Goal: Output(t) matches Input(t - tau)
            # X comes from reservoir at time t.
            # y matches input at time t - tau.
            
            if tau == 0:
                y = S
                X_curr = X
            else:
                # We align X[t] with S[t-tau]
                # Valid t range: [tau, N-1]
                X_curr = X[tau:]     # Reservoir states from t=tau to End
                y = S[:-tau]         # Inputs from t=0 to End-tau
                
        elif task_type == 'prediction':
            # Goal: Output(t) matches Input(t + tau)
            # Valid t range: [0, N-1-tau]
            
            if tau == 0:
                y = S
                X_curr = X
            else:
                # We align X[t] with S[t+tau]
                X_curr = X[:-tau]    # Reservoir states from t=0 to End-tau
                y = S[tau:]          # Inputs from t=tau to End
        
        # 3. Split Train/Test
        # Sequential split to respect time-series dependencies
        n_curr = len(X_curr)
        n_train_curr = int(n_curr * TRAIN_RATIO)
        
        X_train = X_curr[:n_train_curr]
        y_train = y[:n_train_curr]
        X_test = X_curr[n_train_curr:]
        y_test = y[n_train_curr:]
        
        # 4. Train Readout (Ridge Regression)
        # Linear map: y = W * x + b
        clf = Ridge(alpha=REG_ALPHA)
        clf.fit(X_train, y_train)
        
        # 5. Evaluate Performance
        # Metric: Squared Correlation Coefficient (R^2 in Capacity context)
        # Note: 'clf.score' returns Coefficient of Determination R^2 which can be negative.
        # Standard MC literature often defines Capacity = corr(y, y_pred)^2.
        
        y_pred = clf.predict(X_test)
        
        # Capture traces if requested
        if return_traces_tau is not None and tau == return_traces_tau:
            collected_traces = (y_test, y_pred)
        
        # Handle constant output edge case
        if np.std(y_test) < 1e-9 or np.std(y_pred) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(y_test, y_pred)[0, 1]
            
        cap_tau = corr**2
        capacities.append(cap_tau)
        
    if return_traces_tau is not None:
        return np.array(capacities), collected_traces
    return np.array(capacities)


# ==============================================================================


def process_group(name, group, feature_list):
    """
    Wrapper to process a group of realizations sharing the same hyperparameters.
    Iterates over MEASUREMENT_ON_RAIL to generate multiple capacity profiles.
    """
    realizations = group['realization'].unique()
    
    # Base Metadata (Take from first row)
    row = group.iloc[0]
    base_res = {
        't_evol': row['t_evol'],
        'h_mag': row['h_mag'],
        'input_state_type': row['input_state_type'],
        'data_input_type': row['data_input_type'],
        'topology': row['topology'],
        'j_rail_left_name': row.get('j_rail_left_name', 'Unknown'),
        'j_rail_right_name': row.get('j_rail_right_name', 'Unknown'),
        'j_rung_name': row.get('j_rung_name', 'Unknown'),
        'field_disorder': row['field_disorder'],
        'n_realizations': len(realizations)
    }
    
    results_for_group = []
    
    # Iterate over Measurement Strategies
    for meas_conf in MEASUREMENT_ON_RAIL:
        meas_name = meas_conf['name']
        target_rails = meas_conf['measured_rails']
        
        caps_list = []
        all_y_preds = []
        all_y_tests = []
        TRACE_TAU = 1 
        
        valid_realizations = 0
        
        for r in realizations:
            # Filter by realization
            rdf = group[group['realization'] == r]
            
            # Filter by Measured Rails
            # We only keep rows where rail_idx is in the target set
            rail_filtered = rdf[rdf['rail_idx'].isin(target_rails)].copy()
            
            if rail_filtered.empty:
                # print(f"DEBUG: rail_filtered empty for r={r}, target={target_rails}")
                continue
                
            # --- FEATURE CONSTRUCTION ---
            # We build the feature matrix based on passed feature_list
            # Format: 'ColName_local' or 'ColName_total'
            
            feature_dfs = []
            
            for f_req in feature_list:
                # Parse Request: e.g., 'Z_exp_val_local' -> base='Z_exp_val', mode='local'
                # Robust split: Look for trailing '_local' or '_total'
                
                mode = 'local' # Default
                if f_req.endswith('_local'):
                    mode = 'local'
                    base = f_req[:-6] # Remove '_local'
                elif f_req.endswith('_total'):
                    mode = 'total'
                    base = f_req[:-6] # Remove '_total'
                else:
                    mode = 'local'
                    base = f_req
                
                # Check for existence / derivation
                col_name = base
                
                # Lazy Derivation of STD / Squared features
                if col_name not in rail_filtered.columns:
                    
                    # 1. Derive STD from Momements: std = sqrt(<O^2> - <O>^2)
                    if '_std_val' in col_name:
                         root = col_name.replace('_std_val', '') # e.g. 'Z'
                         c_mean = f"{root}_exp_val"
                         c_sq   = f"{root}^2_exp_val"
                         
                         if c_mean in rail_filtered.columns and c_sq in rail_filtered.columns:
                              rail_filtered[col_name] = np.sqrt(np.maximum(0, rail_filtered[c_sq] - rail_filtered[c_mean]**2))
                              
                    # 2. Derive Squared from Std and Mean: <O^2> = std^2 + <O>^2
                    elif '^2_exp_val' in col_name:
                         root = col_name.replace('^2_exp_val', '') # e.g. 'Z'
                         c_mean = f"{root}_exp_val"
                         c_std  = f"{root}_std_val"
                         
                         if c_mean in rail_filtered.columns and c_std in rail_filtered.columns:
                              rail_filtered[col_name] = rail_filtered[c_std]**2 + rail_filtered[c_mean]**2

                if col_name not in rail_filtered.columns:
                    # print(f"Warning: Feature column '{col_name}' not found in rail_filtered.")
                    continue
                    
                if mode == 'local':
                    try:
                        pivoted = rail_filtered.pivot(index='k', columns=['rail_idx', 'qubit_idx'], values=col_name)
                    except ValueError:
                         pivoted = rail_filtered.groupby(['k', 'rail_idx', 'qubit_idx'])[col_name].mean().unstack(['rail_idx', 'qubit_idx'])
                         
                    # Rename columns to be unique
                    pivoted.columns = [f"{base}_R{c[0]}_Q{c[1]}" for c in pivoted.columns]
                    feature_dfs.append(pivoted)
                    
                elif mode == 'total':
                    total_series = rail_filtered.groupby('k')[col_name].sum()
                    total_df = total_series.to_frame(name=f"{base}_total")
                    feature_dfs.append(total_df)
            
            if not feature_dfs:
                # print(f"DEBUG: No features constructed for {meas_name}. Skipping.")
                continue
            
            if not feature_dfs:
                # No features valid?
                continue
                
            # Join all feature blocks on 'k'
            # We assume 'k' matches across them (same time steps)
            final_features = pd.concat(feature_dfs, axis=1)
            # Fill NaNs
            final_features = final_features.fillna(0.0)
            
            # Get Targets (s_k)
            # Just take from one of the pivots or the original group (careful with duplicates)
            # We need s_k aligned with k.
            target_df = rdf[['k', 's_k']].drop_duplicates('k').set_index('k').sort_index()
            
            # Combine
            wide_df = final_features.join(target_df)
            wide_df = wide_df.reset_index() # k back to column
            
            # ---------------------------
            
            cap_tau, traces = calculate_capacity(wide_df, tau_max=TAU_MAX, task_type=TASK_TYPE, return_traces_tau=TRACE_TAU)
            
            if not np.all(np.isnan(cap_tau)):
                caps_list.append(cap_tau)
                valid_realizations += 1
                if traces:
                     all_y_tests.append(traces[0])
                     all_y_preds.append(traces[1])
            
        if valid_realizations == 0:
            continue
            
        # Aggregate across realizations (Mean Capacity)
        caps_array = np.vstack(caps_list)
        mean_caps = np.mean(caps_array, axis=0)
        total_capacity = np.sum(mean_caps)
        std_total = np.std(np.sum(caps_array, axis=1)) # Std of the sums
        
        # Traces
        y_test_mean = None
        y_pred_mean = None
        if all_y_tests:
             # Stack and mean 
             try:
                 y_test_mean = np.mean(np.array(all_y_tests), axis=0)
                 y_pred_mean = np.mean(np.array(all_y_preds), axis=0)
             except ValueError:
                 # Length mismatch? Just take first.
                 y_test_mean = all_y_tests[0]
                 y_pred_mean = all_y_preds[0]

        # Store Result
        res = base_res.copy()
        res.update({
            'meas_config': meas_name,
            'total_capacity': total_capacity,
            'total_capacity_std': std_total,
            'caps_per_tau': mean_caps,
            'example_trace_tau': TRACE_TAU,
            'example_y_test': y_test_mean,
            'example_y_pred': y_pred_mean,
        })
        results_for_group.append(res)
        
    return results_for_group

# ==============================================================================
# 3. MAIN
# ==============================================================================

def load_data(path):
    """Loads data from single file or recursively from directory."""
    if os.path.isfile(path):
        print(f"Loading Results from File: {path}...")
        return pd.read_pickle(path)
    
    elif os.path.isdir(path):
        # 1. Check for consolidated file first
        consolidated_path = os.path.join(path, "collected_simulation_results.pkl")
        if os.path.exists(consolidated_path):
             print(f"Found consolidated results file: {consolidated_path}")
             return pd.read_pickle(consolidated_path)
             
        print(f"Loading Results recursively from Directory: {path}...")
        all_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".pkl") and "collected" not in file and "summary" not in file:
                     all_files.append(os.path.join(root, file))
        
        if not all_files:
            print("No suitable .pkl files found in directory.")
            return pd.DataFrame()
            
        print(f"Found {len(all_files)} partial result files.")
        dfs = []
        for f in all_files:
            try:
                # print(f"  Loading {f}...") # Verbose
                dfs.append(pd.read_pickle(f))
            except Exception as e:
                print(f"  Error loading {f}: {e}")
        
        if not dfs:
            return pd.DataFrame()
            
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Combined {len(combined_df)} rows from partial files.")
        return combined_df
    else:
        print(f"Error: Path {path} not found.")
        return pd.DataFrame()

def main():
    data_path, output_dir, output_file = get_config_context()
    df = load_data(data_path)
    
    if df.empty:
        print("No data loaded. Exiting.")
        return

    # SANITIZATION: Convert types
    print("Sanitizing DataFrame types...")
    # Explicit conversion for key grouping columns
    if 't_evol' in df.columns:
        df['t_evol'] = df['t_evol'].astype(float)
    if 'h_mag' in df.columns:
        df['h_mag'] = df['h_mag'].astype(float)
        
    for col in df.columns:
        if len(df) > 0:
            first_val = df[col].iloc[0]
            type_str = str(type(first_val))
            if 'jax' in type_str:
                df[col] = np.array(df[col].tolist())
                
    print(f"Total Rows: {df.shape[0]}")
    
    # ... (Pre-computed capacity check omitted for brevity) ...
    
    # Define Grouping Keys (Hyperparameters)
    # Update for Multi-Rail
    possible_group_cols = [
        't_evol', 'h_mag', 'input_state_type', 'data_input_type', 'topology',
        'j_rung_name', 'j_rails_name', 'field_disorder', 
        # Legacy/Other topology cols
        'j_rail_left_name', 'j_rail_right_name'
    ]
    
    group_cols = [c for c in possible_group_cols if c in df.columns]
    
    # Fill NaNs in grouping cols to avoid dropping rows
    for c in group_cols:
         df[c] = df[c].fillna('N/A')
    
    grouped = df.groupby(group_cols)
    print(f"Identified {len(grouped)} hyperparameter configurations.")
    
    # --- OUTER LOOP: Feature Configurations ---
    for f_config in FEATURE_SELECTION_CONFIGS:
        f_name = f_config['name']
        f_list = f_config['features']
        print(f"\n=== Running Analysis for Feature Config: {f_name} ===")
        
        results = []
        
        # Iterate Groups
        for name, group in grouped:
            # print(f"Processing Config: {name}") # Reduce verbosity
            # Debug: Check uniqueness of t_evol within group
            uniques = group['t_evol'].unique()
            if len(uniques) > 1:
                print(f"WARNING: Group {name} contains multiple t_evol values: {uniques}. This will cause duplicates!")
                continue     
            
            res_list = process_group(name, group, feature_list=f_list)
            if res_list:
                results.extend(res_list)
        
        # Formatting Output
        summary_df = pd.DataFrame(results)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Filename safe encoding
        # User requested dict name encoded in filename
        safe_name = f_name.replace('[', '').replace(']', '').replace(',', '').replace(' ', '_')
        out_filename = f"analysis_summary_{f_name}.pkl"
        out_path = os.path.join(output_dir, out_filename)
        
        summary_df.to_pickle(out_path)
        
        print(f"Analysis for '{f_name}' complete. Summary saved to {out_path}")
        if not summary_df.empty:
            print(summary_df[['t_evol', 'h_mag', 'total_capacity']].head())

if __name__ == "__main__":
    main()
