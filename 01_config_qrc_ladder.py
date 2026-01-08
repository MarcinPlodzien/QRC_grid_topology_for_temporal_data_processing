"""
QRC Ladder Configuration Script
===============================
Author: Marcin Plodzien

This module serves as the **Control Center** for the QRC Simulation.
It defines the experimental parameters, physics settings, and the grid search hypercube.

Mechanism:
----------
1.  **Definitions**: Constants define the search space (e.g., `J_rungs`, `T_EVOL_VALUES`).
2.  **Generator**: The `generate_configs()` function computes the Cartesian product of all parameters.
3.  **Consumption**: The Runner (`00_runner_parallel_CPU.py`) imports this module and iterates 
    over the list returned by `generate_configs()` to dispatch jobs.

Critical Parameters:
--------------------
- `CONFIG_NAME`: Defines the root directory for results (`results/{CONFIG_NAME}`).
- `INTEGRATION_METHOD`: Selects the solver ('trotter', 'exact_eig', 'rk4_dense', 'rk4_sparse').
- `OBSERVABLES`: List of strings defining what to measure (passed to `utils.engine`).

"""

import itertools
import numpy as np
import utils.hamiltonians as uh

# ==============================================================================
# 1. EXPERIMENT IDENTITY & ORCHESTRATION
# ==============================================================================
# This name determines the output folder: results/{CONFIG_NAME}/
CONFIG_NAME = "QRC_TFIM_ZZ_X_N_rails_3_L_2"

# Number of Parallel Workers for the Runner (ProcessPoolExecutor)
# Set to 1 for Serial Execution (Debugging), or >1 for Parallel.
MAX_WORKERS = 10

# ==============================================================================
# 2. PHYSICS CORE SETTINGS
# ==============================================================================

# Time Integration Strategy
# Options:
# - 'trotter':    Suzuki-Trotter decomposition (1st Order). Fast, memory efficient. 
#                 Best for large N. Error ~ O(dt).
# - 'exact_eig':  Exact Diagonalization. Computes full eigendecomp. 
#                 Best for small N (<= 12). Precision benchmark.
# - 'rk4_dense':  Runge-Kutta 4 (Dense Matrix). Good for N <= 12 with time-dependent H.
# - 'rk4_sparse': Runge-Kutta 4 (Sparse BCOO). Good for N > 12 intermediate scale.
INTEGRATION_METHOD = 'exact_eig'

# Disorder Scaling
# J_LADDER_MID/SCALE are multipliers for J values (if needed). Currently 1.0.
J_LADDER_MID = 1.0   
J_LADDER_SCALE = 1.0 

# Observables to Measure
# Defines the list of operators calculated at each time step.
# - *_local_mean: Vector of values per site.
# - *_total_mean: Scalar average.
# - *_total_std: Scalar standard deviation (fluctuation).
# - 'Norm': Trace(rho) to check conservation of probability.
OBSERVABLES = [
    'Z_local_mean', 'X_local_mean', 'Y_local_mean',
    'Z_local_std',  'X_local_std',  'Y_local_std',
    'ZZ_local_mean', 'XX_local_mean', 'YY_local_mean'
]
 
# ==============================================================================
# 3. HYPERPARAMETER GRID SEARCH
# ==============================================================================


N_RAILS = [3] # New Topology Parameter
L_RAIL_LENGTH = [2] # Length of the rail

# A. COUPLINGS (J)
# Format: List of dictionaries defining coupling sets to sweep.
# 'couplings': (Jx, Jy, Jz) tuple.

# 1. Rung Couplings (Inter-rail)
# For Multi-Rail, we can define a list of couplings per rung or a single uniform one.
# Here we stick to simple uniform for now, but the structure supports [J_r01, J_r12...]
J_rungs = [
     {"name": "ZZ",          "couplings": [(0.0, 0.0, 1.0)]}, # Will be broadcasted if needed
]

# 2. Rail Couplings (Intra-rail)
# For Multi-Rail, we define a list of couplings for each rail 0..N-1
J_rails = [
    {"name": "ZZ_All", "couplings": [(0.0, 0.0, 1.0)]}, # Broadcast to all rails
    {"name": "None_All", "couplings": [(0.0, 0.0, 0.0)]},
]

# B. EXTERNAL FIELDS (Disorder Directions)
h_field_rails = [
    {"name": "X", "couplings": [(1.0, 0.0, 0.0)]},
]

 

# D. TIME EVOLUTION
# Evolution time per input injection step.
T_EVOL_VALUES = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

# Number of integration sub-steps (dt = t_evol / steps).
N_STEPS_PER_T_EVOL = 100

 # Cartesian Product of All Parameters
FIELD_DISORDER = [False] # Toggle Random vs Uniform Fields
    

# E. REALIZATIONS
# Number of random seeds to run per configuration.
N_REALIZATIONS_UNIFORM = 1     # Uniform fields don't need averaging
N_REALIZATIONS_DISORDER = 20   # Random fields need meaningful statistics
REALIZATION_SHIFT = 0 # Offset for seed (useful for resuming/extending batches)
SEED_BASE = 42

# F. TASK SETTINGS
INPUT_STATES = ['product'] # Input encoding: Product state vs GHZ-like
DATA_INPUT_TYPES = ['batch_data_input'] # Placeholder for future data modes
PREPARE_MEASUREMENT = True

# Target Dataset path (Santa Fe, etc.)
TARGET_DATASET = './datasets/santafe.txt'

# Output Directory
RESULTS_DIR = f"results/{CONFIG_NAME}"

# ==============================================================================
# 4. CONFIGURATION GENERATOR
# ==============================================================================
def generate_configs():
    """
    Generates the list of all configuration dictionaries to be run.
    """
    configs = []
    
    # Load Templates
    library = {h['config_name']: h for h in uh.get_hamiltonian_library()}
    # Switch to MultiRail Template
    multirail_template = library.get('Top_MultiRail', None)
    
    if not multirail_template:
        raise ValueError("Could not find Top_MultiRail template in utils.hamiltonians.")
    
    # Grid
    experiment_grid = itertools.product(
        T_EVOL_VALUES, L_RAIL_LENGTH, N_RAILS, INPUT_STATES, DATA_INPUT_TYPES,
        J_rungs, J_rails, h_field_rails, FIELD_DISORDER
    )
    
    for (t_evol, L, n_rails, state, dmode, j_rung_conf, j_rail_conf, h_conf, is_disordered) in experiment_grid:
        
        N_total = L * n_rails
        
        # Expander logic: if config has 1 element list, broadcast to n_rails
        
        # Rungs (n_rails - 1)
        jr_val = j_rung_conf['couplings']
        if len(jr_val) == 1: jr_val = jr_val * (n_rails - 1)
        
        # Rails (n_rails)
        jrail_val = j_rail_conf['couplings']
        if len(jrail_val) == 1: jrail_val = jrail_val * n_rails
        
        # Fields (n_rails)
        h_val = h_conf['couplings']
        if len(h_val) == 1: h_val = h_val * n_rails
        
        # Names
        jr_name = j_rung_conf['name']
        jrail_name = j_rail_conf['name']
        h_name = h_conf['name']
        
        # Build Hamiltonian Configuration
        ham_config = multirail_template.copy()
        ham_config.update({
            'config_name': f"MRail_Nr{n_rails}_Jr{jrail_name}_JRung{jr_name}_F{h_name}",
            'n_rails': n_rails,
            'J_rails': jrail_val,
            'J_rungs': jr_val,
            'field_rails': h_val
        })
        
        # Unique Run Name
        dis_str = "Dis" if is_disordered else "Uni" 
        config_name = f"MRail_Nr{n_rails}_L{L}_T{t_evol}_Jr{jrail_name}_JRung{jr_name}_F{h_name}_{state}_{dmode}_{dis_str}"
        
        n_realizations = N_REALIZATIONS_DISORDER if is_disordered else N_REALIZATIONS_UNIFORM

        configs.append({
            'name': config_name,
            'config_name_batch': CONFIG_NAME,
            'ham_config': ham_config,
            't_evol': float(t_evol), 
            'dt': float(t_evol) / N_STEPS_PER_T_EVOL,
            'h_mag': 1.0,
            'field_disorder': is_disordered,
            'N': N_total, 
            'L': L,
            'n_rails': n_rails, # Explicitly pass n_rails
            'J_ladder_mid': J_LADDER_MID, 'J_ladder_scale': J_LADDER_SCALE,
            'input_state_type': state, 'data_input_type': dmode,
            'realization_start': REALIZATION_SHIFT, 'n_realizations': n_realizations, 'seed_base': SEED_BASE,
            'target_dataset': TARGET_DATASET,
            'integration_method': INTEGRATION_METHOD,
            'observables': OBSERVABLES,
            'prepare_measurement': PREPARE_MEASUREMENT,
            'output_dir': f"{RESULTS_DIR}/{config_name}/data",
            'param_names': {
                'topology': 'MultiRail',
                'J_rungs': jr_name,
                'J_rails': jrail_name,
                'field_rails': h_name,
                'field_disorder': str(is_disordered)
            }
        })
        
    return configs
