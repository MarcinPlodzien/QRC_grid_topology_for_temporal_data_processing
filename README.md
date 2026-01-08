# Quantum Reservoir Computing (QRC) on Multi-Rail Spin Systems: Protocol & Implementation

## 1. Introduction

This project implements a high-performance simulation framework for **Quantum Reservoir Computing (QRC)** using spin-1/2 systems arranged in a **Multi-Rail (Ladder) Topology**. The primary objective is to evaluate the memory and prediction capacity of quantum substrates processing temporal time-series information.

By leveraging **JAX** for GPU/CPU acceleration and automatic differentiation (vectorization), the framework allows for the efficient exploration of:
*   **Topological variations**: Tunable couplings between $N$ "rails" (chains) and input mechanisms.
*   **Input encoding strategies**: Single-qubit injection vs. Sliding-window (Batch) embedding.
*   **Disorder effects**: The role of Anderson localization in preserving information.

---

## 2. Physical System: The Multi-Rail Spin Ladder

The quantum reservoir is modeled as a system consisting of $N$ coupled 1D chains (rails) of length $L$. The total system size is $N_{qubits} = N \times L$.

### 2.1. Topology
The system is divided into functional sub-systems:
1.  **Input Rail (Rail 0)**: A 1D chain where external information is injected.
2.  **Reservoir Rails (Rails 1 to $N-1$)**: 1D chains that act as the primary processing substrate, coupled to the input rail and each other via "rungs".

The Hamiltonian is defined as:

$$
H = \sum_{r=0}^{N-1} H_{rail}^{(r)} + H_{rungs} + H_{field}
$$

### 2.2. Hamiltonian Interaction Terms
The couplings are generally of the Heisenberg or XXZ type:

**Intra-Rail Couplings**:

$$
H_{rail}^{(r)} = \sum_{i=1}^{L-1} \sum_{\alpha \in \{x,y,z\}} J_{rail}^{\alpha} \sigma_{i, r}^\alpha \sigma_{i+1, r}^\alpha
$$

**Rung (Inter-Rail) Couplings**:

$$
H_{rungs} = \sum_{r=0}^{N-2} \sum_{i=1}^{L} \sum_{\alpha \in \{x,y,z\}} J_{rung}^{\alpha} \sigma_{i, r}^\alpha \sigma_{i, r+1}^\alpha
$$

**External Fields**:

$$
H_{field} = \sum_{r=0}^{N-1} \sum_{i=1}^{L} \sum_{\alpha \in \{x,y,z\}} h_{i, r}^\alpha \sigma_{i, r}^\alpha
$$

Where:
*   $\sigma^\alpha$ are Pauli matrices.
*   $J_{rail}$ controls intra-chain dynamics (transport).
*   $J_{rung}$ controls inter-chain information transfer.
*   $h_{i,r}$ represents external magnetic fields (Uniform or Disordered).

---

## 3. Operational Protocol

The QRC protocol operates in discrete time steps $k = 1, \dots, T$. Each step involves four distinct phases: **Reset**, **Injection**, **Evolution**, and **Measurement**.

### 3.1. Discrete Time Cycle

For each time step $k$:

1.  **Input Reset & Preparation**:
    The state of the **Input Rail (Rail 0)** is traced out and replaced with a fresh product state encoding the input signal $u_k$. The Reservoir Rails are **untouched**, preserving their memory.

$$
\rho_{total}^{(k)} = \rho_{input}(u_k) \otimes \text{Tr}_{input} \left( \rho_{total}^{(k-1)} \right)
$$

2.  **Input Encoding ($\rho_{input}$)**:
    The input data $u_k$ is encoded into the input rail. Two strategies are implemented:
    *   **Single Data Input** (`single_data_input`): Scalar injection.
    *   **Batch (Sliding Window) Input** (`batch_data_input`): Spatial-Temporal embedding of history vector $\vec{u}_k$.

3.  **Unitary Evolution**:
    The coupled system evolves for a duration $t_{evol}$ under the full Hamiltonian $H$.

$$
\rho_{total}^{(k)'} = e^{-i H t_{evol}} \rho_{total}^{(k)} e^{i H t_{evol}}
$$

4.  **Measurement (Readout)**:
    Observable expectations are collected from the Reservoir Rails.

$$
x_{k, i}^\alpha = \text{Tr}(O_i^\alpha \rho_{total}^{(k)'})
$$

    **Supported Observables**:
    *   **Pauli Expectations**: $\langle Z_i \rangle, \langle X_i \rangle, \langle Y_i \rangle$ (Local Means, Total Means, Total Stds)
    *   **Correlations**: $\langle Z_i Z_{i+1} \rangle, \langle X_i X_{i+1} \rangle, \langle Y_i Y_{i+1} \rangle$

---

## 4. Usage

The project uses a pipeline of scripts to Run, Analyze, and Plot results.

### 1. Configuration & Execution
Edit `01_config_qrc_ladder.py` to set parameters (Topology, $N_{rails}$, $L$, $t_{evol}$ values, Observables).
Then run the parallel simulation runner:
```bash
python 00_runner_parallel_CPU.py
```
*   This generates raw data in `results/{CONFIG_NAME}/`.

### 2. Analysis
Process the raw data to calculate Memory Capacity ($C_{total}$) and other metrics.
```bash
python 02_analyze_qrc_ladder.py
```
*   This creates `analysis_summary_*.pkl` files in the results directory.

### 3. Visualization
Generate standard performance plots (Capacity vs Time, Heatmaps).
```bash
python 03_plot_qrc_ladder.py
```
*   Helper: `python 04_plot_feature_comparison.py` compares how different subsets of observables (features) affect performance.

---

## 5. Implementation Details

### 5.1. Scientific Stack
*   **Language**: Python 3.10+
*   **Core Logic**: `JAX` (Google's XLA framework) for vectorization (`vmap`) and compilation/optimization (`jit`).
*   **Data Handling**: `pandas` DataFrame for structured results, `pickle` for serialization.

### 5.2. Time Integration strategies
*   **Exact Diagonalization (`exact_eig`)**: Best for small $N \le 12$.
*   **Runge-Kutta 4 (`rk4_dense`, `rk4_sparse`)**: For larger or time-dependent systems.
*   **Trotterization (`trotter`)**: For large sparse systems.

---

## 6. Benchmark Task: Mackey-Glass Prediction

The standard benchmark is the prediction of the chaotic **Mackey-Glass** time series.
*   **Task**: Given inputs $u_0, \dots, u_t$, predict future value $u_{t+\tau}$.
*   **Metric**: Memory Capacity $C_{total} = \sum_{\tau} R^2(y_{pred}(\tau), y_{true}(\tau))$.

---

**Author**: Marcin Plodzien
**Date**: January 2026
