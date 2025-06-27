# -*- coding: utf-8 -*-
"""
Optimized simulation and change-point detection with progress bar
@author: Jiaqi
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm

def simulate_patient_flow_staged(mu_rate,
                                  stage_durations,
                                  arrival_rate_sampler,
                                  N0=0,
                                  seed=None):
    rng = np.random.default_rng(seed)
    T_max = sum(stage_durations)
    arrival_rates = [arrival_rate_sampler() for _ in stage_durations]
    boundaries = np.cumsum([0] + stage_durations)

    t = 0.0
    N = N0
    times = [t]
    patients = [N]

    while t < T_max:
        stage_idx = np.searchsorted(boundaries, t, side='right') - 1
        λ = arrival_rates[stage_idx]
        arrival_rate = λ
        discharge_rate = mu_rate * N
        total_rate = arrival_rate + discharge_rate
        if total_rate <= 0:
            break

        dt = rng.exponential(1 / total_rate)
        t_next = t + dt
        next_boundary = boundaries[stage_idx + 1]
        if t_next > next_boundary:
            t = next_boundary
            times.append(t)
            patients.append(N)
            continue

        t = t_next
        if rng.random() < arrival_rate / total_rate:
            N += 1
        else:
            N = max(0, N - 1)

        times.append(t)
        patients.append(N)

    return np.array(times), np.array(patients), boundaries, arrival_rates

def estimate_transition_matrix(times, patients, delta_t, max_state=None):
    T_max = times[-1]
    grid = np.arange(0, T_max + delta_t, delta_t)
    N_grid = np.zeros_like(grid, dtype=int)
    idx = 0
    for k, t_k in enumerate(grid):
        while idx+1 < len(times) and times[idx+1] <= t_k:
            idx += 1
        N_grid[k] = patients[idx]

    S = max_state if max_state is not None else N_grid.max()
    counts = np.zeros((S+1, S+1), dtype=int)
    row_totals = np.zeros(S+1, dtype=int)

    for k in range(len(N_grid)-1):
        i = N_grid[k]
        j = N_grid[k+1]
        if i <= S and j <= S:
            counts[i, j] += 1
            row_totals[i] += 1

    P = np.zeros_like(counts, dtype=float)
    for i in range(S+1):
        if row_totals[i] > 0:
            P[i, :] = counts[i, :] / row_totals[i]

    return P

# Run simulation
mu = 0.3
stages = [10, 30, 10]
sampler = lambda: np.random.uniform(1.0, 5.0)

times, patients, bounds, rates = simulate_patient_flow_staged(
    mu_rate=mu,
    stage_durations=stages,
    arrival_rate_sampler=sampler,
    N0=2,
    seed=123
)

delta_t = 1.0
P_hat = estimate_transition_matrix(times, patients, delta_t)
print("Empirical transition matrix P_hat(delta_t=1):")
print(P_hat)

# Plot patient trajectory
t_grid = np.linspace(0, sum(stages), 500)
plt.step(times, patients, where='post', label='N(t)')
for b in bounds:
    plt.axvline(b, color='gray', linestyle='--', linewidth=1)
plt.xlabel('Time')
plt.ylabel('Number')
plt.legend()
plt.tight_layout()
plt.show()

# Generate segmented vector
val1 = np.random.rand()
val2 = np.random.rand()
val3 = np.random.rand()
new_generated_vector = np.concatenate([
    np.full(10, val1),
    np.full(30, val2),
    np.full(10, val3)
])

X_i = new_generated_vector
n = len(X_i)
X_sorted = np.sort(X_i)

# Vectorized empirical CDF
def f_vec(tau_i, tau_j, u_vec):
    segment = X_i[tau_i:tau_j]
    segment_sorted = np.sort(segment)
    return np.searchsorted(segment_sorted, u_vec, side='right') / (tau_j - tau_i)

def R_n_fast(tau, X_sorted):
    R = 0
    for i in range(len(tau) - 2):
        f1 = f_vec(tau[i], tau[i+1], X_sorted)
        f2 = f_vec(tau[i+1], tau[i+2], X_sorted)
        R += np.sum(np.abs(f1 - f2)) / n
    return R

def find_best_tau_fast(L, X_sorted):
    positions = range(1, n)
    best_score = -np.inf
    best_tau = None
    for internal in combinations(positions, L):
        tau = [0] + list(internal) + [n]
        score = R_n_fast(tau, X_sorted)
        if score > best_score:
            best_score = score
            best_tau = tau
    return best_tau, -1 * best_score + len(tau) * np.log(n)

# BIC minimization loop with progress bar
BIC = np.inf
location = None
print("Searching for optimal number of change-points using BIC...")
for K_n in tqdm(range(1, 10), desc="Change-point search"):
    change_points, loglike = find_best_tau_fast(K_n, X_sorted)
    # Optional: comment next 2 lines for a cleaner progress bar
    print("  Estimated change-points:", change_points)
    print("  Maximum log-likelihood:", loglike)
    if loglike < BIC:
        BIC = loglike
        location = change_points

print("\nFinal Estimated change-points:", location)
print("Minimum BIC:", BIC)
