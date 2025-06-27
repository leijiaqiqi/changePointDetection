# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 15:22:28 2025

@author: Jiaqi
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 17:15:18 2025

@author: Jiaqi
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 01:16:30 2025

@author: Jiaqi
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

def simulate_patient_flow_staged(mu_rate,
                                  stage_durations,
                                  arrival_rate_sampler,
                                  N0=0,
                                  seed=None):
    """
    Simulate a birth–death process with piecewise-constant arrival rates.

    Parameters
    ----------
    mu_rate : float
        Discharge rate per patient (constant across all stages).
    stage_durations : list of float
        Length (in time units) of each stage.  The sum of these is T_max.
    arrival_rate_sampler : callable
        Zero-argument function returning the arrival rate λ for each stage.
    N0 : int, optional
        Initial number of patients.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    times : ndarray
        Event times (including t=0).
    patients : ndarray
        Number of patients immediately after each event.
    stage_boundaries : list of float
        The times at which stages switch.
    arrival_rates : list of float
        The sampled arrival rate λ for each stage.
    """
    # initialize
    rng = np.random.default_rng(seed)
    T_max = sum(stage_durations)
    # sample one arrival rate per stage
    arrival_rates = [arrival_rate_sampler() for _ in stage_durations]
    # compute stage boundaries
    boundaries = np.cumsum([0] + stage_durations)
    
    t = 0.0
    N = N0
    times = [t]
    patients = [N]
    
    # while we're before T_max, simulate one event at a time
    while t < T_max:
        # determine which stage we're in
        stage_idx = np.searchsorted(boundaries, t, side='right') - 1
        λ = arrival_rates[stage_idx]
        # rates for next event
        arrival_rate = λ
        discharge_rate = mu_rate * N
        total_rate = arrival_rate + discharge_rate
        if total_rate <= 0:
            break
        
        # time to next event
        dt = rng.exponential(1 / total_rate)
        t_next = t + dt
        # if that jumps past the next stage boundary, clamp to boundary
        next_boundary = boundaries[stage_idx + 1]
        if t_next > next_boundary:
            t = next_boundary
            # no change in N, but record the stage change
            times.append(t)
            patients.append(N)
            continue
        
        # otherwise accept the event
        t = t_next
        if rng.random() < arrival_rate / total_rate:
            N += 1
        else:
            N = max(0, N - 1)
        
        times.append(t)
        patients.append(N)
    
    return np.array(times), np.array(patients), boundaries, arrival_rates

# Example usage
mu = 0.3
stages = [10, 30, 10]           
sampler = lambda: np.random.uniform(1.0, 5.0)  # draw λ ~ Uniform(1,5) each stage

times, patients, bounds, rates = simulate_patient_flow_staged(
    mu_rate=mu,
    stage_durations=stages,
    arrival_rate_sampler=sampler,
    N0=2,
    seed=123
)



def estimate_transition_matrix(times, patients, delta_t, max_state=None):
    """
    Estimate P_ij(delta_t) from a continuous‐time trajectory.

    Parameters
    ----------
    times : array_like
        Event times from the Gillespie simulation.
    patients : array_like
        Patient counts at each event time.
    delta_t : float
        Time step for defining the discrete‐time chain.
    max_state : int, optional
        Maximum state to include in the matrix; if None, inferred from data.

    Returns
    -------
    P : 2D ndarray
        Transition matrix of size (S+1)x(S+1), where S = max_state.
    """
    T_max = times[-1]
    # Define grid times t_k = 0, delta_t, 2*delta_t, ..., <= T_max
    grid = np.arange(0, T_max + delta_t, delta_t)
    # Sample N at those times (right‐continuous)
    N_grid = np.zeros_like(grid, dtype=int)
    idx = 0
    for k, t_k in enumerate(grid):
        # Advance idx until times[idx] > t_k
        while idx+1 < len(times) and times[idx+1] <= t_k:
            idx += 1
        N_grid[k] = patients[idx]
    
    # Determine state space size
    S = max_state if max_state is not None else N_grid.max()
    
    # Initialize counts
    counts = np.zeros((S+1, S+1), dtype=int)
    row_totals = np.zeros(S+1, dtype=int)
    
    # Count transitions
    for k in range(len(N_grid)-1):
        i = N_grid[k]
        j = N_grid[k+1]
        if i <= S and j <= S:
            counts[i, j] += 1
            row_totals[i] += 1
    
    # Build empirical transition matrix
    P = np.zeros_like(counts, dtype=float)
    for i in range(S+1):
        if row_totals[i] > 0:
            P[i, :] = counts[i, :] / row_totals[i]
    
    return P

# Example usage:
# assume you have arrays `times` and `patients` from the previous simulation
delta_t = 1.0
P_hat = estimate_transition_matrix(times, patients, delta_t)

print("Empirical transition matrix P_hat(delta_t=1):")
print(P_hat)




# Plot
t_grid = np.linspace(0, sum(stages), 500)
plt.step(times, patients, where='post', label='N(t)')
for b in bounds:
    plt.axvline(b, color='gray', linestyle='--', linewidth=1)
#for idx, r in enumerate(rates):
#    plt.text((bounds[idx]+bounds[idx+1])/2, max(patients)*0.8,
#             f'λ={r:.2f}', ha='center', va='center', 
#             bbox=dict(facecolor='white', edgecolor='black'))
plt.xlabel('Time')
plt.ylabel('Number')
#plt.title('Staged Arrival Rates with μ={:.2f}'.format(mu))
plt.legend()
plt.tight_layout()
plt.show()


val1 = np.random.rand()
val2 = np.random.rand()
val3 = np.random.rand()

# Create the vector with specified segments
new_generated_vector = np.concatenate([
    np.full(10, val1),   # First 10 elements
    np.full(30, val2),   # Next 30 elements (11–40)
    np.full(10, val3)    # Last 10 elements
])


X_i = new_generated_vector
n = len(new_generated_vector)
def f(tau_i, tau_j, u):
    """Empirical CDF over a segment [tau_i, tau_j) evaluated at u."""
    segment = X_i[tau_i:tau_j]
    return np.sum(segment <= u)/(tau_j-tau_i)

def R_n(tau):
    """
    Compute R(tau) as defined:
    R = sum of integrated absolute differences of empirical CDFs
    over adjacent segments defined by tau.
    """
    X_sorted = np.sort(X_i)  # integration grid from empirical support

    R = 0
    for i in range(len(tau) - 2):
        diffs = sum(abs(f(tau[i], tau[i+1], u) - f(tau[i+1], tau[i+2], u)) for u in X_sorted)
        R += diffs/n  # ∫ |f - f| dF_n

    return R

def find_best_tau(L):
    positions = range(1, n)  # valid internal tau positions
    best_score = -np.inf
    best_tau = None

    for internal in combinations(positions, L):
        tau = [0] + list(internal) + [n]
        score = R_n(tau)
        if score > best_score:
            best_score = score
            best_tau = tau

    return best_tau, -1*best_score + len(tau)*np.log(n)

BIC = np.inf
location = None
for K_n in range(1,10):
    change_points, loglike = find_best_tau(K_n)
    print("Estimated change-points:", change_points)
    print("Maximum log-likelihood:", loglike)
    if loglike < BIC:
            BIC = loglike
            location = change_points

    

print("Estimated change-points:", location)
print("Minimum BIC:", BIC)