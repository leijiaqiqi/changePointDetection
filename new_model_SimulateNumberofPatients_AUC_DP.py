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


#X_i = patients
#n = sum(stages)


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

# Placeholder for Q(h, m) 
def Q(h, m, t):
    X_sorted = np.sort(X_i)  # integration grid from empirical support
    result = sum(abs(f(h, m, u) - f(m, t, u)) for u in X_sorted)
    return result/n

def dynamic_programming_segmentation(n, k):
    F = np.full((k + 1, n + 1), np.inf)  # F[r][m]
    H = np.full((k + 1, n + 1, 2), -1, dtype=int)  # H[r][m]

    # F(1, t) = Q(h, m,t)
    for t in range(2, n + 1):          # t is the third point
        for h in range(0, t - 1):
            for m in range(h + 1, t):
                cost = Q(h, m, t)
                if cost < F[1][t]:
                    F[1][t] = cost
                    H[1][t] = (h, m)

    # Fill in the DP table
    for r in range(2, k+1):
        for t in range(2, n + 1):
            for h in range(0, t - 1):
                for m in range(h + 1, t):
                    prev_cost = F[r - 1][h]
                    if prev_cost < np.inf:
                        cost = prev_cost + Q(h, m, t)
                        if cost < F[r][t]:
                            F[r][t] = cost
                            H[r][t] = (h, m)

    # Backtracking to find change-points
    tau = [0] * (k + 1)
    tau[k] = n
    for r in range(k-1, 0, -1):
        h, m = H[r+1][tau[r + 1]]
        tau[r-1] = h
        tau[r] = m

    # Remove leading zero and return change-points only
    change_points = tau[1:]
    max_log_likelihood = -0.5 * F[k][n] + k*np.log(n)
    
    return change_points, max_log_likelihood, F, H

K_n =3
change_points, loglike, F_table, H_table = dynamic_programming_segmentation(n, K_n)

print("Estimated change-points:", change_points)
print("Maximum log-likelihood:", loglike)