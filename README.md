# Optimization-Integrated Ratio-Test Refinement for Change-Point Detection

## Overview

This code implements an optimization-based change-point detection framework that combines **Pruned Exact Linear Time (PELT)** with a **mixed-integer linear programming (MILP)** refinement procedure. The purpose of PELT is therefore to provide a computationally efficient initial segmentation, while the MILP is used selectively to improve change-point locations or modify the number of change points.

The overall algorithm consists of two stages:

1. **PELT initialization**  
   PELT is first applied to the full data sequence to obtain an initial set of change points, $\Gamma^0=(\tau_1,\ldots,\tau_L)$.

2. **Optimization-integrated refinement**  
   The initial change-point set is refined using a ratio-test procedure that determines whether change points should be:
   - removed through a **merge step**, or
   - added through a **split step**.

   Whenever a merge is accepted, a local MILP is solved to re-optimize the locations of nearby change points.
   The final output is a refined change-point set $\Gamma$.
---

## Requirements

### Python

Required packages:

```text
gurobipy
numpy
ruptures
```

A working installation and license for **Gurobi** are also required.

### R

Required packages:

```text
gurobi
Matrix
changepoint
```

The `gurobi` R package is distributed with the Gurobi installation and requires a valid Gurobi license.


## Algorithm Description

### 1. Fixed-Number MILP

The function

```text
change_point_detection_fixed_num
```

solves the optimization model for a fixed number of segments.

An important convention is:

```text
Length = number of segments
```

Therefore,

```text
1 change point  -> Length = 2
m change points -> Length = m + 1
```

Given a sequence $X=(X_1,\ldots,X_n)$, the MILP assigns each observation to exactly one ordered segment using binary variables $z_{i,l}$.

The model imposes:

- exactly one segment assignment for each observation;
- a minimum segment length $\Delta$;
- ordered and contiguous segment assignments;
- a nonparametric clustering variance risk function used to evaluate the segmentation.

After solving the MILP, the cumulative segment lengths are converted into change-point locations.


### 2. Segment Cost

The ratio-test algorithm uses a risk function $R(\tau_i,\tau_{i+1})$. The default implementation uses nonparametric clustering variance, which represents within-segment variance.

```python
segment_cost_variance(X, tau_left, tau_right)
```

If a different risk function $R$ is used in the study, only the risk function needs to be replaced.


### 3. PELT Initialization

PELT is used to obtain an initial segmentation. Use the Python implementation as an example, in the `ruptures` package:

```python
rpt.Pelt(
    model="l2",
    min_size=Delta,
    jump=1
)
```

`min_size=Delta` ensures that the initial PELT segmentation respects the minimum segment length.

`jump=1` allows every possible observation index to be considered as a candidate change-point location.

The returned PELT breakpoint list contains the final endpoint \(n\). Therefore, the endpoint is removed before constructing $\Gamma^0$.

The R implementation uses the `changepoint` package:

```r
changepoint::cpt.mean(
    data = X,
    penalty = "Manual",
    pen.value = pen,
    method = "PELT",
    test.stat = "Normal",
    class = FALSE,
    minseglen = Delta
)
```

Again, the final endpoint $n$ is excluded when constructing the initial change-point set.


### 4. Local Change-Point Refinement

The local refinement procedure re-optimizes change-point locations within a selected interval $(\tau_{\mathrm{left}},\tau_{\mathrm{right}})$.

Suppose there are $m$ existing change points within this interval. The observations outside the interval remain unchanged, while the $m$ change points inside the interval are replaced by the optimal locations returned by the MILP.

Conceptually, the procedure performs $\Gamma\leftarrow\text{LocalRefine}(\Gamma,\tau_{\mathrm{left}},\tau_{\mathrm{right}})$.

This step is particularly important after removing a change point because neighboring change points may no longer be optimal once the segmentation structure changes.


### 5. Merge Test

For the current change-point set $\Gamma=\{\tau_1,\ldots,\tau_L\}$, sentinel endpoints are defined as $\tau_0=0,\tau_{L+1}=n$.

For each current change point $\tau_{i+1}$, the algorithm evaluates the effect of merging its two adjacent segments. The merge ratio is $\delta_{\mathrm{merge},i} =\frac{R(\tau_i,\tau_{i+2})-R(\tau_i,\tau_{i+1})-R(\tau_{i+1},\tau_{i+2})}{\sum_j R(\tau_j,\tau_{j+1})}$.

The algorithm selects $i^*=\arg\min_i\delta_{\mathrm{merge},i}$. Thus, the most negative value represents the largest relative reduction in the total segment cost.

A merge is accepted when $\delta_{\mathrm{merge}}(\Gamma)<-\eta$, where $\eta>0$ is a user-specified threshold. The selected change point is removed: $\tau_{i^*+1}\notin\Gamma$.

### 6. Local Refinement After a Merge

After removing a change point, nearby change-point locations are re-optimized. A neighborhood half-width $w$ determines the local interval, and the left and right boundaries are selected around the removed change point. The MILP is then solved only over this local interval.

The number of change points used in the local optimization is determined from the number of points that actually remain inside $(\tau_{\mathrm{left}},
\tau_{\mathrm{right}})$.

This is important near the beginning or end of the sequence, where the refinement window may contain fewer than $2w$ change points.

The merge procedure continues until $\delta_{\mathrm{merge}}(\Gamma)
\ge -\eta$ or only one change point remains.


### 7. Split Test

If the initial merge test does not satisfy the merge criterion, the algorithm enters the split phase. Each existing segment $(\tau_i,\tau_{i+1})$ is considered separately.

For each segment, the fixed-number MILP is solved with $L=2$ because one additional change point divides the segment into two segments.

Let the optimal candidate split location be $\tau^*$. The relative improvement from splitting is$\delta_{\mathrm{split},i}=\frac{R(\tau_i,\tau_{i+1})-R(\tau_i,\tau^*)-R(\tau^*,\tau_{i+1})}{\sum_j R(\tau_j,\tau_{j+1})}$.

The segment producing the largest improvement is selected: $i^*=\arg\max_i\delta_{\mathrm{split},i}$.

A new change point is accepted when $\delta_{\mathrm{split}}(\Gamma)>\eta$. The selected point \(\tau^*\) is added to \(\Gamma\), after which the split test is recomputed.

The procedure terminates when $\delta_{\mathrm{split}}(\Gamma)\le \eta$.

## Overall Algorithm

The complete algorithm is:

```text
Input:
    input data sequence 'X'
    minimum segment length 'Delta'
    ratio-test threshold 'eta'
    neighborhood half-width 'w'
    penalty parameter used by PELT `pelt_pen`

Step 1:
    Run PELT and obtain 'Gamma_0'.

Step 2:
    Set Gamma = Gamma_0.

Step 3:
    Compute the best merge ratio.

Step 4:
    If delta_merge < -eta:

        Repeatedly:
            remove the selected change point;
            construct a local neighborhood;
            solve the local MILP;
            update Gamma;
            recompute the best merge ratio.

    Otherwise:

        Repeatedly:
            solve a two-segment MILP for each current segment;
            identify the best candidate split;
            add the candidate if delta_split > eta;
            recompute the split test.

Step 5:
    Return the refined Gamma.
```


###  Main Functions

`change_point_detection_fixed_num`: Solves the MILP for a predetermined number of segments.

`pelt_initialization`: Runs PELT and constructs the initial change-point set $\Gamma^0$.

`local_change_point_refinement`: Re-optimizes all existing change points within a specified local interval.

`total_segment_cost`: Computes $\sum_i R(\tau_i,\tau_{i+1})$ for the current segmentation.

`compute_delta_merge`: Evaluates all possible single-change-point removals and returns $\tau^*$ and $\delta_{\mathrm{merge}}(\Gamma)$

`compute_delta_split`: Solves a two-segment MILP for every splittable segment and returns $i^*$ and $\delta_{\mathrm{split}}(\Gamma)$

`optimization_integrated_ratio_test_refinement`:  Implements the merge/split ratio-test refinement algorithm given an initial change-point set.

`pelt_milp_ratio_refinement`: Main wrapper returns both the initial and final change-point sets.

Python example: 
```python
Gamma_0, Gamma = pelt_milp_ratio_refinement(
    X=X,
    Delta=3,
    eta=0.10,
    w=2,
    pelt_pen=10,
    R=segment_cost_variance,
    pelt_model="l2",
    use_same_R_in_pelt=False,
    output_flag=0
)

print("Initial PELT change points:")
print(Gamma_0)

print("Final refined change points:")
print(Gamma)
```

R example:
```r
result <- pelt_milp_ratio_refinement(
  X = X,
  Delta = 3,
  eta = 0.10,
  w = 2,
  pelt_pen = 10,
  R = segment_cost_variance,
  pelt_model = "l2",
  output_flag = 0
)

Gamma_0 <- result$Gamma_0
Gamma <- result$Gamma

print("Initial PELT change points:")
print(Gamma_0)

print("Final refined change points:")
print(Gamma)
```

## Notes

1. `Delta` should be chosen consistently in both PELT and the MILP so that the initial and refined segmentations satisfy the same minimum-length requirement.

2. The PELT penalty controls the number of initial change points. Different penalty values may therefore lead to different initial values of $\Gamma^0$.

3. The default `segment_cost_variance` function is provided as a convenient implementation of \(R\). If the computational study uses a different definition of \(R\), that function can be replaced with the corresponding risk function.


## Installation

Clone the repository:

```bash
git clone https://github.com/leijiaqiqi/change-point-detection.git
cd change-point-detection
python -m pip install -e .
