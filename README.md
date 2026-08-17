# Optimization-Integrated Ratio-Test Refinement for Change-Point Detection

## Overview

This code implements an optimization-based change-point detection framework that combines **Pruned Exact Linear Time (PELT)** with a **mixed-integer linear programming (MILP)** refinement procedure.

The overall algorithm consists of two stages:

1. **PELT initialization**  
   PELT is first applied to the full data sequence to obtain an initial set of change points,
   \(\Gamma^0=\{\tau_1,\ldots,\tau_L\}.
   \)

2. **Optimization-integrated refinement**  
   The initial change-point set is refined using a ratio-test procedure that determines whether change points should be:
   - removed through a **merge step**, or
   - added through a **split step**.

   Whenever a merge is accepted, a local MILP is solved to re-optimize the locations of nearby change points.

The final output is a refined change-point set \(\Gamma\).
---

## Requirements

### Python

Required packages:

```text
gurobipy
numpy
ruptures
```

Install the non-Gurobi packages using:

```bash
pip install numpy ruptures
```

A working installation and license for **Gurobi** are also required.

The Python implementation uses:

```python
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import ruptures as rpt
from ruptures.base import BaseCost
```

### R

Required packages:

```text
gurobi
Matrix
changepoint
```

Install the CRAN packages using:

```r
install.packages("Matrix")
install.packages("changepoint")
```

The `gurobi` R package is distributed with the Gurobi installation and requires a valid Gurobi license.

The R implementation uses:

```r
library(gurobi)
library(Matrix)
library(changepoint)
```

---

# Algorithm Description

## 1. Fixed-Number MILP

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

Given a sequence

\[
X=(X_1,\ldots,X_n),
\]

the MILP assigns each observation to exactly one ordered segment using binary variables \(z_{i,l}\).

The model imposes:

- exactly one segment assignment for each observation;
- a minimum segment length \(\Delta\);
- ordered and contiguous segment assignments;
- the empirical-CDF-based objective used to evaluate the segmentation.

After solving the MILP, the cumulative segment lengths are converted into change-point locations.

For example,

```text
Z_1 = [20, 50, 100]
```

corresponds to two interior change points,

```text
[20, 50]
```

with `100` representing the final endpoint of the sequence.

---

## 2. Segment Cost

The ratio-test algorithm uses a segment cost

\[
R(\tau_i,\tau_{i+1}).
\]

The default implementation uses within-segment variance.

In Python:

```python
segment_cost_variance(X, tau_left, tau_right)
```

In R:

```r
segment_cost_variance(X, tau_left, tau_right)
```

The implemented variance is

\[
R(a,b)
=
\frac{1}{b-a}
\sum_{i=a+1}^{b}
(X_i-\bar X_{a:b})^2.
\]

This corresponds to the default behavior of `numpy.var()`.

If a different segment loss \(R\) is used in the study, only the segment-cost function needs to be replaced.

---

## 3. PELT Initialization

PELT is used to obtain an initial segmentation.

### Python

The Python implementation uses the `ruptures` package:

```python
rpt.Pelt(
    model="l2",
    min_size=Delta,
    jump=1
)
```

`min_size=Delta` ensures that the initial PELT segmentation respects the minimum segment length.

`jump=1` allows every possible observation index to be considered as a candidate change-point location.

The returned PELT breakpoint list contains the final endpoint \(n\). Therefore, the endpoint is removed before constructing

\[
\Gamma^0.
\]

For example,

```text
PELT output: [100, 250, 500]
```

is converted to

```text
Gamma_0 = [100, 250]
```

when \(n=500\).

### R

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

Again, the final endpoint \(n\) is excluded when constructing the initial change-point set.

---

# 4. Local Change-Point Refinement

The local refinement procedure re-optimizes change-point locations within a selected interval

\[
(\tau_{\mathrm{left}},\tau_{\mathrm{right}}).
\]

Suppose there are \(m\) existing change points within this interval.

The local MILP is solved with

\[
m+1
\]

segments.

The observations outside the interval remain unchanged, while the \(m\) change points inside the interval are replaced by the optimal locations returned by the MILP.

Conceptually, the procedure performs

\[
\Gamma
\leftarrow
\operatorname{LocalRefine}
(
\Gamma,
\tau_{\mathrm{left}},
\tau_{\mathrm{right}}
).
\]

This step is particularly important after removing a change point because neighboring change points may no longer be optimal once the segmentation structure changes.

---

# 5. Merge Test

For the current change-point set

\[
\Gamma
=
\{\tau_1,\ldots,\tau_L\},
\]

sentinel endpoints are defined as

\[
\tau_0=0,
\qquad
\tau_{L+1}=n.
\]

For each current change point \(\tau_{i+1}\), the algorithm evaluates the effect of merging its two adjacent segments.

The merge ratio is

\[
\delta_{\mathrm{merge},i}
=
\frac{
R(\tau_i,\tau_{i+2})
-
R(\tau_i,\tau_{i+1})
-
R(\tau_{i+1},\tau_{i+2})
}{
\sum_j R(\tau_j,\tau_{j+1})
}.
\]

The algorithm selects

\[
i^*
=
\arg\min_i
\delta_{\mathrm{merge},i}.
\]

Thus, the most negative value represents the largest relative reduction in the total segment cost.

A merge is accepted when

\[
\delta_{\mathrm{merge}}(\Gamma)<-\eta,
\]

where \(\eta>0\) is a user-specified threshold.

The selected change point is removed:

\[
\tau_{i^*+1}
\notin
\Gamma.
\]

---

# 6. Local Refinement After a Merge

After removing a change point, nearby change-point locations are re-optimized.

A neighborhood half-width \(w\) determines the local interval.

The left and right boundaries are selected around the removed change point, subject to the sentinel endpoints \(0\) and \(n\).

The MILP is then solved only over this local interval.

The number of change points used in the local optimization is determined from the number of points that actually remain inside

\[
(\tau_{\mathrm{left}},
\tau_{\mathrm{right}}).
\]

This is important near the beginning or end of the sequence, where the refinement window may contain fewer than \(2w\) change points.

The merge procedure continues until

\[
\delta_{\mathrm{merge}}(\Gamma)
\ge -\eta
\]

or only one change point remains.

---

# 7. Split Test

If the initial merge test does not satisfy the merge criterion, the algorithm enters the split phase.

Each existing segment

\[
(\tau_i,\tau_{i+1})
\]

is considered separately.

For each segment, the fixed-number MILP is solved with

```text
Length = 2
```

because one additional change point divides the segment into two segments.

Let the optimal candidate split location be \(\tau^*\).

The relative improvement from splitting is

\[
\delta_{\mathrm{split},i}
=
\frac{
R(\tau_i,\tau_{i+1})
-
R(\tau_i,\tau^*)
-
R(\tau^*,\tau_{i+1})
}{
\sum_j R(\tau_j,\tau_{j+1})
}.
\]

The segment producing the largest improvement is selected:

\[
i^*
=
\arg\max_i
\delta_{\mathrm{split},i}.
\]

A new change point is accepted when

\[
\delta_{\mathrm{split}}(\Gamma)>\eta.
\]

The selected point \(\tau^*\) is added to \(\Gamma\), after which the split test is recomputed.

The procedure terminates when

\[
\delta_{\mathrm{split}}(\Gamma)
\le \eta.
\]

---

# 8. Overall Optimization-Integrated Ratio-Test Algorithm

The complete algorithm is:

```text
Input:
    data X
    minimum segment length Delta
    PELT penalty
    ratio-test threshold eta
    neighborhood half-width w

Step 1:
    Run PELT and obtain Gamma_0.

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

The procedure executes either the merge phase or the split phase according to the initial merge test.

---

# Main Parameters

### `X`

One-dimensional input data sequence.

### `Delta`

Minimum allowed segment length.

Every segment must contain at least `Delta` observations.

### `eta`

Relative-improvement threshold used in the merge and split tests.

For example,

```text
eta = 0.10
```

corresponds to a 10% threshold.

### `w`

Neighborhood half-width used after a merge.

Larger values allow more nearby change points to be jointly re-optimized but increase MILP computation time.

### `pelt_pen`

Penalty parameter used by PELT.

The appropriate numerical value depends on the PELT cost function and data scale.

---

# Main Functions

## `change_point_detection_fixed_num`

Solves the MILP for a predetermined number of segments.

**Input**

```text
Length
X
Delta
output_flag
```

**Output**

Cumulative segment endpoints.

---

## `pelt_initialization`

Runs PELT and constructs the initial change-point set

\[
\Gamma^0.
\]

---

## `local_change_point_refinement`

Re-optimizes all existing change points within a specified local interval.

---

## `total_segment_cost`

Computes

\[
\sum_i R(\tau_i,\tau_{i+1})
\]

for the current segmentation.

---

## `compute_delta_merge`

Evaluates all possible single-change-point removals and returns:

```text
delta_merge
i_star
```

where `i_star` identifies the selected change point.

---

## `compute_delta_split`

Solves a two-segment MILP for every splittable segment and returns:

```text
delta_split
i_star
tau_star
```

where `tau_star` is the optimal candidate new change point.

---

## `optimization_integrated_ratio_test_refinement`

Implements the merge/split ratio-test refinement algorithm given an initial change-point set.

---

## `pelt_milp_ratio_refinement`

Main wrapper that performs

```text
PELT initialization
        +
MILP ratio-test refinement
```

and returns both the initial and final change-point sets.

---

# Example: Python

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

The output has the form

```text
Initial PELT change points:
[105, 248, 403]

Final refined change points:
[100, 251, 399]
```

---

# Example: R

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

---

# Indexing Convention

The algorithm uses change-point locations as **segment boundaries**.

For example,

```text
Gamma = [100, 250]
```

for a sequence of length 500 represents

```text
Segment 1: observations 1--100
Segment 2: observations 101--250
Segment 3: observations 251--500
```

The endpoint \(n\) is not stored in `Gamma`; it is treated as a sentinel endpoint internally.

Similarly, \(0\) is used internally as the left sentinel endpoint.

---

# Notes

1. The fixed-number optimization problem can be computationally expensive because the formulation contains \(O(n^2L)\) auxiliary variables and constraints.

2. The purpose of PELT is therefore to provide a computationally efficient initial segmentation, while the MILP is used selectively to improve change-point locations or modify the number of change points.

3. `Delta` should be chosen consistently in both PELT and the MILP so that the initial and refined segmentations satisfy the same minimum-length requirement.

4. The PELT penalty controls the number of initial change points. Different penalty values may therefore lead to different initial values of \(\Gamma^0\).

5. The numerical values of PELT penalties are not necessarily directly comparable across different PELT implementations or cost functions.

6. The default `segment_cost_variance` function is provided as a convenient implementation of \(R\). If the computational study uses a different definition of \(R\), that function should be replaced with the corresponding segment loss.

---

## Output

The algorithm returns two change-point sets:

\[
\Gamma^0
\]

the initial solution obtained from PELT, and

\[
\Gamma
\]

the final solution after optimization-integrated ratio-test refinement.

## Installation

Clone the repository:

```bash
git clone https://github.com/leijiaqiqi/change-point-detection.git
cd change-point-detection
python -m pip install -e .
