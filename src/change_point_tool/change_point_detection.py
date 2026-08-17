import gurobipy as gp
from gurobipy import GRB
import numpy as np


# ============================================================
# Fixed-number MILP
#
# IMPORTANT:
# Length = number of SEGMENTS in this function.
# Therefore:
#   1 change point  -> Length = 2
#   m change points -> Length = m + 1
# ============================================================

def change_point_detection_fixed_num(
        Length, X, Delta, output_flag=0):

    L = Length
    X = np.asarray(X)
    u_vals = np.sort(np.array(X))
    n = len(X)
    delta = Delta

    if X.ndim != 1:
        raise ValueError("X must be one-dimensional.")

    if L < 1:
        raise ValueError("Length must be at least 1.")

    if n < L * delta:
        raise ValueError(
            f"Infeasible: n={n} < Length*Delta={L * delta}."
        )

    model = gp.Model("Segmented_CDF_Diff")

    # Variables
    z = model.addVars(
        n, L,
        vtype=GRB.BINARY,
        name="z"
    )

    s = model.addVars(
        n, L,
        vtype=GRB.CONTINUOUS,
        name="s"
    )

    cdf_l = model.addVars(
        n, L,
        lb=0,
        ub=1,
        vtype=GRB.CONTINUOUS,
        name="cdf_l"
    )

    t = model.addVars(
        n, L,
        vtype=GRB.CONTINUOUS,
        name="t"
    )

    k = model.addVars(
        L,
        lb=0,
        ub=1 / delta,
        vtype=GRB.CONTINUOUS,
        name="k"
    )

    b = model.addVars(
        n, L, n,
        lb=0,
        ub=1,
        vtype=GRB.CONTINUOUS,
        name="b"
    )

    diff = model.addVars(
        n, L,
        lb=0,
        ub=1,
        vtype=GRB.CONTINUOUS,
        name="diff"
    )

    # --------------------------------------------------------
    # Each observation must be assigned to exactly one segment
    # --------------------------------------------------------

    for i in range(n):
        model.addConstr(
            sum(z[i, l] for l in range(L)) == 1,
            name=f"segment_{i}"
        )

    # --------------------------------------------------------
    # Minimum segment length
    # --------------------------------------------------------

    for l in range(L):
        model.addConstr(
            sum(z[i, l] for i in range(n)) >= delta,
            name=f"segment_l_{l}"
        )

    # --------------------------------------------------------
    # Monotonic segment assignment
    # --------------------------------------------------------

    for i in range(n - 1):
        for l in range(L):
            model.addConstr(
                z[i, l]
                <= sum(
                    z[i + 1, lp]
                    for lp in range(l, L)
                ),
                name=f"monotone_{i}_{l}"
            )

    # NOTE:
    # The following constraints from your original code
    # are intentionally removed:
    #
    # for i in range(n):
    #     for j in range(i,n):
    #         model.addConstr(z[i,0] >= z[j,0])
    #         model.addConstr(z[i,1] <= z[j,1])
    #
    # They are not valid for a general number of segments.

    # --------------------------------------------------------
    # Indicators
    # --------------------------------------------------------

    indicators = [
        [
            int(X[i] <= u)
            for i in range(n)
        ]
        for u in u_vals
    ]

    # --------------------------------------------------------
    # CDF constraints
    # --------------------------------------------------------

    for l in range(L):

        model.addConstr(
            gp.quicksum(
                t[i, l]
                for i in range(n)
            ) == 1
        )

        for u in range(n):

            model.addConstr(
                cdf_l[u, l]
                ==
                sum(
                    indicators[u][i] * t[i, l]
                    for i in range(n)
                )
            )

            model.addConstr(
                t[u, l] <= z[u, l]
            )

            model.addConstr(
                t[u, l]
                <=
                k[l]
                + 1 / n * z[u, l]
                - 1 / n
            )

            model.addConstr(
                t[u, l]
                >=
                1 / n * z[u, l]
            )

            model.addConstr(
                t[u, l]
                >=
                k[l] + z[u, l] - 1
            )

    # --------------------------------------------------------
    # Linearization
    # --------------------------------------------------------

    for l in range(L):
        for u in range(n):
            for i in range(n):

                model.addConstr(
                    b[i, l, u]
                    <= diff[u, l]
                )

                model.addConstr(
                    b[i, l, u]
                    <= z[i, l]
                )

                model.addConstr(
                    b[i, l, u]
                    >=
                    z[i, l]
                    + diff[u, l]
                    - 1
                )

    # --------------------------------------------------------
    # s and diff
    # --------------------------------------------------------

    for l in range(L):
        for u in range(n):

            model.addConstr(
                diff[u, l]
                + cdf_l[u, l]
                == 1
            )

            model.addConstr(
                s[u, l]
                ==
                sum(
                    indicators[u][i]
                    * b[i, l, u]
                    for i in range(n)
                )
            )

    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------

    objective = sum(
        s[u, l]
        for u in range(n)
        for l in range(L)
    )

    model.setObjective(
        objective,
        GRB.MINIMIZE
    )

    model.Params.OutputFlag = output_flag

    model.optimize()

    # --------------------------------------------------------
    # Solution
    # --------------------------------------------------------

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(
            "Gurobi did not return an optimal solution. "
            f"Status = {model.Status}"
        )

    z_sol = np.zeros((n, L))

    for i in range(n):
        for l in range(L):
            z_sol[i, l] = z[i, l].X

    col_sums = np.rint(
        np.sum(z_sol, axis=0)
    ).astype(int)

    Z_1 = np.cumsum(col_sums)

    # Example:
    #
    # Z_1 = [20, 50, 100]
    #
    # means:
    # change points = [20, 50]
    # final endpoint = 100

    return Z_1


# ============================================================
# Segment cost R
# ============================================================
#
# This is only a runnable default example.
#
# If the R(tau_i, tau_{i+1}) in your paper is different,
# replace THIS function with your actual R.
#
# The rest of the algorithm does not need to change.
# ============================================================

def segment_cost_variance(
        X, tau_left, tau_right):

    segment = np.asarray(X)[
        tau_left:tau_right
    ]

    if len(segment) == 0:
        return 0.0

    return float(
        np.var(segment)
    )


# ============================================================
# PELT initialization
# ============================================================

def pelt_initialization(
        X, Delta, pen,
        R=None,
        model="l2"):

    try:
        import ruptures as rpt
        from ruptures.base import BaseCost

    except ImportError as exc:

        raise ImportError(
            "The package 'ruptures' is required. "
            "Install it using:\n"
            "python -m pip install ruptures"
        ) from exc

    X = np.asarray(X)

    # --------------------------------------------------------
    # Option 1:
    # Standard PELT cost from ruptures
    # --------------------------------------------------------

    if R is None:

        algo = rpt.Pelt(
            model=model,
            min_size=Delta,
            jump=1
        ).fit(X)

    # --------------------------------------------------------
    # Option 2:
    # Use your own R as the PELT segment cost
    # --------------------------------------------------------

    else:

        class RCost(BaseCost):

            model = "custom_R"
            min_size = 1

            def __init__(
                    self, R, Delta):

                self.R = R
                self.min_size = Delta

            def fit(
                    self, signal):

                self.signal = np.asarray(signal)

                return self

            def error(
                    self, start, end):

                return float(
                    self.R(
                        self.signal,
                        start,
                        end
                    )
                )

        algo = rpt.Pelt(
            custom_cost=RCost(
                R,
                Delta
            ),
            min_size=Delta,
            jump=1
        ).fit(X)

    breakpoints = algo.predict(
        pen=pen
    )

    # ruptures includes n as the final breakpoint.
    #
    # For example:
    #
    # breakpoints = [100, 250, 500]
    #
    # Gamma_0 should be:
    #
    # [100, 250]
    #
    # because 500 = n is the endpoint.

    Gamma_0 = [
        int(tau)
        for tau in breakpoints
        if tau < len(X)
    ]

    return Gamma_0


# ============================================================
# Algorithm: Local Change-point Refinement
# ============================================================

def local_change_point_refinement(
        Gamma,
        X,
        tau_left,
        tau_right,
        Delta,
        output_flag=0):

    Gamma = sorted([
        int(tau)
        for tau in Gamma
    ])

    tau_left = int(tau_left)
    tau_right = int(tau_right)

    # --------------------------------------------------------
    # Number m of existing change points inside the interval
    # --------------------------------------------------------

    interior_points = [
        tau
        for tau in Gamma
        if tau_left < tau < tau_right
    ]

    m = len(interior_points)

    if m == 0:
        return Gamma

    # --------------------------------------------------------
    # Local data
    # --------------------------------------------------------

    X_local = np.asarray(X)[
        tau_left:tau_right
    ]

    # IMPORTANT:
    #
    # The pseudocode says m change points.
    #
    # Your MILP uses Length = number of segments.
    #
    # Therefore:
    #
    # m change points -> m+1 segments.

    Length = m + 1

    if len(X_local) < Length * Delta:

        raise ValueError(
            "Local interval is too short for "
            f"{Length} segments with Delta={Delta}."
        )

    # --------------------------------------------------------
    # Solve the local MILP
    # --------------------------------------------------------

    Z_local = (
        change_point_detection_fixed_num(
            Length=Length,
            X=X_local,
            Delta=Delta,
            output_flag=output_flag
        )
    )

    # Z_local includes the final endpoint.
    #
    # Therefore Z_local[:-1] gives the m
    # interior change points.

    refined_points = [
        tau_left + int(tau)
        for tau in Z_local[:-1]
    ]

    # --------------------------------------------------------
    # Keep points outside the refinement interval
    # --------------------------------------------------------

    Gamma_outside = [
        tau
        for tau in Gamma
        if not (
            tau_left < tau < tau_right
        )
    ]

    # --------------------------------------------------------
    # Replace old local points by refined local points
    # --------------------------------------------------------

    Gamma = sorted(
        Gamma_outside
        + refined_points
    )

    return Gamma


# ============================================================
# Total loss for current Gamma
# ============================================================

def total_segment_cost(
        Gamma, X, R):

    Gamma = sorted([
        int(tau)
        for tau in Gamma
    ])

    boundaries = (
        [0]
        + Gamma
        + [len(X)]
    )

    total = 0.0

    for i in range(
            len(boundaries) - 1):

        total += R(
            X,
            boundaries[i],
            boundaries[i + 1]
        )

    return float(total)


# ============================================================
# Compute delta_merge(Gamma)
# ============================================================

def compute_delta_merge(
        Gamma, X, R):

    Gamma = sorted([
        int(tau)
        for tau in Gamma
    ])

    L = len(Gamma)

    if L == 0:
        return np.inf, None

    boundaries = (
        [0]
        + Gamma
        + [len(X)]
    )

    denominator = (
        total_segment_cost(
            Gamma,
            X,
            R
        )
    )

    if abs(denominator) <= 1e-12:
        return np.inf, None

    delta_merge = np.inf
    i_star = None

    # --------------------------------------------------------
    # Examine each current change point
    # --------------------------------------------------------

    for i in range(L):

        # Cost before removing tau_{i+1}

        old_cost = (
            R(
                X,
                boundaries[i],
                boundaries[i + 1]
            )
            +
            R(
                X,
                boundaries[i + 1],
                boundaries[i + 2]
            )
        )

        # Cost after removing tau_{i+1}

        merged_cost = R(
            X,
            boundaries[i],
            boundaries[i + 2]
        )

        delta_i = (
            merged_cost
            - old_cost
        ) / denominator

        # Most negative delta = largest reduction
        # from performing a merge.

        if delta_i < delta_merge:

            delta_merge = delta_i
            i_star = i

    return (
        float(delta_merge),
        i_star
    )


# ============================================================
# Compute delta_split(Gamma)
# ============================================================

def compute_delta_split(
        Gamma,
        X,
        Delta,
        R,
        output_flag=0):

    Gamma = sorted([
        int(tau)
        for tau in Gamma
    ])

    boundaries = (
        [0]
        + Gamma
        + [len(X)]
    )

    denominator = (
        total_segment_cost(
            Gamma,
            X,
            R
        )
    )

    if abs(denominator) <= 1e-12:

        return (
            -np.inf,
            None,
            None
        )

    delta_split = -np.inf

    i_star = None
    tau_star = None

    # --------------------------------------------------------
    # Test every current segment
    # --------------------------------------------------------

    for i in range(
            len(boundaries) - 1):

        tau_left = boundaries[i]
        tau_right = boundaries[i + 1]

        # We need two segments, each having
        # at least Delta observations.

        if (
            tau_right - tau_left
            < 2 * Delta
        ):
            continue

        X_local = np.asarray(X)[
            tau_left:tau_right
        ]

        # ----------------------------------------------------
        # Solve the MILP with exactly ONE change point.
        #
        # One change point = two segments,
        # so Length = 2 in your Python MILP.
        # ----------------------------------------------------

        Z_local = (
            change_point_detection_fixed_num(
                Length=2,
                X=X_local,
                Delta=Delta,
                output_flag=output_flag
            )
        )

        # First cumulative segment length
        # gives the candidate split point.

        candidate_tau = (
            tau_left
            + int(Z_local[0])
        )

        # ----------------------------------------------------
        # Loss before splitting
        # ----------------------------------------------------

        old_cost = R(
            X,
            tau_left,
            tau_right
        )

        # ----------------------------------------------------
        # Loss after splitting
        # ----------------------------------------------------

        new_cost = (
            R(
                X,
                tau_left,
                candidate_tau
            )
            +
            R(
                X,
                candidate_tau,
                tau_right
            )
        )

        delta_i = (
            old_cost
            - new_cost
        ) / denominator

        # Largest delta = largest reduction
        # from adding a change point.

        if delta_i > delta_split:

            delta_split = delta_i
            i_star = i
            tau_star = candidate_tau

    return (
        float(delta_split),
        i_star,
        tau_star
    )


# ============================================================
# Algorithm:
# Optimization-Integrated Ratio-Test Refinement
# ============================================================

def optimization_integrated_ratio_test_refinement(
        X,
        Gamma_0,
        Delta,
        eta,
        w,
        R=segment_cost_variance,
        output_flag=0):

    X = np.asarray(X)

    n = len(X)

    # --------------------------------------------------------
    # Initial Gamma
    # --------------------------------------------------------

    Gamma = sorted([
        int(tau)
        for tau in Gamma_0
        if 0 < int(tau) < n
    ])

    if w < 1:

        raise ValueError(
            "w must be at least 1."
        )

    if eta < 0:

        raise ValueError(
            "eta must be nonnegative."
        )

    # ========================================================
    # Step 1:
    # Compute delta_merge(Gamma) and i*
    # ========================================================

    delta_merge, i_star = (
        compute_delta_merge(
            Gamma=Gamma,
            X=X,
            R=R
        )
    )

    # ========================================================
    # MERGE PHASE
    # ========================================================

    if (
        len(Gamma) > 1
        and i_star is not None
        and delta_merge < -eta
    ):

        while (
            len(Gamma) > 1
            and i_star is not None
            and delta_merge < -eta
        ):

            # ------------------------------------------------
            # Remove tau_{i*+1}
            #
            # Gamma is 0-indexed, therefore Gamma[i_star]
            # corresponds to tau_{i*+1}.
            # ------------------------------------------------

            Gamma.pop(i_star)

            # ------------------------------------------------
            # Reindex Gamma
            # ------------------------------------------------

            boundaries = (
                [0]
                + Gamma
                + [n]
            )

            # ------------------------------------------------
            # Construct local refinement window
            #
            # tau_left =
            # tau_{max(0, i*-w)}
            #
            # tau_right =
            # tau_{min(L+1, i*+w+1)}
            # ------------------------------------------------

            left_index = max(
                0,
                i_star - w
            )

            right_index = min(
                len(boundaries) - 1,
                i_star + w + 1
            )

            tau_left = (
                boundaries[left_index]
            )

            tau_right = (
                boundaries[right_index]
            )

            # ------------------------------------------------
            # Local refinement
            #
            # m is determined automatically from the
            # number of Gamma points inside the interval.
            # ------------------------------------------------

            Gamma = (
                local_change_point_refinement(
                    Gamma=Gamma,
                    X=X,
                    tau_left=tau_left,
                    tau_right=tau_right,
                    Delta=Delta,
                    output_flag=output_flag
                )
            )

            # ------------------------------------------------
            # Recompute merge ratio
            # ------------------------------------------------

            delta_merge, i_star = (
                compute_delta_merge(
                    Gamma=Gamma,
                    X=X,
                    R=R
                )
            )

    # ========================================================
    # SPLIT PHASE
    # ========================================================

    else:

        # ----------------------------------------------------
        # Find the best segment and candidate split
        # ----------------------------------------------------

        (
            delta_split,
            i_star,
            tau_star
        ) = compute_delta_split(
            Gamma=Gamma,
            X=X,
            Delta=Delta,
            R=R,
            output_flag=output_flag
        )

        # ----------------------------------------------------
        # Continue splitting while improvement > eta
        # ----------------------------------------------------

        while (
            tau_star is not None
            and delta_split > eta
        ):

            # Add tau*
            Gamma.append(
                int(tau_star)
            )

            # Reindex
            Gamma = sorted(Gamma)

            # Recompute best split
            (
                delta_split,
                i_star,
                tau_star
            ) = compute_delta_split(
                Gamma=Gamma,
                X=X,
                Delta=Delta,
                R=R,
                output_flag=output_flag
            )

    return Gamma


# ============================================================
# Complete wrapper:
#
# PELT
#   ->
# Gamma^0
#   ->
# Optimization-Integrated Ratio-Test Refinement
# ============================================================

def pelt_milp_ratio_refinement(
        X,
        Delta,
        eta,
        w,
        pelt_pen,
        R=segment_cost_variance,
        pelt_model="l2",
        use_same_R_in_pelt=False,
        output_flag=0):

    # --------------------------------------------------------
    # Step 1: PELT
    # --------------------------------------------------------

    if use_same_R_in_pelt:

        Gamma_0 = pelt_initialization(
            X=X,
            Delta=Delta,
            pen=pelt_pen,
            R=R
        )

    else:

        Gamma_0 = pelt_initialization(
            X=X,
            Delta=Delta,
            pen=pelt_pen,
            R=None,
            model=pelt_model
        )

    # --------------------------------------------------------
    # Step 2: MILP + ratio-test refinement
    # --------------------------------------------------------

    Gamma = (
        optimization_integrated_ratio_test_refinement(
            X=X,
            Gamma_0=Gamma_0,
            Delta=Delta,
            eta=eta,
            w=w,
            R=R,
            output_flag=output_flag
        )
    )

    return Gamma_0, Gamma


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
