library(gurobi)
library(Matrix)
library(changepoint)


# ============================================================
# Fixed-number MILP
#
# IMPORTANT:
# Length = number of SEGMENTS
#
#   1 change point  -> Length = 2
#   m change points -> Length = m + 1
# ============================================================

change_point_detection_fixed_num <- function(    Length, X, Delta, output_flag = 0) {

  L <- as.integer(Length)
  X <- as.numeric(X)

  u_vals <- sort(X)

  n <- length(X)
  delta <- as.integer(Delta)

  if (L < 1) {
    stop("Length must be at least 1.")
  }

  if (n < L * delta) {
    stop(
      paste0(   "Infeasible: n=", n,   " < Length*Delta=", L * delta )
    )
  }


  # ==========================================================
  # Variable indices
  # ==========================================================

  n_z <- n * L
  n_s <- n * L
  n_cdf <- n * L
  n_t <- n * L
  n_k <- L
  n_b <- n * L * n
  n_diff <- n * L

  offset_z <- 0
  offset_s <- offset_z + n_z
  offset_cdf <- offset_s + n_s
  offset_t <- offset_cdf + n_cdf
  offset_k <- offset_t + n_t
  offset_b <- offset_k + n_k
  offset_diff <- offset_b + n_b

  nvar <- offset_diff + n_diff


  idx_z <- function(i, l) {   offset_z + (l - 1) * n + i }

  idx_s <- function(i, l) {  offset_s + (l - 1) * n + i}

  idx_cdf <- function(i, l) {  offset_cdf + (l - 1) * n + i}

  idx_t <- function(i, l) {  offset_t + (l - 1) * n + i}

  idx_k <- function(l) {  offset_k + l}

  idx_b <- function(i, l, u) {    offset_b +  ((l - 1) * n + (u - 1)) * n +    i}

  idx_diff <- function(i, l) { offset_diff + (l - 1) * n + i}


  # ==========================================================
  # Objective, bounds, and variable types
  # ==========================================================

  obj <- numeric(nvar)

  lb <- rep(0, nvar)
  ub <- rep(Inf, nvar)

  vtype <- rep("C", nvar)


  # z variables

  for (l in seq_len(L)) {
    for (i in seq_len(n)) {
      vtype[idx_z(i, l)] <- "B"
      ub[idx_z(i, l)] <- 1
    }
  }


  # s objective coefficients

  for (l in seq_len(L)) {
    for (u in seq_len(n)) {
      obj[idx_s(u, l)] <- 1
    }
  }


  # cdf bounds

  for (l in seq_len(L)) {
    for (u in seq_len(n)) {
      ub[idx_cdf(u, l)] <- 1
    }
  }


  # k bounds

  for (l in seq_len(L)) {
    ub[idx_k(l)] <- 1 / delta
  }


  # b bounds

  for (l in seq_len(L)) {
    for (u in seq_len(n)) {
      for (i in seq_len(n)) {
        ub[idx_b(i, l, u)] <- 1
      }
    }
  }


  # diff bounds

  for (l in seq_len(L)) {
    for (u in seq_len(n)) {
      ub[idx_diff(u, l)] <- 1
    }
  }


  # ==========================================================
  # Sparse constraint matrix construction
  # ==========================================================

  A_i <- integer(0)
  A_j <- integer(0)
  A_x <- numeric(0)

  rhs <- numeric(0)
  sense <- character(0)

  row_id <- 0


  add_constr <- function(cols, vals, constraint_sense, constraint_rhs) {
    row_id <<- row_id + 1
    keep <- abs(vals) > 1e-15
    cols <- cols[keep]
    vals <- vals[keep]
    A_i <<- c( A_i, rep(row_id, length(cols)))
    A_j <<- c(  A_j,  as.integer(cols))
    A_x <<- c(  A_x,  as.numeric(vals))

    sense <<- c(  sense,  constraint_sense)

    rhs <<- c( rhs, constraint_rhs)
  }


  # ==========================================================
  # Each observation belongs to exactly one segment
  # ==========================================================

  for (i in seq_len(n)) {
    cols <- sapply(    seq_len(L),  function(l) idx_z(i, l))
    vals <- rep(1, L)
    add_constr(   cols,   vals,   "=",   1 )
  }


  # ==========================================================
  # Minimum segment length
  # ==========================================================

  for (l in seq_len(L)) {
    cols <- sapply(  seq_len(n),  function(i) idx_z(i, l))
    vals <- rep(1, n)
    add_constr(   cols,   vals,   ">",   delta )
  }


  # ==========================================================
  # Monotonic segment assignment
  # ==========================================================

  if (n >= 2) {
    for (i in seq_len(n - 1)) {
      for (l in seq_len(L)) {
        lp_values <- l:L
        cols <- c(  idx_z(i, l),  sapply(    lp_values,  function(lp) {    idx_z(i + 1, lp)  } ))
        vals <- c(  1,  rep(-1, length(lp_values)))
        add_constr(  cols,  vals,  "<",  0)
      }
    }
  }


  # ==========================================================
  # Indicators
  # indicators[u,i] = I(X[i] <= u_vals[u])
  # ==========================================================

  indicators <- outer( u_vals, X, FUN = function(u, x) {   as.numeric(x <= u) })


  # ==========================================================
  # CDF constraints
  # ==========================================================

  for (l in seq_len(L)) {
    # sum_i t[i,l] = 1
    cols <- sapply(seq_len(n),function(i) idx_t(i, l))
    vals <- rep(1, n)
    add_constr( cols,vals, "=", 1)

    for (u in seq_len(n)) {
      # cdf_l[u,l] =sum indicators[u,i] * t[i,l]
      cols <- c( idx_cdf(u, l), sapply(seq_len(n),function(i) idx_t(i, l)))
      vals <- c(1,-indicators[u, ])
      add_constr(cols,vals, "=",  0)

      # t[u,l] <= z[u,l]
      add_constr(c(idx_t(u, l),idx_z(u, l) ), c(1,-1), "<",0 )
                                       
      # t[u,l] <= k[l] + 1/n*z[u,l] - 1/n

      add_constr(c(idx_t(u, l),idx_k(l),idx_z(u, l)),c(1,-1,-1 / n),"<", -1 / n)
      
      # t[u,l] >= 1/n*z[u,l]
      add_constr(c(idx_t(u, l),idx_z(u, l)),c( 1,-1 / n ), ">",0)


      # t[u,l] >= k[l] + z[u,l] - 1
      add_constr(c( idx_t(u, l),idx_k(l),idx_z(u, l)),c(1,-1,-1),">", -1 )
    }
  }


  # ==========================================================
  # Linearization b
  # ==========================================================

  for (l in seq_len(L)) {
    for (u in seq_len(n)) {
      for (i in seq_len(n)) {

        # b <= diff
        add_constr( c(idx_b(i, l, u),idx_diff(u, l)),c(  1,-1),"<",0)

        # b <= z
        add_constr(c(idx_b(i, l, u),idx_z(i, l) ), c( 1, -1),"<",0)

        # b >= z + diff - 1
        add_constr( c( idx_b(i, l, u), idx_z(i, l),idx_diff(u, l) ),c(1, -1,-1 ),">", -1)
      }
    }
  }


  # ==========================================================
  # diff + cdf = 1
  #
  # s = sum indicators * b
  # ==========================================================

  for (l in seq_len(L)) {
    for (u in seq_len(n)) {
      add_constr( c( idx_diff(u, l), idx_cdf(u, l)),c( 1, 1 ),"=", 1)
      cols <- c( idx_s(u, l), sapply(seq_len(n),function(i) {idx_b(i, l, u)}))
      vals <- c( 1,  -indicators[u, ])
      add_constr( cols, vals, "=",0 )
    }
  }


  # ==========================================================
  # Gurobi model
  # ==========================================================

  model <- list()

  model$A <- sparseMatrix( i = A_i, j = A_j, x = A_x, dims = c(row_id, nvar))

  model$obj <- obj
  model$rhs <- rhs
  model$sense <- sense

  model$lb <- lb
  model$ub <- ub

  model$vtype <- vtype

  model$modelsense <- "min"


  params <- list(OutputFlag = as.integer(output_flag))


  result <- gurobi( model, params )


  if (result$status != "OPTIMAL") {
    stop(paste0( "Gurobi did not return an optimal solution. ","Status = ",result$status ))
  }


  # ==========================================================
  # Extract z
  # ==========================================================

  z_sol <- matrix(0,nrow = n,ncol = L)

  for (l in seq_len(L)) {
    for (i in seq_len(n)) {
      z_sol[i, l] <-result$x[idx_z(i, l)]
    }
  }
  col_sums <- round(colSums(z_sol))

  Z_1 <- cumsum(col_sums )

  return(as.integer(Z_1))
}



# ============================================================
# Segment cost R
#
# IMPORTANT:
# This matches numpy:
#
# np.var(segment)
#
# R's normal var() uses denominator n-1,
# so do NOT use var(segment) if you want the same result.
# ============================================================

segment_cost_variance <- function(X, tau_left, tau_right) {

  if (tau_right <= tau_left) {
    return(0)
  }

  segment <- X[ seq.int(tau_left + 1,tau_right)]

  if (length(segment) == 0) {
    return(0)
  }

  segment_mean <- mean(segment)

  return(mean((segment - segment_mean)^2 ))
}



# ============================================================
# PELT initialization
#
# R package:
# changepoint
#
# Python:
# rpt.Pelt(model="l2")
#
# R analogue used here:
# changepoint::cpt.mean(..., method="PELT")
# ============================================================

pelt_initialization <- function(X,Delta,pen,model = "l2") {

  X <- as.numeric(X)

  if (model != "l2") {
    stop(
      "This R version currently maps model='l2' ",
      "to changepoint::cpt.mean()."
    )
  }


  breakpoints <- changepoint::cpt.mean( data = X, penalty = "Manual", pen.value = pen, method = "PELT",
    test.stat = "Normal", class = FALSE, param.estimates = FALSE, minseglen = Delta)


  # changepoint ends with n.
  #
  # Example:
  #
  # breakpoints = c(100, 250, 500)
  # Gamma_0 = c(100, 250)

  Gamma_0 <- breakpoints[breakpoints < length(X) ]

  return( as.integer(Gamma_0))
}



# ============================================================
# Algorithm:
# Local Change-point Refinement
# ============================================================

local_change_point_refinement <- function(Gamma,  X,tau_left,tau_right,Delta,output_flag = 0) {

  Gamma <- sort( as.integer(Gamma))

  tau_left <- as.integer( tau_left)

  tau_right <- as.integer( tau_right)


  # ----------------------------------------------------------
  # m = number of change points inside
  # (tau_left, tau_right)
  # ----------------------------------------------------------

  interior_points <- Gamma[Gamma > tau_left &Gamma < tau_right]

  m <- length(interior_points)

  if (m == 0) {
    return(Gamma)
  }


  # ----------------------------------------------------------
  # Local data
  # ----------------------------------------------------------

  X_local <- X[seq.int(tau_left + 1,tau_right)]


  # m change points = m+1 segments

  Length <- m + 1
  
  if ( length(X_local) < Length * Delta) {
    stop( paste0( "Local interval is too short for ", Length, " segments with Delta=", Delta, "."))
  }


  # ----------------------------------------------------------
  # Solve local MILP
  # ----------------------------------------------------------

  Z_local <-change_point_detection_fixed_num(  Length = Length, X = X_local,Delta = Delta,output_flag = output_flag)
  
  # Z_local includes final endpoint

  refined_points <- tau_left + Z_local[seq_len( length(Z_local) - 1 )]

  # ----------------------------------------------------------
  # Keep Gamma outside interval
  # ----------------------------------------------------------

  Gamma_outside <- Gamma[ !( Gamma > tau_left & Gamma < tau_right)]
  Gamma <- sort(c( Gamma_outside, refined_points))

  return( as.integer(Gamma))
}



# ============================================================
# Total segment cost
# ============================================================

total_segment_cost <- function(Gamma, X, R) {

  Gamma <- sort(as.integer(Gamma))

  boundaries <- c( 0, Gamma, length(X))

  total <- 0

  for (i in seq_len(  length(boundaries) - 1) ) {
    total <- total + R(X,boundaries[i],boundaries[i + 1])
  }

  return(as.numeric(total))
}

# ============================================================
# Compute delta_merge(Gamma)
# ============================================================

compute_delta_merge <- function(Gamma, X, R) {

  Gamma <- sort(as.integer(Gamma))

  L <- length(Gamma)
  if (L == 0) {
    return(list(delta_merge = Inf,i_star = NULL))
  }

  boundaries <- c(0,Gamma,length(X))

  denominator <-total_segment_cost( Gamma, X, R )

  if (abs(denominator) <= 1e-12) {
    return(list(delta_merge = Inf,i_star = NULL))
  }

  delta_merge <- Inf
  i_star <- NULL


  # ----------------------------------------------------------
  # Examine every change point
  # ----------------------------------------------------------

  for (i in seq_len(L)) {

    old_cost <- R( X, boundaries[i], boundaries[i + 1] ) +R(X,boundaries[i + 1],boundaries[i + 2])
    merged_cost <-R( X, boundaries[i], boundaries[i + 2])
    
    delta_i <-( merged_cost -  old_cost ) / denominator
    
    if ( delta_i < delta_merge) {
      delta_merge <- delta_i
      i_star <- i
    }
  }

  return(list(delta_merge = as.numeric(delta_merge),i_star =i_star))
}

# ============================================================
# Compute delta_split(Gamma)
# ============================================================

compute_delta_split <- function(Gamma, X,Delta, R, output_flag = 0) {

  Gamma <- sort( as.integer(Gamma) )
  boundaries <- c( 0, Gamma, length(X))
  denominator <-total_segment_cost(  Gamma,  X,  R)

  if (abs(denominator) <= 1e-4) {
    return(list( delta_split = -Inf, i_star = NULL, tau_star = NULL ))
  }

  delta_split <- -Inf
  i_star <- NULL
  tau_star <- NULL

  # ----------------------------------------------------------
  # Test every current segment
  # ----------------------------------------------------------

  for (i in seq_len( length(boundaries) - 1)) {
    tau_left <-boundaries[i]
    tau_right <- boundaries[i + 1]
    
    # Need two segments,each at least Delta

    if ( tau_right - tau_left <2 * Delta ) {
      next
    }
    X_local <- X[seq.int(tau_left + 1,tau_right)]

    # --------------------------------------------------------
    # One change point = two segments
    # --------------------------------------------------------

    Z_local <-change_point_detection_fixed_num(Length = 2,X = X_local,Delta = Delta,output_flag = output_flag)
    candidate_tau <-tau_left +Z_local[1]

    # --------------------------------------------------------
    # Cost before splitting
    # --------------------------------------------------------

    old_cost <- R(X,tau_left,tau_right)

    # --------------------------------------------------------
    # Cost after splitting
    # --------------------------------------------------------

    new_cost <-R( X, tau_left, candidate_tau ) + R( X,candidate_tau,  tau_right)
    delta_i <- (old_cost -  new_cost) /denominator
    if (delta_i >delta_split) {
      delta_split <- delta_i
      i_star <- i
      tau_star <-as.integer(candidate_tau)
    }
  }

  return(list(delta_split = as.numeric(delta_split),i_star = i_star,tau_star = tau_star))
}



# ============================================================
# Algorithm:
# Optimization-Integrated Ratio-Test Refinement
# ============================================================

optimization_integrated_ratio_test_refinement <- function( X, Gamma_0, Delta, eta, w, R = segment_cost_variance, output_flag = 0) {
  X <- as.numeric(X)
  n <- length(X)
  Gamma <- sort(as.integer( Gamma_0[Gamma_0 > 0 &Gamma_0 < n ]))
  if (w < 1) {
    stop("w must be at least 1.")
  }
  if (eta < 0) {
    stop("eta must be nonnegative.")
  }

  # ==========================================================
  # Compute merge test
  # ==========================================================

  merge_result <-compute_delta_merge( Gamma = Gamma, X = X, R = R)
  delta_merge <-merge_result$delta_merge
  i_star <-merge_result$i_star

  # ==========================================================
  # MERGE PHASE
  # ==========================================================

  if (length(Gamma) > 1 &&!is.null(i_star) && delta_merge < -eta) {
    while (length(Gamma) > 1 &&  !is.null(i_star) &&  delta_merge < -eta) {
      # ------------------------------------------------------
      # Remove selected change point
      # ------------------------------------------------------
      Gamma <-Gamma[-i_star]
      # ------------------------------------------------------
      # Reindex
      # ------------------------------------------------------
      boundaries <- c(  0,  Gamma,  n)

      # ------------------------------------------------------
      # Refinement window
      #
      # R uses 1-based array positions.
      # ------------------------------------------------------

      left_index <- max( 1, i_star - w)
      right_index <-min(length(boundaries),i_star + w + 1)
      tau_left <- boundaries[left_index]
      tau_right <-boundaries[right_index]

      # ------------------------------------------------------
      # Local refinement
      # ------------------------------------------------------

      Gamma <-local_change_point_refinement(Gamma = Gamma,X = X,tau_left = tau_left, tau_right = tau_right, Delta = Delta,output_flag = output_flag)
      # ------------------------------------------------------
      # Recompute merge test
      # ------------------------------------------------------

      merge_result <- compute_delta_merge( Gamma = Gamma, X = X, R = R)
      delta_merge <-merge_result$delta_merge
      i_star <-merge_result$i_star
    }


  } else {
    # ========================================================
    # SPLIT PHASE
    # ========================================================

    split_result <-compute_delta_split( Gamma = Gamma, X = X, Delta = Delta,  R = R, output_flag = output_flag )
    delta_split <-split_result$delta_split
    i_star <-split_result$i_star
    tau_star <-split_result$tau_star
    while (  !is.null(tau_star) &&delta_split > eta ) {
      # ------------------------------------------------------
      # Add tau*
      # ------------------------------------------------------
      Gamma <- sort(c( Gamma,  as.integer(tau_star) ))
      # ------------------------------------------------------
      # Recompute best split
      # ------------------------------------------------------
      split_result <-compute_delta_split( Gamma = Gamma,X = X,  Delta = Delta, R = R, output_flag = output_flag )
      delta_split <- split_result$delta_split
      i_star <-  split_result$i_star
      tau_star <- split_result$tau_star
    }
  }
  return(as.integer(Gamma) )
}

# ============================================================
# Complete wrapper
#
# PELT ->Gamma_0 ->MILP ratio-test refinement
# ============================================================

pelt_milp_ratio_refinement <- function( X, Delta,  eta,  w,pelt_pen,  R = segment_cost_variance, pelt_model = "l2",  output_flag = 0) {
  # ----------------------------------------------------------
  # Step 1: PELT
  # ----------------------------------------------------------
  Gamma_0 <-pelt_initialization(X = X, Delta = Delta,pen = pelt_pen,model = pelt_model)

  # ----------------------------------------------------------
  # Step 2: MILP refinement
  # ----------------------------------------------------------
  Gamma <- optimization_integrated_ratio_test_refinement( X = X, Gamma_0 = Gamma_0, Delta = Delta,eta = eta,w = w, R = R, output_flag = output_flag )

  return( list(Gamma_0 = Gamma_0, Gamma = Gamma) )
}



# ============================================================
# Example
# ============================================================

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
