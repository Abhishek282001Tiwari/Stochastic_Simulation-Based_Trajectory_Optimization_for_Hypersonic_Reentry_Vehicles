# Mathematical Foundations

## Overview

This document presents the mathematical foundations underlying the Hypersonic Reentry Trajectory Optimization Framework, including the fundamental equations, numerical methods, and theoretical background.

---

## 1. Vehicle Dynamics

### 1.1 Coordinate Systems

#### Spherical Earth-Centered Coordinates

The framework uses a spherical Earth-centered coordinate system where the vehicle state is represented as:

$$\\vec{r} = [\\phi, \\lambda, h]^T$$

$$\\vec{v} = [V, \\gamma, \\psi]^T$$

where:
- $\\phi$ = latitude (rad)
- $\\lambda$ = longitude (rad)  
- $h$ = altitude above Earth surface (m)
- $V$ = velocity magnitude (m/s)
- $\\gamma$ = flight path angle (rad)
- $\\psi$ = heading angle (rad)

#### Coordinate Transformations

**Position transformation from spherical to Cartesian:**

$$\\begin{align}
x &= (R_E + h) \\cos\\phi \\cos\\lambda \\\\
y &= (R_E + h) \\cos\\phi \\sin\\lambda \\\\
z &= (R_E + h) \\sin\\phi
\\end{align}$$

where $R_E = 6.371 \\times 10^6$ m is Earth's radius.

**Velocity transformation:**

$$\\begin{align}
\\dot{x} &= V [\\cos\\gamma \\cos\\psi \\cos\\phi \\cos\\lambda - \\sin\\psi \\sin\\lambda - \\sin\\gamma \\sin\\phi \\cos\\lambda] \\\\
\\dot{y} &= V [\\cos\\gamma \\cos\\psi \\cos\\phi \\sin\\lambda + \\sin\\psi \\cos\\lambda - \\sin\\gamma \\sin\\phi \\sin\\lambda] \\\\
\\dot{z} &= V [\\cos\\gamma \\cos\\psi \\sin\\phi + \\sin\\gamma \\cos\\phi]
\\end{align}$$

### 1.2 Equations of Motion

#### 3-DOF Point Mass Dynamics

The equations of motion for a 3-DOF point mass hypersonic vehicle are:

$$\\begin{align}
\\dot{\\phi} &= \\frac{V \\cos\\gamma \\cos\\psi}{R_E + h} \\\\
\\dot{\\lambda} &= \\frac{V \\cos\\gamma \\sin\\psi}{(R_E + h) \\cos\\phi} \\\\
\\dot{h} &= V \\sin\\gamma \\\\
\\dot{V} &= -g \\sin\\gamma - \\frac{D}{m} + \\omega_E^2 (R_E + h) \\cos\\phi [\\sin\\gamma \\cos\\phi - \\cos\\gamma \\sin\\phi \\cos\\psi] \\\\
\\dot{\\gamma} &= \\frac{1}{V} \\left[ \\frac{L \\cos\\sigma}{m} - g \\cos\\gamma + \\frac{V^2 \\cos\\gamma}{R_E + h} + 2\\omega_E V \\cos\\phi \\sin\\psi + \\omega_E^2 (R_E + h) \\cos\\phi [\\cos\\gamma \\cos\\phi + \\sin\\gamma \\sin\\phi \\cos\\psi] \\right] \\\\
\\dot{\\psi} &= \\frac{1}{V \\cos\\gamma} \\left[ \\frac{L \\sin\\sigma}{m} + \\frac{V^2 \\cos\\gamma \\sin\\psi \\tan\\phi}{R_E + h} - 2\\omega_E V [\\sin\\phi - \\cos\\phi \\cos\\psi \\tan\\gamma] \\right]
\\end{align}$$

where:
- $g$ = gravitational acceleration
- $D$ = drag force
- $L$ = lift force  
- $m$ = vehicle mass
- $\\sigma$ = bank angle (control variable)
- $\\omega_E$ = Earth rotation rate
- $R_E$ = Earth radius

#### Gravitational Acceleration

$$g(h) = \\frac{\\mu}{(R_E + h)^2}$$

where $\\mu = 3.986 \\times 10^{14}$ m³/s² is Earth's gravitational parameter.

### 1.3 Aerodynamic Forces

#### Drag Force

$$D = \\frac{1}{2} \\rho V^2 S C_D$$

#### Lift Force

$$L = \\frac{1}{2} \\rho V^2 S C_L$$

where:
- $\\rho$ = atmospheric density
- $S$ = reference area
- $C_D$ = drag coefficient
- $C_L$ = lift coefficient

#### Hypersonic Aerodynamic Modeling

For hypersonic vehicles, the aerodynamic coefficients depend on Mach number, angle of attack, and vehicle geometry:

$$\\begin{align}
C_D &= C_{D,0} + C_{D,\\alpha} \\alpha + C_{D,\\alpha^2} \\alpha^2 \\\\
C_L &= C_{L,\\alpha} \\alpha + C_{L,\\alpha^3} \\alpha^3
\\end{align}$$

where $\\alpha$ is the angle of attack (control variable).

---

## 2. Atmospheric Modeling

### 2.1 US Standard Atmosphere 1976

The framework implements the US Standard Atmosphere 1976 model with seven atmospheric layers.

#### Layer Structure

| Layer | Altitude Range (km) | Temperature Gradient (K/m) |
|-------|-------------------|---------------------------|
| 1 | 0 - 11 | -0.0065 |
| 2 | 11 - 20 | 0.0 |
| 3 | 20 - 32 | 0.001 |
| 4 | 32 - 47 | 0.0028 |
| 5 | 47 - 51 | 0.0 |
| 6 | 51 - 71 | -0.0028 |
| 7 | 71 - 84.852 | -0.002 |

#### Temperature Model

For layers with constant temperature gradient:

$$T(h) = T_b + L_b (h - h_b)$$

For isothermal layers ($L_b = 0$):

$$T(h) = T_b$$

where:
- $T_b$ = base temperature
- $L_b$ = temperature lapse rate
- $h_b$ = base altitude

#### Pressure Model

For layers with non-zero temperature gradient:

$$P(h) = P_b \\left(\\frac{T(h)}{T_b}\\right)^{-\\frac{g_0 M}{R L_b}}$$

For isothermal layers:

$$P(h) = P_b \\exp\\left(-\\frac{g_0 M (h - h_b)}{R T_b}\\right)$$

where:
- $g_0 = 9.80665$ m/s² (standard gravity)
- $M = 0.0289644$ kg/mol (molar mass of air)
- $R = 8.31432$ J/(mol·K) (universal gas constant)

#### Density Model

$$\\rho(h) = \\frac{P(h) M}{R T(h)}$$

### 2.2 Atmospheric Uncertainty Modeling

Atmospheric uncertainties are modeled using multiplicative factors:

$$\\rho_{uncertain}(h) = \\rho_{nominal}(h) \\cdot f_{\\rho}$$

where $f_{\\rho}$ follows a log-normal distribution:

$$\\ln(f_{\\rho}) \\sim \\mathcal{N}(\\mu_{\\ln}, \\sigma_{\\ln}^2)$$

with parameters chosen to achieve desired uncertainty levels (typically 10-20% standard deviation).

---

## 3. Uncertainty Quantification

### 3.1 Monte Carlo Simulation

#### Standard Monte Carlo

For $n$ uncertain parameters $\\vec{\\xi} = [\\xi_1, \\xi_2, \\ldots, \\xi_n]^T$, Monte Carlo simulation generates $N$ samples:

$$\\vec{\\xi}^{(i)} \\sim p(\\vec{\\xi}), \\quad i = 1, 2, \\ldots, N$$

The output statistics are estimated as:

$$\\begin{align}
\\hat{\\mu}_Y &= \\frac{1}{N} \\sum_{i=1}^N Y(\\vec{\\xi}^{(i)}) \\\\
\\hat{\\sigma}_Y^2 &= \\frac{1}{N-1} \\sum_{i=1}^N [Y(\\vec{\\xi}^{(i)}) - \\hat{\\mu}_Y]^2
\\end{align}$$

#### Latin Hypercube Sampling

Latin Hypercube Sampling (LHS) provides more efficient sampling by stratifying the parameter space:

1. Divide each parameter range into $N$ equal-probability intervals
2. Sample once from each interval
3. Randomly permute the samples to create $N$ parameter combinations

#### Sobol Sequences

Low-discrepancy Sobol sequences provide quasi-random sampling with better coverage:

$$\\vec{\\xi}^{(i)} = \\Phi^{-1}(\\vec{u}^{(i)})$$

where $\\vec{u}^{(i)}$ is the $i$-th Sobol sequence point and $\\Phi^{-1}$ is the inverse CDF.

### 3.2 Polynomial Chaos Expansion

#### Theory

Polynomial Chaos Expansion (PCE) represents the output as:

$$Y(\\vec{\\xi}) = \\sum_{\\alpha \\in \\mathbb{N}^n} y_{\\alpha} \\Psi_{\\alpha}(\\vec{\\xi})$$

where:
- $y_{\\alpha}$ are the PCE coefficients
- $\\Psi_{\\alpha}$ are orthogonal polynomials
- $\\alpha$ is a multi-index

#### Orthogonal Polynomials

For different parameter distributions:

| Distribution | Polynomial Type | Weight Function |
|-------------|----------------|-----------------|
| Uniform | Legendre | $w(\\xi) = 1$ |
| Normal | Hermite | $w(\\xi) = e^{-\\xi^2/2}$ |
| Beta | Jacobi | $w(\\xi) = (1-\\xi)^a (1+\\xi)^b$ |
| Gamma | Laguerre | $w(\\xi) = \\xi^a e^{-\\xi}$ |

#### Coefficient Computation

Using spectral projection:

$$y_{\\alpha} = \\frac{\\langle Y, \\Psi_{\\alpha} \\rangle}{\\langle \\Psi_{\\alpha}, \\Psi_{\\alpha} \\rangle}$$

Or regression-based approach:

$$\\vec{y} = (\\mathbf{A}^T \\mathbf{A})^{-1} \\mathbf{A}^T \\vec{Y}$$

where $\\mathbf{A}_{ij} = \\Psi_j(\\vec{\\xi}^{(i)})$.

### 3.3 Sensitivity Analysis

#### Sobol Sensitivity Indices

The Sobol variance decomposition:

$$\\text{Var}(Y) = \\sum_i V_i + \\sum_{i<j} V_{ij} + \\sum_{i<j<k} V_{ijk} + \\ldots + V_{1,2,\\ldots,n}$$

where:
- $V_i$ = first-order variance due to parameter $\\xi_i$
- $V_{ij}$ = second-order variance due to interaction between $\\xi_i$ and $\\xi_j$

#### First-Order Sensitivity Index

$$S_i = \\frac{V_i}{\\text{Var}(Y)} = \\frac{\\text{Var}(\\mathbb{E}[Y|\\xi_i])}{\\text{Var}(Y)}$$

#### Total Effect Index

$$S_T^i = \\frac{\\mathbb{E}[\\text{Var}(Y|\\vec{\\xi}_{\\sim i})]}{\\text{Var}(Y)} = 1 - \\frac{\\text{Var}(\\mathbb{E}[Y|\\vec{\\xi}_{\\sim i}])}{\\text{Var}(Y)}$$

where $\\vec{\\xi}_{\\sim i}$ denotes all parameters except $\\xi_i$.

#### Computational Method

Using the Saltelli sampling scheme:

1. Generate two independent matrices $\\mathbf{A}$ and $\\mathbf{B}$ of size $N \\times n$
2. Create matrices $\\mathbf{C}_i$ by replacing the $i$-th column of $\\mathbf{A}$ with the $i$-th column of $\\mathbf{B}$
3. Compute model evaluations: $Y_A$, $Y_B$, and $Y_{C_i}$
4. Estimate sensitivity indices:

$$\\begin{align}
S_i &\\approx \\frac{\\frac{1}{N} \\sum_{j=1}^N Y_A^{(j)} Y_{C_i}^{(j)} - f_0^2}{\\text{Var}(Y)} \\\\
S_T^i &\\approx 1 - \\frac{\\frac{1}{N} \\sum_{j=1}^N Y_B^{(j)} Y_{C_i}^{(j)} - f_0^2}{\\text{Var}(Y)}
\\end{align}$$

---

## 4. Trajectory Optimization

### 4.1 Problem Formulation

#### General Form

$$\\begin{align}
\\min_{\\vec{u}(t)} \\quad & J = \\phi(\\vec{x}(t_f), t_f) + \\int_{t_0}^{t_f} L(\\vec{x}(t), \\vec{u}(t), t) dt \\\\
\\text{subject to:} \\quad & \\dot{\\vec{x}} = \\vec{f}(\\vec{x}, \\vec{u}, t) \\\\
& \\vec{g}(\\vec{x}, \\vec{u}, t) \\leq 0 \\\\
& \\vec{h}(\\vec{x}, \\vec{u}, t) = 0 \\\\
& \\psi(\\vec{x}(t_0), \\vec{x}(t_f), t_0, t_f) = 0
\\end{align}$$

where:
- $\\vec{x}(t)$ = state vector
- $\\vec{u}(t)$ = control vector  
- $J$ = objective function
- $\\vec{g}$ = inequality constraints
- $\\vec{h}$ = equality constraints
- $\\psi$ = boundary conditions

#### Hypersonic Reentry Specific Formulation

**Objective**: Maximize downrange distance
$$J = -\\int_0^{t_f} V \\cos\\gamma \\cos\\psi \\, dt$$

**State vector**: $\\vec{x} = [\\phi, \\lambda, h, V, \\gamma, \\psi]^T$

**Control vector**: $\\vec{u} = [\\alpha, \\sigma]^T$ (angle of attack, bank angle)

**Path constraints**:
- Heat rate: $\\dot{q} = k \\rho^{0.5} V^3 \\leq \\dot{q}_{max}$
- Dynamic pressure: $q = \\frac{1}{2} \\rho V^2 \\leq q_{max}$
- Load factor: $n = \\frac{\\sqrt{L^2 + D^2}}{mg} \\leq n_{max}$

**Boundary conditions**:
- Initial: prescribed entry conditions
- Final: $h(t_f) = h_{target} \\pm \\delta h$

### 4.2 Sequential Quadratic Programming (SQP)

#### Algorithm Overview

SQP solves the optimization problem by iteratively solving quadratic programming (QP) subproblems:

$$\\begin{align}
\\min_{\\vec{d}} \\quad & \\frac{1}{2} \\vec{d}^T \\mathbf{H}_k \\vec{d} + \\nabla J_k^T \\vec{d} \\\\
\\text{subject to:} \\quad & \\nabla g_{i,k}^T \\vec{d} + g_{i,k} \\leq 0 \\\\
& \\nabla h_{j,k}^T \\vec{d} + h_{j,k} = 0
\\end{align}$$

where:
- $\\vec{d} = \\vec{x}_{k+1} - \\vec{x}_k$ is the search direction
- $\\mathbf{H}_k$ is the Hessian approximation (BFGS update)
- Subscript $k$ denotes the current iteration

#### BFGS Hessian Update

$$\\mathbf{H}_{k+1} = \\mathbf{H}_k + \\frac{\\vec{y}_k \\vec{y}_k^T}{\\vec{y}_k^T \\vec{s}_k} - \\frac{\\mathbf{H}_k \\vec{s}_k \\vec{s}_k^T \\mathbf{H}_k}{\\vec{s}_k^T \\mathbf{H}_k \\vec{s}_k}$$

where:
- $\\vec{s}_k = \\vec{x}_{k+1} - \\vec{x}_k$
- $\\vec{y}_k = \\nabla \\mathcal{L}_{k+1} - \\nabla \\mathcal{L}_k$
- $\\mathcal{L}$ is the Lagrangian function

#### Merit Function

To ensure global convergence, a merit function is used:

$$\\Phi(\\vec{x}, \\mu) = J(\\vec{x}) + \\mu \\left( \\sum_i |h_i(\\vec{x})| + \\sum_j \\max(0, g_j(\\vec{x})) \\right)$$

where $\\mu > 0$ is the penalty parameter.

### 4.3 Transcription Methods

#### Direct Collocation

The continuous optimization problem is discretized using collocation methods:

1. **Time discretization**: Divide $[t_0, t_f]$ into $N$ intervals
2. **State approximation**: Use polynomial interpolation within each interval
3. **Collocation points**: Enforce dynamics at specific points (e.g., Gauss points)

#### Hermite-Simpson Method

State and control approximation:
$$\\begin{align}
\\vec{x}(t) &= \\vec{x}_k + (t - t_k) \\dot{\\vec{x}}_k + \\frac{(t - t_k)^2}{h_k^2} (\\vec{x}_{k+1} - \\vec{x}_k - h_k \\dot{\\vec{x}}_k) \\\\
\\vec{u}(t) &= \\frac{1}{2}[(\\vec{u}_k + \\vec{u}_{k+1}) + (t - t_{k+1/2})(\\vec{u}_{k+1} - \\vec{u}_k)/h_k]
\\end{align}$$

Collocation constraint at midpoint:
$$\\dot{\\vec{x}}_{k+1/2} = \\vec{f}(\\vec{x}_{k+1/2}, \\vec{u}_{k+1/2}, t_{k+1/2})$$

where:
$$\\vec{x}_{k+1/2} = \\frac{1}{2}(\\vec{x}_k + \\vec{x}_{k+1}) + \\frac{h_k}{8}(\\dot{\\vec{x}}_k - \\dot{\\vec{x}}_{k+1})$$

---

## 5. Statistical Analysis

### 5.1 Distribution Fitting

#### Maximum Likelihood Estimation

For a parametric distribution with parameters $\\vec{\\theta}$, the likelihood function is:

$$L(\\vec{\\theta}) = \\prod_{i=1}^n f(y_i | \\vec{\\theta})$$

The MLE estimates maximize the log-likelihood:

$$\\hat{\\vec{\\theta}} = \\arg\\max_{\\vec{\\theta}} \\sum_{i=1}^n \\ln f(y_i | \\vec{\\theta})$$

#### Goodness-of-Fit Tests

**Kolmogorov-Smirnov Test**:
$$D_n = \\max_{1 \\leq i \\leq n} \\left| F_n(y_i) - F(y_i | \\hat{\\vec{\\theta}}) \\right|$$

where $F_n$ is the empirical CDF and $F$ is the fitted CDF.

**Anderson-Darling Test**:
$$A^2 = -n - \\frac{1}{n} \\sum_{i=1}^n (2i-1)[\\ln F(y_i) + \\ln(1 - F(y_{n+1-i}))]$$

#### Information Criteria

**Akaike Information Criterion (AIC)**:
$$\\text{AIC} = 2k - 2\\ln(L)$$

**Bayesian Information Criterion (BIC)**:
$$\\text{BIC} = k\\ln(n) - 2\\ln(L)$$

where $k$ is the number of parameters and $n$ is the sample size.

### 5.2 Confidence Intervals

#### Bootstrap Method

1. Generate $B$ bootstrap samples by resampling with replacement
2. Compute statistic $\\hat{\\theta}^{(b)}$ for each bootstrap sample
3. Percentile method: $CI_{1-\\alpha} = [\\hat{\\theta}_{\\alpha/2}, \\hat{\\theta}_{1-\\alpha/2}]$

#### Bias-Corrected and Accelerated (BCa) Bootstrap

$$CI_{1-\\alpha} = [\\hat{\\theta}_{\\alpha_1}, \\hat{\\theta}_{\\alpha_2}]$$

where:
$$\\begin{align}
\\alpha_1 &= \\Phi\\left(\\hat{z}_0 + \\frac{\\hat{z}_0 + z_{\\alpha/2}}{1 - \\hat{a}(\\hat{z}_0 + z_{\\alpha/2})}\\right) \\\\
\\alpha_2 &= \\Phi\\left(\\hat{z}_0 + \\frac{\\hat{z}_0 + z_{1-\\alpha/2}}{1 - \\hat{a}(\\hat{z}_0 + z_{1-\\alpha/2})}\\right)
\\end{align}$$

$\\hat{z}_0$ is the bias correction and $\\hat{a}$ is the acceleration parameter.

### 5.3 Reliability Analysis

#### Mission Success Probability

For multiple performance criteria, the mission success probability is:

$$P_{success} = P(Y_1 \\in S_1 \\cap Y_2 \\in S_2 \\cap \\ldots \\cap Y_m \\in S_m)$$

where $S_i$ is the success region for output $Y_i$.

#### Subset Simulation

For rare event estimation, subset simulation uses:

$$P_F = P(Y > y_{threshold}) = \\prod_{i=1}^m P(Y > y_i | Y > y_{i-1})$$

where $y_0 < y_1 < \\ldots < y_m = y_{threshold}$.

---

## 6. Numerical Methods

### 6.1 Ordinary Differential Equation Integration

#### Runge-Kutta Methods

**4th-order Runge-Kutta (RK4)**:
$$\\begin{align}
k_1 &= h f(t_n, y_n) \\\\
k_2 &= h f(t_n + h/2, y_n + k_1/2) \\\\
k_3 &= h f(t_n + h/2, y_n + k_2/2) \\\\
k_4 &= h f(t_n + h, y_n + k_3) \\\\
y_{n+1} &= y_n + \\frac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4)
\\end{align}$$

#### Adaptive Step Size Control

Error estimation using embedded methods (e.g., Dormand-Prince):
$$\\text{err} = |y_{n+1}^{(5)} - y_{n+1}^{(4)}|$$

Step size adaptation:
$$h_{new} = h \\left(\\frac{\\text{tol}}{\\text{err}}\\right)^{1/5}$$

### 6.2 Root Finding and Optimization

#### Newton-Raphson Method

$$x_{k+1} = x_k - \\frac{f(x_k)}{f'(x_k)}$$

#### BFGS Quasi-Newton Method

Update formula:
$$\\mathbf{H}_{k+1} = \\mathbf{H}_k + \\frac{\\vec{y}_k \\vec{y}_k^T}{\\vec{y}_k^T \\vec{s}_k} - \\frac{\\mathbf{H}_k \\vec{s}_k \\vec{s}_k^T \\mathbf{H}_k}{\\vec{s}_k^T \\mathbf{H}_k \\vec{s}_k}$$

#### Line Search Methods

**Armijo condition**:
$$f(x_k + \\alpha_k p_k) \\leq f(x_k) + c_1 \\alpha_k \\nabla f_k^T p_k$$

**Wolfe conditions**:
$$\\nabla f(x_k + \\alpha_k p_k)^T p_k \\geq c_2 \\nabla f_k^T p_k$$

where $0 < c_1 < c_2 < 1$.

---

## References

1. Vinh, N. X., et al. \"Hypersonic and Planetary Entry Flight Mechanics.\" University of Michigan Press, 1980.

2. Anderson, J. D. \"Hypersonic and High-Temperature Gas Dynamics.\" 2nd Edition, AIAA, 2006.

3. Sudret, B. \"Global sensitivity analysis using polynomial chaos expansions.\" Reliability Engineering & System Safety, 2008.

4. Sobol, I. M. \"Sensitivity estimates for nonlinear mathematical models.\" Mathematical Modelling and Computational Experiments, 1993.

5. Nocedal, J. and Wright, S. J. \"Numerical Optimization.\" 2nd Edition, Springer, 2006.

6. US Standard Atmosphere 1976. NASA-TM-X-74335, 1976.

7. Betts, J. T. \"Practical Methods for Optimal Control and Estimation Using Nonlinear Programming.\" 2nd Edition, SIAM, 2010.