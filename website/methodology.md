---
layout: default
title: "Methodology"
description: "Comprehensive methodology for stochastic simulation and trajectory optimization of hypersonic reentry vehicles"
---

# Methodology

This page details the comprehensive mathematical framework and computational methods used in our hypersonic reentry vehicle trajectory optimization research.

## 📋 Overview

Our methodology combines advanced mathematical modeling with modern computational techniques to address the complex challenges of hypersonic reentry under uncertainty:

<div class="methodology-overview">
  <div class="method-flow">
    <div class="method-step">
      <div class="step-number">1</div>
      <h3>Vehicle Dynamics Modeling</h3>
      <p>3-DOF point mass equations with aerodynamic and gravitational forces</p>
    </div>
    
    <div class="method-arrow">→</div>
    
    <div class="method-step">
      <div class="step-number">2</div>
      <h3>Uncertainty Quantification</h3>
      <p>Monte Carlo and polynomial chaos methods for uncertainty propagation</p>
    </div>
    
    <div class="method-arrow">→</div>
    
    <div class="method-step">
      <div class="step-number">3</div>
      <h3>Trajectory Optimization</h3>
      <p>Gradient-based and evolutionary algorithms with constraint handling</p>
    </div>
    
    <div class="method-arrow">→</div>
    
    <div class="method-step">
      <div class="step-number">4</div>
      <h3>Statistical Analysis</h3>
      <p>Comprehensive statistical assessment and reliability evaluation</p>
    </div>
  </div>
</div>

## 🚀 Vehicle Dynamics Model

### Mathematical Formulation

Our vehicle dynamics model is based on the 3-DOF point mass equations of motion in spherical coordinates. The state vector consists of:

$$\mathbf{x} = [h, \phi, \lambda, V, \gamma, \psi]^T$$

Where:
- $h$: altitude above Earth surface (m)
- $\phi$: latitude (rad)  
- $\lambda$: longitude (rad)
- $V$: velocity magnitude (m/s)
- $\gamma$: flight path angle (rad)
- $\psi$: azimuth angle (rad)

### Equations of Motion

The differential equations governing vehicle motion are:

<div class="equation-block">
$$\frac{dh}{dt} = V \sin \gamma$$

$$\frac{d\phi}{dt} = \frac{V \cos \gamma \cos \psi}{r}$$

$$\frac{d\lambda}{dt} = \frac{V \cos \gamma \sin \psi}{r \cos \phi}$$

$$\frac{dV}{dt} = -\frac{D}{m} - g \sin \gamma + \omega_E^2 r \cos \phi (\sin \phi \cos \gamma - \cos \phi \sin \gamma \cos \psi)$$

$$\frac{d\gamma}{dt} = \frac{L \cos \sigma}{mV} - \frac{g \cos \gamma}{V} + \frac{V \cos \gamma}{r} + \text{Coriolis terms}$$

$$\frac{d\psi}{dt} = \frac{L \sin \sigma}{mV \cos \gamma} + \frac{V \cos \gamma \sin \psi \tan \phi}{r} + \text{Earth rotation effects}$$
</div>

Where:
- $r = R_E + h$: geocentric radius
- $g = \mu/r^2$: gravitational acceleration
- $D$, $L$: aerodynamic drag and lift forces
- $\sigma$: bank angle (control input)
- $\omega_E$: Earth rotation rate

### Aerodynamic Forces

Aerodynamic forces are calculated using hypersonic correlations:

<div class="equation-block">
$$D = \frac{1}{2} \rho V^2 S C_D$$

$$L = \frac{1}{2} \rho V^2 S C_L$$
</div>

Where the aerodynamic coefficients are modeled using:

**Modified Newtonian Theory for Drag:**
$$C_D = C_{D,0} + C_{D,\alpha} \sin^2 \alpha$$

**Linearized Theory for Lift:**
$$C_L = C_{L,\alpha} \alpha \cos \sigma$$

### Heat Transfer Modeling

Stagnation point heating is calculated using the Fay-Riddell correlation:

<div class="equation-block">
$$\dot{q}_{stag} = C_h \sqrt{\frac{\rho}{R_n}} V^{3.15}$$
</div>

Where $C_h = 1.7415 \times 10^{-4}$ and $R_n$ is the nose radius.

## 🌍 Atmospheric Model

### US Standard Atmosphere 1976

Our atmospheric model implements the US Standard Atmosphere 1976 with seven distinct layers:

| Layer | Altitude Range (km) | Lapse Rate (K/m) | Description |
|-------|-------------------|-----------------|-------------|
| 1 | 0 - 11 | -0.0065 | Troposphere |
| 2 | 11 - 20 | 0.0 | Tropopause |
| 3 | 20 - 32 | 0.001 | Lower Stratosphere |
| 4 | 32 - 47 | 0.0028 | Upper Stratosphere |
| 5 | 47 - 51 | 0.0 | Stratopause |
| 6 | 51 - 71 | -0.0028 | Lower Mesosphere |
| 7 | 71 - 84.852 | -0.002 | Upper Mesosphere |

### Temperature Profile

Temperature is calculated as:

<div class="equation-block">
$$T(h) = T_b + L_b(h - h_b)$$
</div>

For layers with non-zero lapse rate, or $T(h) = T_b$ for isothermal layers.

### Pressure and Density

Pressure follows the hydrostatic equation:

**For layers with lapse rate:**
<div class="equation-block">
$$P(h) = P_b \left(\frac{T(h)}{T_b}\right)^{-\frac{g_0 M}{R^* L_b}}$$
</div>

**For isothermal layers:**
<div class="equation-block">
$$P(h) = P_b \exp\left(-\frac{g_0 M (h-h_b)}{R^* T_b}\right)$$
</div>

Density is obtained from the ideal gas law:
<div class="equation-block">
$$\rho = \frac{P}{R T}$$
</div>

### Uncertainty Modeling

Atmospheric uncertainties are modeled as:

- **Density uncertainty:** 15% standard deviation (log-normal distribution)
- **Temperature uncertainty:** 5% standard deviation (normal distribution)
- **Wind uncertainty:** 50 m/s standard deviation (normal distribution)

## 📊 Uncertainty Quantification

### Monte Carlo Simulation

Our Monte Carlo approach uses advanced sampling techniques:

#### Latin Hypercube Sampling (LHS)

LHS provides better space-filling properties than random sampling:

1. **Stratification:** Divide each parameter range into $N$ equal-probability intervals
2. **Sampling:** Select one sample from each interval for each parameter
3. **Permutation:** Randomly permute samples to avoid correlation

#### Sobol Sequences

For quasi-random sampling, we use Sobol sequences to achieve better convergence:

<div class="equation-block">
$$x_n^{(i)} = \sum_{j=0}^{\infty} b_j^{(n)} v_j^{(i)}$$
</div>

Where $b_j^{(n)}$ are the binary digits of $n$ and $v_j^{(i)}$ are direction numbers.

### Polynomial Chaos Expansion

For efficient uncertainty propagation, we employ Polynomial Chaos Expansion (PCE):

<div class="equation-block">
$$Y(\xi) = \sum_{i=0}^P c_i \Psi_i(\xi)$$
</div>

Where:
- $Y(\xi)$: model output as function of random variables $\xi$
- $c_i$: PCE coefficients
- $\Psi_i(\xi)$: orthogonal polynomials
- $P$: truncation order

#### Orthogonal Polynomials

We use distribution-appropriate polynomials:

| Distribution | Polynomial Family | Support |
|-------------|------------------|---------|
| Normal | Hermite | $(-\infty, \infty)$ |
| Uniform | Legendre | $[-1, 1]$ |
| Beta | Jacobi | $[0, 1]$ |
| Gamma | Laguerre | $[0, \infty)$ |

#### Coefficient Computation

PCE coefficients are computed using:

**Projection method:**
<div class="equation-block">
$$c_i = \frac{\langle Y, \Psi_i \rangle}{\langle \Psi_i, \Psi_i \rangle} = \frac{\int Y(\xi) \Psi_i(\xi) p(\xi) d\xi}{\int \Psi_i^2(\xi) p(\xi) d\xi}$$
</div>

**Regression method:**
<div class="equation-block">
$$\mathbf{c} = (\mathbf{A}^T \mathbf{A})^{-1} \mathbf{A}^T \mathbf{Y}$$
</div>

### Sensitivity Analysis

#### Sobol Indices

Global sensitivity is quantified using Sobol indices:

**First-order index:**
<div class="equation-block">
$$S_i = \frac{\text{Var}[E(Y|X_i)]}{\text{Var}(Y)}$$
</div>

**Total-effect index:**
<div class="equation-block">
$$S_T^i = 1 - \frac{\text{Var}[E(Y|X_{\sim i})]}{\text{Var}(Y)}$$
</div>

#### Morris Method

For screening studies, we use the Morris method:

<div class="equation-block">
$$\mu_i^* = \frac{1}{r} \sum_{j=1}^r \left| \frac{f(x + \Delta e_i) - f(x)}{\Delta} \right|_j$$
</div>

Where $\mu_i^*$ is the modified mean for parameter $i$.

## 🎯 Trajectory Optimization

### Problem Formulation

The trajectory optimization problem is formulated as:

<div class="equation-block">
$$\begin{align}
\min_{\mathbf{u}(t)} \quad & J = \int_{t_0}^{t_f} L(\mathbf{x}, \mathbf{u}, t) dt + \Phi(\mathbf{x}(t_f)) \\
\text{subject to:} \quad & \dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \mathbf{u}, t) \\
& \mathbf{g}(\mathbf{x}, \mathbf{u}, t) \leq 0 \\
& \boldsymbol{\psi}(\mathbf{x}(t_0), \mathbf{x}(t_f)) = 0 \\
& \mathbf{u}_{\min} \leq \mathbf{u} \leq \mathbf{u}_{\max}
\end{align}$$
</div>

### Control Parameterization

Controls are parameterized using piecewise functions:

**Piecewise Constant:**
<div class="equation-block">
$$u(t) = u_i, \quad t \in [t_i, t_{i+1})$$
</div>

**Piecewise Linear:**
<div class="equation-block">
$$u(t) = u_i + \frac{u_{i+1} - u_i}{t_{i+1} - t_i}(t - t_i)$$
</div>

### Sequential Quadratic Programming (SQP)

Our gradient-based optimization uses SQP with the quadratic subproblem:

<div class="equation-block">
$$\begin{align}
\min_{\mathbf{d}} \quad & \nabla f^T \mathbf{d} + \frac{1}{2} \mathbf{d}^T \mathbf{H} \mathbf{d} \\
\text{subject to:} \quad & \nabla g_i^T \mathbf{d} + g_i = 0, \quad i \in \mathcal{E} \\
& \nabla g_i^T \mathbf{d} + g_i \leq 0, \quad i \in \mathcal{I}
\end{align}$$
</div>

Where $\mathbf{H}$ is the Hessian approximation (BFGS update):

<div class="equation-block">
$$\mathbf{H}_{k+1} = \mathbf{H}_k + \frac{\mathbf{y}_k \mathbf{y}_k^T}{\mathbf{y}_k^T \mathbf{s}_k} - \frac{\mathbf{H}_k \mathbf{s}_k \mathbf{s}_k^T \mathbf{H}_k}{\mathbf{s}_k^T \mathbf{H}_k \mathbf{s}_k}$$
</div>

### Line Search

Step size is determined using Armijo condition:

<div class="equation-block">
$$f(\mathbf{x} + \alpha \mathbf{d}) \leq f(\mathbf{x}) + c_1 \alpha \nabla f^T \mathbf{d}$$
</div>

With $c_1 = 10^{-4}$ and backtracking factor $\rho = 0.5$.

### Robust Optimization

For optimization under uncertainty, we use:

**Expected Value Approach:**
<div class="equation-block">
$$\min_{\mathbf{u}} \quad E[J(\mathbf{x}(\mathbf{u}), \mathbf{u}, \boldsymbol{\xi})]$$
</div>

**Chance Constraints:**
<div class="equation-block">
$$P(g_i(\mathbf{x}, \mathbf{u}, \boldsymbol{\xi}) \leq 0) \geq 1 - \epsilon_i$$
</div>

## 📈 Statistical Analysis

### Distribution Fitting

We fit multiple distributions to output data:

| Distribution | PDF | Parameters |
|-------------|-----|------------|
| Normal | $\frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$, $\sigma$ |
| Log-normal | $\frac{1}{x\sqrt{2\pi\sigma^2}} e^{-\frac{(\ln x-\mu)^2}{2\sigma^2}}$ | $\mu$, $\sigma$ |
| Weibull | $\frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1} e^{-(x/\lambda)^k}$ | $k$, $\lambda$ |
| Beta | $\frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}$ | $\alpha$, $\beta$ |

### Goodness of Fit

Distribution quality is assessed using:

**Kolmogorov-Smirnov test:**
<div class="equation-block">
$$D_n = \sup_x |F_n(x) - F(x)|$$
</div>

**Akaike Information Criterion:**
<div class="equation-block">
$$AIC = 2k - 2\ln(L)$$
</div>

### Reliability Analysis

Mission reliability is calculated as:

<div class="equation-block">
$$R = P(\text{all constraints satisfied}) = \prod_{i=1}^n P(G_i > 0)$$
</div>

Assuming independence, or using copulas for dependent failure modes.

### Confidence Intervals

We compute confidence intervals using:

**Parametric (Normal):**
<div class="equation-block">
$$CI = \bar{x} \pm t_{\alpha/2,n-1} \frac{s}{\sqrt{n}}$$
</div>

**Non-parametric (Bootstrap):**
<div class="equation-block">
$$CI = [Q_{\alpha/2}, Q_{1-\alpha/2}]$$
</div>

Where $Q_p$ is the $p$-th quantile of bootstrap samples.

## 💻 Computational Implementation

### Numerical Integration

Trajectory integration uses adaptive Runge-Kutta methods:

**RK45 (Dormand-Prince):**
<div class="equation-block">
$$\mathbf{x}_{n+1} = \mathbf{x}_n + h \sum_{i=1}^7 b_i \mathbf{k}_i$$
</div>

With embedded error estimation for adaptive step size control.

### Parallel Processing

Monte Carlo simulations are parallelized using:

1. **Task-based parallelism:** Distribute parameter samples across cores
2. **Shared memory:** Use multiprocessing.Pool for CPU-bound tasks
3. **Load balancing:** Dynamic work distribution to handle varying computation times

### Performance Optimization

Key optimizations include:

- **Vectorized operations:** NumPy arrays for mathematical computations
- **Just-in-time compilation:** Numba for hot loops
- **Memory management:** Efficient array operations and garbage collection
- **Caching:** Memoization of expensive function calls

## 🔬 Validation and Verification

### Code Verification

Our implementation is verified through:

1. **Unit tests:** Individual function testing with known solutions
2. **Integration tests:** End-to-end workflow validation
3. **Regression tests:** Ensuring consistency across code changes
4. **Benchmark comparisons:** Validation against published results

### Physical Validation

Model validation includes:

1. **Analytical solutions:** Comparison with simplified analytical cases
2. **Literature comparison:** Validation against published trajectory data
3. **Sensitivity checks:** Verification of expected parameter dependencies
4. **Conservation laws:** Energy and momentum conservation checks

### Uncertainty Validation

UQ methods are validated through:

1. **Analytical test functions:** Known uncertainty propagation
2. **Method comparison:** Monte Carlo vs. PCE consistency
3. **Convergence studies:** Sample size adequacy assessment
4. **Cross-validation:** Out-of-sample prediction accuracy

---

<div class="methodology-footer">
  <h3>📚 References</h3>
  <div class="references">
    <p>1. Vinh, N.X., et al. "Optimal trajectories in atmospheric flight." (1981)</p>
    <p>2. Xiu, D. "Numerical methods for stochastic computations." (2010)</p>
    <p>3. Nocedal, J. & Wright, S.J. "Numerical optimization." (2006)</p>
    <p>4. Sobol, I.M. "Global sensitivity indices for nonlinear mathematical models." (2001)</p>
    <p>5. US Committee on Extension to the Standard Atmosphere. "US standard atmosphere, 1976." (1976)</p>
  </div>
</div>

<style>
.methodology-overview {
  background: linear-gradient(135deg, #f8f9fa, #e9ecef);
  border-radius: 12px;
  padding: 2rem;
  margin: 2rem 0;
}

.method-flow {
  display: flex;
  align-items: center;
  justify-content: center;
  flex-wrap: wrap;
  gap: 1rem;
}

.method-step {
  background: white;
  border: 2px solid #3498db;
  border-radius: 12px;
  padding: 1.5rem;
  text-align: center;
  min-width: 200px;
  max-width: 250px;
  position: relative;
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.method-step:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 25px rgba(52, 152, 219, 0.2);
}

.step-number {
  background: #3498db;
  color: white;
  width: 30px;
  height: 30px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  margin: 0 auto 1rem;
}

.method-step h3 {
  color: #2c3e50;
  margin-bottom: 0.5rem;
  font-size: 1.1rem;
}

.method-step p {
  color: #7f8c8d;
  font-size: 0.9rem;
  line-height: 1.4;
}

.method-arrow {
  font-size: 2rem;
  color: #3498db;
  font-weight: bold;
}

.equation-block {
  background: white;
  border: 1px solid #dee2e6;
  border-left: 4px solid #3498db;
  border-radius: 6px;
  padding: 1.5rem;
  margin: 1.5rem 0;
  overflow-x: auto;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.equation-block .MathJax {
  font-size: 1.1em !important;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin: 1.5rem 0;
  background: white;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

th, td {
  padding: 1rem;
  text-align: left;
  border-bottom: 1px solid #e9ecef;
}

th {
  background: #34495e;
  color: white;
  font-weight: 600;
}

tbody tr:hover {
  background: #f8f9fa;
}

.methodology-footer {
  margin: 3rem 0 2rem;
  padding: 2rem;
  background: #f8f9fa;
  border-radius: 12px;
  border-left: 4px solid #3498db;
}

.references p {
  margin: 0.5rem 0;
  font-size: 0.9rem;
  color: #5d6d7e;
}

@media (max-width: 768px) {
  .method-flow {
    flex-direction: column;
  }
  
  .method-arrow {
    transform: rotate(90deg);
    font-size: 1.5rem;
  }
  
  .method-step {
    min-width: unset;
    max-width: unset;
    width: 100%;
  }
  
  .equation-block {
    font-size: 0.9rem;
  }
  
  table {
    font-size: 0.8rem;
  }
  
  th, td {
    padding: 0.5rem;
  }
}
</style>