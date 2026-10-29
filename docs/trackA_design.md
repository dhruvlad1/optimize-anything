# Track A: Continuous Optimization – Design Document

## 1. Problem Statement

We study unconstrained continuous minimization:

$$\min_{x \in \mathbb{R}^d} f(x)$$

where $f$ is differentiable (twice for Newton methods). We compare first-order
and second-order iterative algorithms on a suite of objectives with known
properties, measuring convergence speed, sensitivity to hyperparameters, and
behaviour under gradient noise.

---

## 2. Optimizers

### 2.1 Gradient Descent (GD)

$$x_{k+1} = x_k - \alpha \nabla f(x_k)$$

- $\alpha$: fixed learning rate (step size).
- One gradient evaluation per iteration.

### 2.2 Stochastic Gradient Descent (SGD)

$$x_{k+1} = x_k - \alpha_k \tilde{g}_k$$

where $\tilde{g}_k$ is an unbiased estimate of $\nabla f(x_k)$ computed from
one sample (or a mini-batch). With a constant step the method oscillates near
the optimum noise floor. For decay schedules of the form $\alpha_k = \alpha_0 / (1 + \beta k)$,
the step size decays harmonically (sum of steps grows only like $\log k$), requiring
proper parameter tuning (such as $\alpha_k = \alpha_0 / (1 + k/\tau)$ or $\alpha_0 / \sqrt{1 + k}$)
to avoid freezing before reaching the minimum.

### 2.3 Mini-Batch SGD

Same as SGD but $\tilde{g}_k$ is averaged over a mini-batch of size $B$:

$$\tilde{g}_k = \frac{1}{B} \sum_{i \in \mathcal{B}_k} \nabla f_i(x_k)$$

Reduces variance by factor $B$ relative to SGD.

### 2.4 Momentum / Heavy Ball

$$m_{k+1} = \beta m_k + \nabla f(x_k)$$
$$x_{k+1} = x_k - \alpha m_{k+1}$$

- $\beta \in [0,1)$: momentum coefficient (typically 0.9).
- Accumulates gradient history to accelerate along consistent directions.

### 2.5 Nesterov Accelerated Gradient (NAG)

$$y_k = x_k + \beta (x_k - x_{k-1})$$
$$x_{k+1} = y_k - \alpha \nabla f(y_k)$$

- Gradient evaluated at the "look-ahead" point $y_k$.
- Achieves optimal $O(1/k^2)$ rate on convex functions.

### 2.6 AdaGrad

$$G_{k+1} = G_k + \nabla f(x_k) \odot \nabla f(x_k)$$
$$x_{k+1} = x_k - \frac{\alpha}{\sqrt{G_{k+1}} + \epsilon} \odot \nabla f(x_k)$$

- Adapts per-coordinate learning rate; large for infrequent features.
- Learning rate decreases monotonically → can stop learning too early.

### 2.7 RMSProp

$$v_{k+1} = \rho v_k + (1-\rho) \nabla f(x_k)^2$$
$$x_{k+1} = x_k - \frac{\alpha}{\sqrt{v_{k+1}} + \epsilon} \odot \nabla f(x_k)$$

- Exponential moving average of squared gradients (fixes AdaGrad decay).

### 2.8 Adam

$$m_{k+1} = \beta_1 m_k + (1-\beta_1) \nabla f(x_k)$$
$$v_{k+1} = \beta_2 v_k + (1-\beta_2) \nabla f(x_k)^2$$
$$\hat{m} = m_{k+1} / (1 - \beta_1^{k+1}), \quad \hat{v} = v_{k+1} / (1 - \beta_2^{k+1})$$
$$x_{k+1} = x_k - \frac{\alpha}{\sqrt{\hat{v}} + \epsilon} \odot \hat{m}$$

- Combines momentum and RMSProp with bias correction.
- Default: $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$.

### 2.9 Newton's Method

$$x_{k+1} = x_k - \alpha [\nabla^2 f(x_k)]^{-1} \nabla f(x_k)$$

Implemented via `numpy.linalg.solve` ($H p = g$, then $x \leftarrow x - \alpha p$).

Two modes:
- **pure**: use $H$ as-is (may converge to saddle points if $H \not\succ 0$).
- **damped**: add $\lambda I$ (doubling $\lambda$ from $10^{-3}$) until $H + \lambda I \succ 0$,
  then solve. Guarantees a descent direction.

---

## 3. Objective Functions

### 3.1 Quadratic (`quadratic(kappa, dim)`)

$$f(x) = \tfrac{1}{2} x^\top A x, \quad A = \mathrm{diag}(\lambda_1, \ldots, \lambda_d)$$

Eigenvalues spread log-uniformly from 1 to $\kappa$ (condition number).

- **Why:** Strongly convex and $L$-smooth with $\mu=1$, $L=\kappa$. Allows
  exact verification of the linear convergence rate $1 - \mu/L = 1 - 1/\kappa$.
- **Minimizer:** $x^* = 0$, $f^* = 0$.

### 3.2 Quartic

$$f(x) = \sum_{i=1}^d x_i^4$$

- **Why:** Convex but *not* strongly convex ($\mu=0$, $f''(0)=0$).
  GD exhibits sublinear convergence near the origin.
- **Minimizer:** $x^* = 0$, $f^* = 0$.

### 3.3 Saddle Function (2-D)

$$f(x,y) = x^2 - y^2 + \tfrac{1}{4} y^4$$

- **Why:** Has a saddle point at the origin ($\nabla^2 f$ has one negative eigenvalue there).
  Near the saddle $(0, y_0)$, GD escapes slowly; escape time grows with $\log(1/y_0)$. In contrast, pure Newton converges *to* saddles.
- **Minima:** $(0, \pm\sqrt{2})$, $f^* = -1$.

### 3.4 Himmelblau (2-D)

$$f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2$$

- **Why:** Four equally valued local minima (all $f=0$), useful for studying
  basin of attraction.
- **Known minima:** $(3,2)$, $(-2.805,3.131)$, $(-3.779,-3.283)$, $(3.584,-1.848)$.

### 3.5 Rosenbrock (2-D)

$$f(x,y) = (1-x)^2 + 100(y - x^2)^2$$

- **Why:** Convex in a global sense but with a narrow curved valley.  Tests
  the ability of adaptive methods (Adam, RMSProp) to follow the valley.
- **Minimizer:** $(1,1)$, $f^* = 0$.

### 3.6 Rastrigin (2-D)

$$f(x) = 10d + \sum_{i=1}^d \bigl[x_i^2 - 10\cos(2\pi x_i)\bigr]$$

- **Why:** Highly multimodal; $\sim 10^d$ local minima. Tests escape from
  local minima and global search ability.
- **Minimizer:** $x^* = 0$, $f^* = 0$.

---

## 4. Fair-Comparison Protocol

| Element | Choice |
|---------|--------|
| Starting points | Same $x_0$ (or same seed for random starts) for all optimizers |
| Max iterations | 2000 (exp1, exp3); 500 (exp2 trajectories) |
| Gradient-norm tolerance | $10^{-8}$ |
| Learning rate selection | **exp3_sensitivity runs first**: grid of 16 log-spaced values in $[10^{-5}, 10^2]$; best rate stored in `results/trackA_tuned_lr.json`; exp1, exp2, exp4 read from that file |
| Stability definition | An optimizer run is **stable** if all iterates are finite and the final objective gap is strictly below the initial gap: $f(x_T) - f^* < f(x_0) - f^*$. A run is **converged** if the final gap is below $10^{-3}$. |
| Seeds | Fixed seeds; stochastic experiments use 10 seeds for mean/std |
| Scipy reference | `scipy.optimize.minimize` (BFGS) used *only* in test assertions, never inside optimizers |

---

## 5. Metrics

| Metric | Definition |
|--------|-----------|
| Objective gap | $f(x_k) - f^*$ |
| Iterations to target | Smallest $k$ such that gap $\leq 10^{-6}$; capped at max_iter |
| Gradient evaluations | Counted by each optimizer (e.g. mini-batch SGD counts 1 per batch regardless of size) |
| Runtime | Wall-clock seconds (`time.perf_counter`) |

---

## 6. Theory to Verify

### 6.1 Linear Convergence on Strongly Convex Functions

For GD with step $\alpha = 1/L$ on an $L$-smooth, $\mu$-strongly convex quadratic function:

The distance to the minimizer contracts as:
$$\|x_{k+1} - x^*\|_2 \leq \left(1 - \frac{\mu}{L}\right) \|x_k - x^*\|_2$$

Because the objective is quadratic, $f(x) - f^* = \frac{1}{2} (x - x^*)^\top A (x - x^*)$, the iterate error along the slowest contracting eigen-direction (with eigenvalue $\lambda_{\min} = \mu$) dominates the tail:
$$x_k^{(1)} = \left(1 - \frac{\mu}{L}\right)^k x_0^{(1)}$$
Squaring this component in the objective gap implies that asymptotically the gap contracts by:
$$\frac{f(x_{k+1}) - f^*}{f(x_k) - f^*} \to \left(1 - \frac{\mu}{L}\right)^2 = \left(1 - \frac{1}{\kappa}\right)^2$$

Empirical rate: fit $\log(\text{gap}_k) = k \log\rho + c$ on the tail of the run (the last half of iterations before reaching $10^{-12}$); compare $\rho$ to $(1 - 1/\kappa)^2$. For $\kappa=10$, $(1-1/10)^2 = 0.81$; for $\kappa=100$, $(1-1/100)^2 = 0.9801$.

### 6.2 Sublinear Convergence on Convex (not SC) Functions

For GD with appropriate step on a convex (non-SC) function:

$$f(x_k) - f^* = O(1/k)$$

Visible on the quartic objective as a slower-than-linear decrease in the log-gap plot.

---

## 7. Known Limitations

- Pure Newton can converge to saddle points or diverge on non-convex objectives.
- SGD with constant step has a noise floor proportional to the step size.
- For $\alpha / (1+k)$, the step size decays harmonically (sum of steps grows only like $\log k$), risking premature stagnation without schedule tuning.
- Adam on mini-batches can get worse with more budget and that the cause has not been verified.
- GD near saddle points escapes slowly; escape time grows with $\log(1/y_0)$.
- AdaGrad's monotonically shrinking effective step can stall on non-convex problems.
- Rastrigin: first-order methods are unlikely to find the global minimum from random starts far from 0.
