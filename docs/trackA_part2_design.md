# Track A Part 2 Design: Curvature Methods & Backpropagation in Machine Learning

## 1. Machine Learning as Continuous Optimization

In supervised learning with a parametric model, predictions are parameterized by weights and biases $\theta \in \mathbb{R}^D$:
$$\hat{y}_i = f_\theta(x_i)$$

Given a training dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$, model training is framed as unconstrained continuous optimization:
$$\min_{\theta \in \mathbb{R}^D} \mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^n \ell(f_\theta(x_i), y_i) + \frac{1}{2} \lambda \|\theta_W\|_2^2$$

where:
- $\ell(\cdot, \cdot)$ is the sample loss (e.g., cross-entropy for classification or squared error for regression),
- $\lambda \ge 0$ is the $L_2$ weight regularization (weight decay) penalty coefficient applied to weight matrices $\theta_W$, penalizing model complexity and controlling overfitting.

---

## 2. Backpropagation vs. Optimization

A critical architectural distinction in machine learning:
- **Backpropagation**: An efficient algorithm based on the chain rule for computing the exact analytical first-order gradient vector:
  $$g(\theta) = \nabla_\theta \mathcal{L}(\theta) \in \mathbb{R}^D$$
  It acts strictly as an *oracle* (evaluator) providing function values and gradients.
- **Optimizer**: The numerical update rule that consumes $g(\theta)$ (and optionally higher-order curvature information or stochastic approximations) to produce the sequence of iterates $\theta_0, \theta_1, \dots, \theta_k$. Optimizers include first-order methods (Gradient Descent, Momentum, Adam) and curvature-informed second-order methods (Gauss-Newton, Natural Gradient / Fisher preconditioning).

---

## 3. Curvature and Second-Order Approximations

### 3.1 Hessian-Vector Products (HVP) via Central Finite Differences
Computing and storing the full $D \times D$ Hessian matrix $H = \nabla^2 \mathcal{L}(\theta)$ requires $\mathcal{O}(D^2)$ memory and $\mathcal{O}(D^2)$ or $\mathcal{O}(D^3)$ time, which quickly becomes intractable.

However, many second-order algorithms (e.g., Truncated Newton, Conjugate Gradient, Power Iteration for spectral analysis) require only directional curvature: the directional derivative of the gradient along a vector $v \in \mathbb{R}^D$. This is the Hessian-vector product:
$$H v = \nabla^2 \mathcal{L}(\theta) v = \lim_{\epsilon \to 0} \frac{\nabla \mathcal{L}(\theta + \epsilon v) - \nabla \mathcal{L}(\theta - \epsilon v)}{2\epsilon}$$

Using central finite differences with a small step size $\epsilon > 0$ (typically $\epsilon \approx 10^{-5}$):
$$Hv \approx \frac{g(\theta + \epsilon v) - g(\theta - \epsilon v)}{2\epsilon}$$
This approximation requires only two gradient evaluations and $\mathcal{O}(D)$ memory, without forming $H$.

### 3.2 Gauss-Newton for Non-Linear Least Squares
For non-linear least squares objectives:
$$\min_\theta f(\theta) = \frac{1}{2} \|r(\theta)\|_2^2 = \frac{1}{2} \sum_{i=1}^m r_i(\theta)^2$$
where $r(\theta) \in \mathbb{R}^m$ is the residual vector and $J(\theta) = \frac{\partial r}{\partial \theta} \in \mathbb{R}^{m \times D}$ is the Jacobian matrix.

The true gradient and Hessian are:
$$\nabla f(\theta) = J(\theta)^T r(\theta)$$
$$\nabla^2 f(\theta) = J(\theta)^T J(\theta) + \sum_{i=1}^m r_i(\theta) \nabla^2 r_i(\theta)$$

When residuals are small or the model is nearly linear, the second term $\sum_{i=1}^m r_i \nabla^2 r_i$ is negligible. The **Gauss-Newton approximation** drops this second-order residual term, yielding:
$$H_{\text{GN}} = J^T J$$
Because $J^T J$ is guaranteed to be positive semi-definite (PSD), Gauss-Newton avoids negative curvature directions that plague Newton's method on non-convex objectives.

With Levenberg-Marquardt (LM) damping parameter $\lambda_{\text{LM}} \ge 0$:
$$\Delta \theta = - (J^T J + \lambda_{\text{LM}} I)^{-1} J^T r$$
- When $\lambda_{\text{LM}} = 0$, this is the standard Gauss-Newton step.
- When $\lambda_{\text{LM}} \gg 0$, the step interpolates smoothly toward damped gradient descent, guaranteeing descent even from poor starting points.

### 3.3 The Empirical Fisher Matrix
The Fisher Information Matrix under a probabilistic model measures the curvature of the relative entropy (KL divergence). In deep learning practice, the **empirical Fisher** matrix $F \in \mathbb{R}^{D \times D}$ is defined as the sample covariance of per-sample gradients:
$$F = \frac{1}{n} \sum_{i=1}^n g_i g_i^T$$
where $g_i = \nabla_\theta \ell(f_\theta(x_i), y_i) \in \mathbb{R}^D$ is the gradient of the loss on individual sample $i$.

> **Theoretical Distinction**: The empirical Fisher $F$ is an uncentered covariance matrix of sample gradients evaluated at the current empirical labels. It is *not* identical to the true Fisher (which takes the expectation over the model's predictive distribution) nor to the Hessian or Gauss-Newton matrix, especially far from the optimum. Nonetheless, $F$ captures directional variance in gradient space and is widely used for preconditioning.

To scale preconditioning when $D$ is large:
1. **Diagonal Fisher**:
   $$\text{diag}(F) = \frac{1}{n} \sum_{i=1}^n (g_i \odot g_i)$$
   Requires $\mathcal{O}(D)$ memory and $\mathcal{O}(D)$ inversion.
2. **Low-Rank Fisher (rank $r$)**:
   Via eigendecomposition of $F = \sum_{k=1}^D \lambda_k u_k u_k^T$ with eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_D \ge 0$:
   $$F_{\text{rank-}r} = \sum_{k=1}^r \lambda_k u_k u_k^T$$
   which provides the optimal rank-$r$ approximation in Frobenius norm by the Eckart-Young-Mirsky theorem.

Preconditioned gradient step with damping $\delta > 0$:
$$\theta_{t+1} = \theta_t - \eta (F_{\text{approx}} + \delta I)^{-1} g(\theta_t)$$

---

## 4. Fair-Comparison Protocol

To ensure rigorous and reproducible comparisons across all optimizers and curvature methods:
1. **Identical Datasets**: All methods evaluate on identical train and test splits generated under fixed seeds.
2. **Identical Starting Points**: All models and nonlinear regressions initialize from identical parameter vectors $\theta_0$.
3. **Iteration and Sample Budgets**: Competing optimizers operate under identical iteration budgets and gradient/sample-evaluation accounting.
4. **Hyperparameter Tuning**: Learning rates and damping parameters are swept systematically across a predefined grid over the training set before reporting final test performance.
