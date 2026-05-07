# 📐 Calculus for Machine Learning — Notes


>
> Calculus is a key tool in developing machine learning algorithms and models.
> It offers a mathematical framework for describing how machines **learn and
> optimize** their performance, allowing practitioners to analyze and improve
> the learning process by modeling changes in system behavior.

---

## Table of Contents

1. [Why Calculus in ML](#1-why-calculus-in-ml)
2. [Differentiation](#2-differentiation)
3. [Partial Derivatives](#3-partial-derivatives)
4. [Gradient and Gradient Descent](#4-gradient-and-gradient-descent)
5. [Chain Rule](#5-chain-rule)
6. [Jacobian and Hessian Matrices](#6-jacobian-and-hessian-matrices)
7. [Integrals](#7-integrals)
8. [Applying Calculus in ML Algorithms](#8-applying-calculus-in-ml-algorithms)
9. [Cheat Sheet](#10-cheat-sheet)

---

## 1. Why Calculus in ML

Calculus enables optimization, understanding of algorithms, and function approximation in machine learning.

| Reason | Description |
|--------|-------------|
| **Optimization** | Used in algorithms like gradient descent to minimize or maximize cost functions |
| **Understanding Algorithms** | Helps explain how algorithms work internally — e.g., backpropagation in neural networks |
| **Function Approximation** | Used when exact solutions aren't possible, to approximate functions mathematically |

```
How Calculus Powers ML:

  Training Data
       ↓
  Cost Function  →  measures how wrong the model is
       ↓
  Differentiation → find the gradient (slope)
       ↓
  Gradient Descent → move parameters in direction that reduces cost
       ↓
  Repeat until cost is minimized  →  Trained Model ✅
```

---

## 2. Differentiation

Differentiation measures how a function's output changes with respect to its input. In machine learning, it is used to calculate gradients in gradient descent, optimize cost functions, and understand how small input changes affect predictions.

### Definition

```
The derivative of f(x) with respect to x:

         f(x + h) - f(x)
f'(x) = lim ───────────────
        h→0       h

Notation:  f'(x)  or  df/dx  or  dy/dx
```

### Differentiation Rules

| Rule | Formula | Example |
|------|---------|---------|
| **Power Rule** | d/dx [xⁿ] = n·xⁿ⁻¹ | d/dx [x³] = 3x² |
| **Constant Rule** | d/dx [c] = 0 | d/dx [5] = 0 |
| **Sum Rule** | d/dx [f+g] = f' + g' | d/dx [x²+x] = 2x+1 |
| **Product Rule** | d/dx [f·g] = f'g + fg' | d/dx [x²·sin(x)] |
| **Quotient Rule** | d/dx [f/g] = (f'g - fg') / g² | d/dx [x/eˣ] |
| **Chain Rule** | d/dx [f(g(x))] = f'(g(x))·g'(x) | d/dx [sin(x²)] = cos(x²)·2x |

### Common Derivatives in ML

```
Function          Derivative        Where used in ML
────────────────────────────────────────────────────
f(x) = xⁿ     →  n·xⁿ⁻¹          Polynomial regression
f(x) = eˣ     →  eˣ               Softmax, exponential functions
f(x) = ln(x)  →  1/x             Log-loss (cross-entropy)
f(x) = σ(x)   →  σ(x)·(1-σ(x))  Sigmoid in logistic regression
f(x) = tanh(x) → 1-tanh²(x)     RNN activation
f(x) = max(0,x)→  0 or 1         ReLU activation in neural networks
```

### In ML — Minimizing Cost Function

```
Mean Squared Error (MSE):
  L(w) = (1/n) Σ (yᵢ - ŷᵢ)²
       = (1/n) Σ (yᵢ - w·xᵢ)²

Derivative w.r.t. w:
  dL/dw = (-2/n) Σ xᵢ(yᵢ - w·xᵢ)

Use this gradient to update w → reduce the loss
```

---

## 3. Partial Derivatives

Partial derivatives extend differentiation to functions of multiple variables, measuring how the function changes as one variable changes while others stay constant. They are important in multivariable optimization problems and training models with multiple parameters like neural networks.

### Definition

```
For f(x, y):

  ∂f/∂x  →  derivative w.r.t. x  (treat y as constant)
  ∂f/∂y  →  derivative w.r.t. y  (treat x as constant)
```

### Worked Example

```
f(x, y) = x² + 3xy + y²

∂f/∂x = 2x + 3y    ← differentiate w.r.t. x (y is constant)
∂f/∂y = 3x + 2y    ← differentiate w.r.t. y (x is constant)
```

### In ML — Multi-Parameter Optimization

```
Loss function with multiple weights:
  L(w₁, w₂, w₃) = (y - w₁x₁ - w₂x₂ - w₃x₃)²

Partial derivatives:
  ∂L/∂w₁ = -2x₁(y - w₁x₁ - w₂x₂ - w₃x₃)
  ∂L/∂w₂ = -2x₂(y - w₁x₁ - w₂x₂ - w₃x₃)
  ∂L/∂w₃ = -2x₃(y - w₁x₁ - w₂x₂ - w₃x₃)

Each tells us: "how much does L change when we tweak that weight?"
```

---

## 4. Gradient and Gradient Descent

### Gradient

The gradient is a vector of partial derivatives showing the direction of the steepest ascent of a function.

```
For f(x₁, x₂, ..., xₙ):

         ┌ ∂f/∂x₁ ┐
∇f(x) = │ ∂f/∂x₂ │   ← points in direction of STEEPEST ASCENT
         │   ...   │
         └ ∂f/∂xₙ ┘

To MINIMIZE → move in the OPPOSITE direction of the gradient (-∇f)
```

### Gradient Descent

Gradient descent uses the gradient to find the function's minimum by adjusting model parameters in the opposite direction of the gradient and iteratively minimizing the cost function during training.

```
Update Rule:
  θ_new = θ_old - α · ∇L(θ)

Where:
  θ    = model parameters (weights)
  α    = learning rate (step size)
  ∇L   = gradient of the loss function
  -∇L  = direction that reduces the loss

Visualised:

  Loss
   │  ╲
   │   ╲
   │    ╲___
   │        ╲___
   │             ╲____ minimum
   └───────────────────→ θ
         ← step ←
         (opposite of gradient direction)
```

### Types of Gradient Descent

| Type | Dataset Used | Pros | Cons |
|------|-------------|------|------|
| **Batch GD** | Full dataset per update | Stable convergence | Very slow for large data |
| **Stochastic GD (SGD)** | 1 sample per update | Fast, noisy updates | Can oscillate |
| **Mini-batch GD** | Small batch per update | Balance of speed & stability | Most commonly used in practice |

### Learning Rate Effect

```
α too large  →  overshoots minimum, diverges   ❌
α too small  →  converges very slowly           ⚠️
α just right →  smooth convergence to minimum  ✅

Common values: α = 0.001, 0.01, 0.1
```

---

## 5. Chain Rule

The chain rule computes the derivative of composite functions. It is essential in backpropagation, where derivatives are chained through layers, and calculating gradients efficiently in deep learning models.

### Definition

```
For composite function h(x) = f(g(x)):

  dh/dx = df/dg · dg/dx   =  f'(g(x)) · g'(x)

"Derivative of outer × derivative of inner"
```

### Worked Example

```
h(x) = sin(x²)
     = f(g(x))  where  f(u) = sin(u),  g(x) = x²

  dh/dx = cos(x²) · 2x
           ↑            ↑
    derivative of f   derivative of g
```

### Chain Rule in Backpropagation

Neural networks rely heavily on calculus, especially in the backpropagation algorithm. The chain rule computes the gradient of the loss function with respect to each weight, allowing efficient weight updates during training to reduce the loss function value.

```
Neural Network: Input → Layer1 → Layer2 → Output → Loss

Forward pass:
  z₁ = W₁·x + b₁
  a₁ = σ(z₁)
  z₂ = W₂·a₁ + b₂
  ŷ  = σ(z₂)
  L  = loss(y, ŷ)

Backpropagation using chain rule:
  ∂L/∂W₁ = ∂L/∂ŷ · ∂ŷ/∂z₂ · ∂z₂/∂a₁ · ∂a₁/∂z₁ · ∂z₁/∂W₁
             ↑         ↑          ↑          ↑           ↑
            loss    output    layer2      layer1      weight
           gradient  deriv    weight      activ.      effect

Each gradient flows BACKWARD through the network layers.
```

---

## 6. Jacobian and Hessian Matrices

### Jacobian Matrix

The Jacobian matrix contains all first-order partial derivatives of a vector-valued function.

```
For f: Rⁿ → Rᵐ  (vector function with n inputs, m outputs):

         ┌ ∂f₁/∂x₁  ∂f₁/∂x₂  ...  ∂f₁/∂xₙ ┐
J(f) =   │ ∂f₂/∂x₁  ∂f₂/∂x₂  ...  ∂f₂/∂xₙ │
         │    ...       ...    ...    ...     │
         └ ∂fₘ/∂x₁  ∂fₘ/∂x₂  ...  ∂fₘ/∂xₙ ┘

Shape: (m × n)  →  "all first-order partial derivatives"
```

**In ML:** Used in neural network layers to compute gradients of vector outputs with respect to vector inputs.

### Hessian Matrix

The Hessian matrix contains all second-order partial derivatives of a scalar-valued function. These are used in analyzing curvature of cost functions and implementing advanced optimization techniques like Newton's method.

```
For f: Rⁿ → R  (scalar function):

         ┌ ∂²f/∂x₁²      ∂²f/∂x₁∂x₂  ...  ∂²f/∂x₁∂xₙ ┐
H(f) =   │ ∂²f/∂x₂∂x₁   ∂²f/∂x₂²    ...  ∂²f/∂x₂∂xₙ │
         │     ...           ...       ...      ...      │
         └ ∂²f/∂xₙ∂x₁   ∂²f/∂xₙ∂x₂  ...  ∂²f/∂xₙ²   ┘

Shape: (n × n)  →  "all second-order partial derivatives"
```

### Jacobian vs Hessian

| Matrix | Order | Input→Output | Tells Us |
|--------|-------|-------------|----------|
| **Jacobian** | 1st order | Rⁿ → Rᵐ | Direction of change of each output |
| **Hessian** | 2nd order | Rⁿ → R | Curvature of the cost function |

**In ML:**
```
Hessian → used in Newton's Method optimization:
  θ_new = θ_old - H⁻¹ · ∇L

  Advantage: Converges faster than gradient descent
  Disadvantage: Computing H⁻¹ is very expensive for large models
```

---

## 7. Integrals

### Definite Integral

```
Area under the curve between a and b:

     b
     ⌠
     │ f(x) dx  =  F(b) - F(a)
     ⌡
     a

Where F(x) is the antiderivative of f(x)
```

### Fundamental Theorem of Calculus

```
If F'(x) = f(x), then:

   b
   ⌠
   │ f(x) dx  =  F(b) - F(a)
   ⌡
   a

Differentiation and Integration are inverse operations.
```

### Common Integrals in ML

| Function | Integral | ML Use |
|----------|---------|--------|
| ∫ xⁿ dx | xⁿ⁺¹/(n+1) + C | Polynomial models |
| ∫ eˣ dx | eˣ + C | Exponential distributions |
| ∫ 1/x dx | ln\|x\| + C | Log-likelihood functions |
| ∫ Normal PDF dx | Φ(x) (CDF) | Probability calculations |

### In ML — Expected Value

```
Continuous Expected Value uses integration:

  E[X] = ∫ x · f(x) dx   (f(x) = probability density)

Used in:
  → Computing expected loss
  → Bayesian inference
  → Probability distributions (Normal, Exponential)
```

---

## 8. Applying Calculus in ML Algorithms

### 8.1 Linear Regression

Linear regression uses calculus to derive the normal equations for the least squares solution. The cost function (mean squared error) is minimized using differentiation to find the optimal parameters.

```
Cost Function (MSE):
  L(w, b) = (1/n) Σ (yᵢ - (w·xᵢ + b))²

Step 1 — Differentiate w.r.t. w:
  ∂L/∂w = (-2/n) Σ xᵢ(yᵢ - ŷᵢ)

Step 2 — Differentiate w.r.t. b:
  ∂L/∂b = (-2/n) Σ (yᵢ - ŷᵢ)

Step 3 — Set derivatives to 0 → Normal Equation:
  β = (XᵀX)⁻¹ Xᵀ Y

Step 4 — Gradient Descent Update:
  w ← w - α · ∂L/∂w
  b ← b - α · ∂L/∂b
```

---

### 8.2 Logistic Regression

Logistic regression uses the sigmoid function to model probabilities for binary outcomes. The cost function (log-loss) is minimized using gradient descent, which relies on derivatives. Gradients of the cost function guide parameter updates during training.

```
Sigmoid Function:
  σ(z) = 1 / (1 + e⁻ᶻ)

Derivative of sigmoid:
  σ'(z) = σ(z) · (1 - σ(z))   ← key result for backprop

Log-Loss Cost Function:
  L(w) = -(1/n) Σ [yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ)]

Gradient:
  ∂L/∂w = (1/n) Xᵀ(ŷ - y)

Update:
  w ← w - α · (1/n) Xᵀ(ŷ - y)
```

---

### 8.3 Neural Networks — Backpropagation

Neural networks rely heavily on calculus, especially in the backpropagation algorithm. The chain rule computes the gradient of the loss function with respect to each weight, allowing efficient weight updates during training to reduce the loss function value.

```
Full Backpropagation Flow:

  FORWARD PASS:
    z = W·a + b
    a = activation(z)
    L = loss(y, ŷ)

  BACKWARD PASS (chain rule):
    δL/δW = δL/δa · δa/δz · δz/δW
           = loss_grad · activation_grad · input

  WEIGHT UPDATE:
    W ← W - α · δL/δW
    b ← b - α · δL/δb

Common Activation Derivatives:
  Sigmoid:  σ'(z) = σ(z)(1-σ(z))
  tanh:     tanh'(z) = 1 - tanh²(z)
  ReLU:     ReLU'(z) = 0 if z<0, else 1
```

---

### 8.4 Support Vector Machines (SVMs)

SVMs use calculus to find the optimal separating hyperplane by maximizing the margin between classes. They solve a constrained optimization problem using Lagrange multipliers which involve partial derivatives. The gradient conditions are used to find the points that lie on the margin.

```
Objective: Maximize margin = 2/||w||
           (equivalent to minimizing ||w||²/2)

Optimization using Lagrange Multipliers:
  L(w, b, α) = ||w||²/2 - Σ αᵢ[yᵢ(w·xᵢ + b) - 1]

KKT Conditions (using partial derivatives):
  ∂L/∂w = 0  →  w = Σ αᵢyᵢxᵢ
  ∂L/∂b = 0  →  Σ αᵢyᵢ = 0

Decision boundary: w·x + b = 0
```

---

### Applications Summary Table

| ML Algorithm | Calculus Concept Used | Purpose |
|-------------|----------------------|---------|
| **Linear Regression** | Differentiation, Normal Equation | Minimize MSE loss |
| **Logistic Regression** | Sigmoid derivative, Log-loss gradient | Binary classification |
| **Neural Networks** | Chain Rule, Backpropagation | Weight updates via gradients |
| **SVM** | Lagrange multipliers, Partial derivatives | Maximize margin |
| **Gradient Descent** | Gradient vector, Learning rate | Iterative optimization |
| **Bayesian ML** | Integration (expected values) | Probability inference |

---



## 9. Cheat Sheet

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CALCULUS FOR ML — QUICK REFERENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  DIFFERENTIATION RULES
  Power Rule    d/dx[xⁿ]    = n·xⁿ⁻¹
  Chain Rule    d/dx[f(g)]  = f′(g)·g′
  Product Rule  d/dx[f·g]   = f′g + fg′
  Quotient Rule d/dx[f/g]   = (f′g - fg′) / g²

  COMMON DERIVATIVES IN ML
  d/dx[eˣ]      = eˣ
  d/dx[ln x]    = 1/x
  d/dx[σ(x)]    = σ(x)·(1-σ(x))     ← sigmoid
  d/dx[tanh(x)] = 1 - tanh²(x)      ← tanh
  d/dx[ReLU(x)] = 0 or 1            ← ReLU

  GRADIENT DESCENT
  θ = θ - α·∇L(θ)
    θ = parameters   α = learning rate
    ∇L = gradient of loss

  CHAIN RULE (Backprop)
  ∂L/∂W = ∂L/∂ŷ · ∂ŷ/∂z · ∂z/∂W
  "Multiply gradients layer by layer going backward"

  JACOBIAN    → all 1st-order partial derivatives (matrix)
  HESSIAN     → all 2nd-order partial derivatives (matrix)

  KEY APPLICATIONS
  Linear Reg   → minimize MSE:  ∂L/∂w = (-2/n)Σxᵢ(y-ŷ)
  Logistic Reg → minimize log-loss: ∂L/∂w = (1/n)Xᵀ(ŷ-y)
  Neural Net   → backprop via chain rule through layers
  SVM          → Lagrange multipliers: ∂L/∂w = 0

  GRADIENT TYPES
  Batch GD      → full dataset, stable but slow
  Stochastic GD → 1 sample, fast but noisy
  Mini-batch GD → small batch, best of both ✅

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Further Reading

- 🌐 [Calculus for ML — GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/calculus-for-machine-learning-key-concepts-and-applications/)
- 🌐 [Gradient Descent — GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/gradient-descent-algorithm-and-its-variants/)
- 🌐 [Partial Derivatives in ML — GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/partial-derivatives-in-machine-learning/)
- 🌐 [Chain Rule in ML — GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/chain-rule-derivative-in-machine-learning/)

*Notes compiled from GeeksforGeeks | Calculus for Machine Learning | May 2026*
