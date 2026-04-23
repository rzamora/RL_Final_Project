# Belief-State Hierarchical Reinforcement Learning for Risk-Constrained Portfolio Allocation

> Project Progress Memorandum

A reinforcement learning system for dynamic portfolio allocation under realistic institutional constraints, combining hierarchical control, belief-state regime inference, and risk-aware utility optimization.

---

## Table of Contents

- [Overview](#overview)
- [Problem Formulation](#problem-formulation)
  - [State Space](#state-space)
  - [Action Space and Hierarchical Control](#action-space-and-hierarchical-control)
  - [Reward Function](#reward-function)
  - [Transition Dynamics](#transition-dynamics)
- [Learning Objective](#learning-objective)
  - [PPO](#proximal-policy-optimization-ppo)
  - [SAC](#soft-actor-critic-sac)
  - [Experimental Strategy](#planned-experimental-strategy)
- [Conceptual Integration](#conceptual-integration)
- [Project Status](#current-status-and-next-steps)

---

## Overview

This project investigates the use of **reinforcement learning (RL)** for dynamic portfolio allocation under realistic institutional constraints. The primary objective is to design a system that allocates capital across a small set of assets — **NVDA, AMD, SMH, TLT, and cash (T-Bills)** — while explicitly controlling for risk, drawdowns, and investor preferences.

Unlike traditional portfolio optimization methods, which rely on static assumptions or one-step optimization, this project formulates allocation as a **sequential decision-making problem**. The system adapts daily, rebalancing a portion of the portfolio while considering both current market conditions and the evolving portfolio state.

A key innovation is the use of a **hierarchical reinforcement learning framework** combined with a **belief-state representation** of market regimes. The resulting model is designed to behave more like a professional portfolio manager, balancing growth and risk while responding dynamically to changing market environments.

**Asset Universe:**
- NVDA (NVIDIA)
- AMD (Advanced Micro Devices)
- SMH (Semiconductor ETF)
- TLT (Long-Term Treasury ETF)
- Cash (T-Bills)

---

## Problem Formulation

The portfolio allocation problem is formulated as a **Partially Observable Markov Decision Process (POMDP)**, reflecting the fact that the true underlying market regime is latent and must be inferred from observable financial signals. The environment is defined by the tuple:

$$\langle \mathcal{S}, \mathcal{A}, R, P, \gamma \rangle$$

Where:
- $\mathcal{S}$ — state space
- $\mathcal{A}$ — action space
- $P(s_{t+1} \mid s_t, a_t)$ — transition dynamics
- $R(s_t, a_t)$ — reward function
- $\gamma$ — discount factor

At each time step $t$, the agent observes a state $s_t$ representing a belief-state approximation of the latent market regime:

$$s_t \approx b_t = P(z_t \mid \text{observations})$$

where $z_t$ denotes the unobserved regime.

### State Space

The state aggregates technical indicators, volatility/correlation structure, wavelet-based multi-scale features, and forward-looking signals from the Kronos forecasting system. Collectively, these features allow the agent to infer whether the market is trending, volatile, stable, or undergoing structural transition.

#### Feature Vector

| Component | Role |
|-----------|------|
| Rolling returns (3, 7, 14, 30 days) | Local trend and momentum |
| Technical indicators (RSI, MACD, moving-average gaps) | Local trend and momentum |
| Rolling correlations among NVDA, AMD, SMH, TLT | Cross-asset structure |
| VIX level, changes, and normalized values | Volatility and market stress indicators |
| Relative performance of TLT versus equities | Hedge effectiveness signal |
| Current drawdown | Portfolio state |
| Gross and net exposure | Portfolio state |
| Current sleeve allocation | Portfolio state |
| Wavelets | Multi-scale regime structure |
| Kronos forecasts | Forward-looking regime expectations |

#### Wavelet Features

We utilize a **discrete wavelet transform (DWT)**, specifically the **Daubechies-4 (db4)** wavelet, decomposed up to **level 4**. Financial time series are inherently multi-scale, with different behaviors emerging at:

- **Short-term** — noise / microstructure
- **Medium-term** — momentum / mean reversion
- **Long-term** — trend / regime shifts

For a given price series (e.g., SMH or NVDA), the wavelet transform produces:

$$\text{Signal} = A_4 + D_4 + D_3 + D_2 + D_1$$

| Component | Interpretation |
|-----------|---------------|
| $A_4$ | Low-frequency approximation (long-term trend, bull vs bear) |
| $D_4$, $D_3$ | Medium-to-long horizon fluctuations (cyclical behavior, transitions) |
| $D_2$, $D_1$ | High-frequency noise and short-term volatility bursts |

Wavelet-derived features include the energy (variance) of each component, normalized amplitudes, and ratios between low- and high-frequency energy. These features are included primarily in the **high-level policy state**, as they describe market structure rather than asset-specific tactics.

#### Kronos Forecast Features

In addition to wavelet-based features, we incorporate **multi-horizon forecasts from the Kronos system**, which provide forward-looking information about expected price behavior. While technical indicators and wavelets are backward-looking, Kronos introduces a **forward-looking component**, allowing the agent to incorporate expectations about future market movements.

Kronos provides forecasts such as:
- Predicted future returns (multiple horizons)
- Prediction errors and stability metrics
- Directional accuracy (hit rates)
- Confidence measures

These outputs implicitly encode regime information:

| Forecast Pattern | Regime Implication |
|------------------|-------------------|
| High-confidence, low-error | Stable, predictable regime |
| Low-confidence, high-error | Unstable or transitioning regime |
| Consistent directional signals | Trending environment |
| Erratic signals | Noisy or regime-shifting environment |

Constructed features from Kronos outputs include rolling forecast error (RMSE / absolute error), forecast stability over time, directional hit rate, confidence score, and a "surprise" metric (deviation from expected error). These are included primarily in the **high-level policy state**.

### Action Space and Hierarchical Control

The action space $\mathcal{A}$ is structured hierarchically to reflect the separation between strategic risk allocation and tactical asset selection — a defining characteristic of institutional portfolio management. At each time step:

$$a_t = \left( a_t^{\text{HL}},\; a_t^{\text{LL}} \right)$$

| High-Level Policy (HL) | Low-Level Policy (LL) |
|------------------------|----------------------|
| Determines strategic posture for the sleeve being rebalanced: target **gross exposure** (risk budget) and target **net exposure** (directional bias) | Allocates the selected sleeve across NVDA, AMD, SMH, TLT, with residual capital assigned to **cash (T-Bills)** |

#### Sleeve-Based Execution Constraint

A key structural feature of the action space is the introduction of a **sleeve-based execution mechanism**. The portfolio is partitioned into $K = 5$ equal sleeves, and only **one sleeve is rebalanced at each time step**. The realized portfolio at time $t$ is:

$$w_t = \frac{1}{K} \sum_{k=0}^{K-1} w_{t-k}$$

where each $w_{t-k}$ corresponds to a sleeve allocation decided at a different past time.

This structure introduces **temporal coupling** between actions: a single decision affects portfolio composition over multiple future periods. The agent must learn policies that are inherently forward-looking, balancing immediate signals against their persistent impact on future portfolio states.

### Reward Function

The reward function approximates the objective of a professional asset manager operating under realistic institutional constraints. Rather than maximizing raw one-period returns, the agent optimizes a risk-adjusted utility functional that balances capital growth, drawdown control, trading efficiency, and sustained performance relative to a benchmark.

The agent maximizes:

$$\mathbb{E}\!\left[ \sum_{t=0}^{T} \gamma^t R_t \right]$$

where the per-period reward is:

$$
\begin{aligned}
R_t = \;& \log(1 + r_t) \\
      & - \lambda_{\text{dd}} \, \max\!\left(0,\; DD_t - DD^{*}\right)^{2} \\
      & - \lambda_{\text{to}} \, \tau_t \\
      & - \lambda_{\text{up}} \, \mathbf{1}\{B_t > 0\} \, \max(0,\; B_t) \\
      & - \lambda_{\text{down}} \, \mathbf{1}\{B_t < 0\} \, \max(0,\; -B_t)
\end{aligned}
$$

| Term | Purpose |
|------|---------|
| $\log(1 + r_t)$ | Long-term capital growth (concave transformation) |
| $\lambda_{\text{dd}} \, \max(0, DD_t - DD^{*})^{2}$ | Convex penalty on drawdowns beyond threshold $DD^{*}$ |
| $\lambda_{\text{to}} \, \tau_t$ | Turnover penalty for excessive trading |
| $\lambda_{\text{up}}$, $\lambda_{\text{down}}$ terms | Asymmetric penalties on relative performance vs benchmark |

The smoothed relative-performance process is:

$$B_t = \alpha \left( r_t^{p} - r_t^{\text{bench}} \right) + (1 - \alpha)\, B_{t-1}$$

This captures persistent outperformance or underperformance relative to a benchmark, ensuring the agent is penalized for **sustained deviations** rather than transient daily noise — aligning with practical evaluation horizons (weekly, monthly, quarterly).

Unlike drawdowns (direct capital impairment, requiring convex penalties), relative performance is modeled linearly after temporal aggregation. This prevents excessive benchmark tracking and preserves the agent's ability to pursue active strategies and generate alpha.

### Transition Dynamics

The evolution of the system is governed by:

$$P(s_{t+1} \mid s_t, a_t)$$

In practice:

$$s_{t+1} = f(s_t, a_t, \varepsilon_t)$$

where $f(\cdot)$ represents deterministic updates (portfolio accounting, sleeve aggregation, feature recomputation) and $\varepsilon_t$ represents stochastic market innovations.

#### State Transition Sequence

After rebalancing the selected sleeve:

1. The remaining four sleeves remain unchanged
2. The full portfolio is the aggregation of all five sleeves
3. Market returns are realized at $t+1$
4. Portfolio equity, drawdown, and exposures are updated

Because financial markets are non-stationary and their data-generating process is unknown, the transition distribution is **not explicitly modeled** but is implicitly defined through sampled transitions from historical or simulated data. The model operates as a **belief-state controller**, where features (correlations, volatility, TLT behavior) serve as proxies for hidden regimes.

---

## Learning Objective

The agent learns a policy $\pi(a_t \mid s_t)$ that maximizes expected discounted utility:

$$J(\pi) = \mathbb{E}_{\pi}\!\left[ \sum_{t=0}^{T} \gamma^t R_t \right]$$

The corresponding value functions are:

$$V_{\pi}(s_t) = \mathbb{E}_{\pi}\!\left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k} \;\middle|\; s_t \right]$$

$$Q_{\pi}(s_t, a_t) = \mathbb{E}_{\pi}\!\left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k} \;\middle|\; s_t, a_t \right]$$

These satisfy the Bellman relation:

$$Q_{\pi}(s_t, a_t) = R_t + \gamma \, \mathbb{E}_{s_{t+1}}\!\left[ V_{\pi}(s_{t+1}) \right]$$

The policy is learned using model-free RL algorithms (PPO and SAC), which rely on sampled transitions rather than explicit knowledge of $P(s_{t+1} \mid s_t, a_t)$.

### Proximal Policy Optimization (PPO)

PPO is selected as the **primary baseline algorithm**.

| Rationale | Role in Project | Advantages |
|-----------|-----------------|------------|
| Stable policy updates through clipping | Train both HL and LL policies | Robust under noisy financial data |
| Well-suited for custom environments | Establish baseline performance for return, drawdown, turnover, and market participation | Good balance between exploration and stability |
| Strong empirical performance in continuous control | | Widely accepted in academic and applied RL |
| Simpler to implement and explain | | |

### Soft Actor-Critic (SAC)

SAC is proposed as an **advanced alternative**.

| Rationale | Role in Project | Trade-offs |
|-----------|-----------------|------------|
| Designed for continuous action spaces | Benchmark against PPO | More sensitive to reward scaling |
| Incorporates entropy regularization for exploration | Evaluate whether entropy-driven exploration improves allocation smoothness, hedge utilization (TLT + shorting), and regime adaptation | More complex tuning |
| Often more sample-efficient | | Better suited after baseline validation |

### Planned Experimental Strategy

1. Build and validate environment
2. Train baseline using **PPO**
3. Introduce **SAC** as comparison
4. Evaluate against the following metrics:
   - Cumulative return
   - Maximum drawdown
   - Sharpe ratio
   - Turnover
   - Upside / downside capture

---

## Conceptual Integration

The proposed framework integrates three key elements:

1. **Belief-state representation** — encodes a probabilistic view of latent market regimes
2. **Hierarchical action control** — provides a structured mechanism for risk allocation and asset selection under execution constraints
3. **Institutional utility optimization** — captures realistic investor preferences across multiple time horizons

This combination yields a reinforcement learning system that is not only mathematically well-defined but also aligned with the operational realities of portfolio management. By incorporating delayed action effects, regime inference, and multi-objective utility, the framework moves beyond standard RL formulations and provides a principled approach to adaptive portfolio allocation under uncertainty.

---

## Current Status and Next Steps

### Completed

- [x] Formal MDP / POMDP formulation
- [x] Hierarchical RL architecture (HL + LL)
- [x] Sleeve-based portfolio design
- [x] Feature engineering framework
- [x] Utility function design

### In Progress

- [ ] Environment implementation
- [ ] Validation of projection and constraints
- [ ] Initial policy training setup

### Next Steps

- [ ] Implement PPO baseline
- [ ] Validate training stability
- [ ] Introduce SAC comparison
- [ ] Conduct out-of-sample testing
- [ ] Evaluate robustness across regimes
