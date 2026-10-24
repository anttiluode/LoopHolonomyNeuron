# Design ledger — LoopHolonomyNeuron v0

This is a deliberately small synthetic experiment with declared controls. **No emergent-gravity, biological gauge field, brain holonomy, or new learning-rule claim.**

## Primary question

When a fixed-size *state-bearing* packet makes a multihop trip through state-conditioned operators, does the origin have more information about the intervening state when it observes both its own reference and the returning packet than when it only sees a late return?

What is being measured: recovery of one of two **engineered** operator orders (`A->B->C` versus `B->A->C`) from new noisy probes and initial states.

What is not being measured: the firing of real layer-5 cortical neurons, burst transmission, apical spike calcium coincidence, sensorimotor behavior, true gauge curvature, or emergence of network topology.

## Model

For each node i, use a vector packet `p` and scalar resident state `s_i`:

```
p_out = R_i(theta_i + 0.22*s_i + nuisance_i) @ p_in
s_i <- rho*s_i + eta*tanh(p_in[axis_i])
```

`R_i` is a known 3D rotation. Every operator is exactly norm-preserving at fixed state, and A/B have nonparallel axes. The packet is *not* automatically a transmitted action-potential voltage waveform; real APs are regenerated at each neuron and their effects pass through synapses. Interpret `p` as an abstract low-dimensional packet signature (such as timing/burst features) with deliberately simplified transformations.

The return observation has four imposed time bins: `y_t = a_t Rz(.23) p + b_t T_order p + noise` where `a=[1,.65,.17,.025]` and `b=[0,.16,.78,.98]`. The internal reference `r = Rz(.13) diag(exp(-24ms/[25,75,180]ms)) p + noise`. Hence a single observed **late** return is isotropically distributed (and label-blind), but the paired relation `r->y` varies with the route. Reading the *full* response also reveals the early direct copy of the outbound packet, so the internal reference is no longer essential.

Train two regularized least-squares forward operators on labeled AB/BA regimes using class-specific paired examples. At test time classify by which learned map best accounts for an observed reference and echo. A return-only attacker fits full-covariance Gaussian distributions per class, not only means. An orientation-blind packet and a helpful fixed packet test whether a rich variable waveform is really necessary.

### Nine declared mechanism checks

| Gate | Required check | Reason |
| --- | --- | --- |
| G1a | `||CBA-CAB||_F > 0.5`, same-axis commute <1e-12 | Separate ordered composition from mere path length |
| G1b | abstract commutator defect >0.2; trace stable under conjugation | Real group-loop mathematics, not rebranding a residual |
| G2 | late matched held-out accuracy >.90 and at least +.28 vs late-return-only and reference-shuffle | Dependence on a paired self-reference |
| G3a | early reference+echo decoding within .08 of chance | Delay control, same per-trial information budget |
| G3b | full-echo-only decoding >.90 | **Attacker**: local direct early echo can replace private reference |
| G4 | chosen order-blind fixed packet <.58; informative fixed packet >.90 | Explicit boundary of rich waveform claim |
| G5 | final post-switch online MSE <15% of frozen MSE | Adaptive forward model, no claim of unique neurobiological rule |
| G6 | worlds observationally identical; independent emission flip changes only causal world | Counterexample to naive spike-triggered causal inference |
| G7 | same query changes after a prior write; gain-zero control unchanged | Resident dynamics genuinely stateful |

Important: These are **mechanism checks on a designed fixture**, not statistically calibrated confirmatory predictions for unseen real systems. The thresholds are chosen to catch coding/design mistakes, not to establish a new scientific effect.

## Geometry naming discipline

- `T_ext - T_cable` is **model mismatch**, not a commutator: it subtracts two operators with compatible input/output coordinates.
- `AB-BA` is a **commutator of two matrices** in a chosen common coordinate system, when the algebra is literally `AB-BA`. For nonlinear mappings the same expression need not be meaningful as a simple matrix difference.
- `B^-1 A^-1 B A` is a **group commutator**, a four-segment loop in an abstract group of invertible transformations. At fixed node state this experiment computes it exactly.
- A genuine Wilson loop generally refers to the trace of holonomy from specified parallel transport over a *closed path with a connection*. A circuit's return path by itself does not supply that geometric structure. Coordinate-basis invariance of `Tr(H)` in our artificial group is a consistency check, not a measured cortical Wilson loop.
- `MovingProblem` studies oriented eigenvector holonomy in parameter space. That is mathematically distinct from the ordered neural-transfer products here.

## Causal caveat

Natural observations can be `(outgoing z, incoming Tz+weather)` under *both* a driven-return model and a common-cause model. That is observational non-identifiability, even when the outgoing waveform is very distinctive.

To separate these toy models, hold `z` and weather fixed while externally reversing the output `x`. Only the driven-return model changes its echo. This assumes we **can intervene** on emission independently without perturbing `z` or the background; a real experiment must earn this assumption and measure side effects.

## Adaptive-model caveat

`M_{t+1} = M_t + eta * (y-M_t r) r^T/(c+||r||²)` is normalized LMS, a known observer update. It learns a **forward** echo predictor `M r`. It is *not* backpropagation and is not an inverse of the outside loop. Low prediction MSE on the frozen toy does not mean the original three node states were uniquely recovered. A non-spatial digital state-space predictor is a direct competitor.

## Follow-up gates worth running

1. Replace ideal rotations with spike regeneration, noisy delay/burst transformations and probabilistic release; quantify *how much information actually survives hop 1, 2, 3*.
2. Make outbound firing endogenous, then probe with randomized sparse interventions to distinguish shared-input correlations from self-caused echoes.
3. Remove labeled AB/BA teaching: learn online prediction error with local eligibility; test state switching and new circuit states without retuning on holdout.
4. Require task improvement: next-event prediction, change-point detection, or controlled probing with a matched digital reservoir/LMS/ Kalman filter and a one-compartment RC baseline.
5. Stress active reading: probes change receiver/network state, so compare read-induced corruption, refresh and conservation of total observation energy.
6. Geometry gate: learn physical cable parameters with constraints, count all states and plasticity parameters, test whether different time bins lead to cheaper representations than general state-space models.