# LoopHolonomyNeuron

**Small signal travels. Large state stays. The return tells you what the path did.**

[**Standalone paper-puncher lab (`index.html`)**](index.html) · [Frozen measured receipt](results/receipt.json) · [Experiment protocol and mathematical ledger](DESIGN.md)

A reproducible **toy operator laboratory** inspired by a specific question: what could a layer-5-like source learn if it sends a state-bearing event through several changing receivers and later observes a transformed return alongside an independently evolving internal reference?

The experiment is not a cortical simulation. It deliberately substitutes exactly auditable 3D packet transformations for axonal spikes / multi-neuron transmission. It does **not** show that real spike waveforms traverse several synapses as analog vectors, that the apical tuft computes holonomy, or that local biology learns an inverse model.

## The machine

```text
      outgoing state-bearing packet ψ (3 numbers)
                       |
                   NODE A/B
                       |        each operator depends on local resident state
                   NODE B/A        and a passing packet updates that state
                       |
                    NODE C
                       |
              delayed return at tuft  ----->  measured echo at 4,12,24,45 ms
                       |                               |
                       +---------- compare ------------+
                                   ^
         independent internal reference ψself(t) = D(t) ψout
                                   |
                           sender's past state
```

Three stages rotate a 3D packet around different axes. For fixed states the two external route operators are

- `T_AB = C @ B @ A` and
- `T_BA = C @ A @ B`.

Reversing A/B changes the result **only because their transformations do not commute**. Node A, B and C each also hold a scalar resident state, which slightly changes its operator and gets updated when a packet passes. Thus the *same* later packet can see a changed network.

The sender keeps an illustrative decaying three-mode cable reference. A late echo contains the transformed return but not a label announcing whether it followed AB or BA. We test whether a decoder can recover that hidden route from **paired internal reference and return**, rather than from the echo alone.

The four time bins come from **imposed known kernels**: the earliest contains only direct-response information; the later bins increasingly carry the loop response. Their latencies are illustrative model coordinates, **not measured neural propagation times**.

## Frozen v0: 12 independent seeds, held-out data

Two supervised ridge-fitted class-conditioned 3×3 transfer models; 360 training examples and 200 held-out examples **per class, per seed**. Distinct train/test random draws, isotropic emitted 3D unit vectors, noisy initial node states and angles, measured reference with noise, additive echo noise. No deep learning, BPTT, autograd, or teacher hidden state passed at test time.

| Held-out regime decoding | Mean accuracy | Interpretation |
| --- | ---: | --- |
| Early 4-ms response **with** private reference | **50.4%** | Negative control: deliberately no network return in the early bin |
| Late 45-ms response **with** private reference | **98.8%** | Transform order recoverable from matched emitted/returned signal |
| Late return **without** reference (Gaussian mean/covariance decoder) | **49.1%** | Orthogonal transforms of isotropic packets leave marginal distribution unchanged |
| Late return with a **shuffled** reference | **50.3%** | The correspondence between outgoing and returning packets matters |
| Full four-bin response + private reference | **99.1%** | Temporal readout works |
| **Full four-bin response, no private reference** | **98.4%** | **Important control:** early direct response already supplies a reference! |
| Fixed 1D packet, chosen along the order-blind axis | **50.7%** | Some packets cannot interrogate this order difference |
| Fixed 1D packet along another chosen axis | **99.0%** | **Critical limit:** a rich state-bearing waveform is not *always* necessary |

All results are classification of a **constructed two-regime world**. The full early+late echo is an attacker that removes the supposed necessity of a separate dendritic self-copy. The informative fixed probe removes the claim that waveform variation itself is the source of the success. Both remain in the project prominently.

### What changes across the temporal windows?

| Imposed observation time | Matched decoder, mean held-out accuracy |
| --- | ---: |
| 4 ms | 50.4% |
| 12 ms | 77.1% |
| 24 ms | 98.4% |
| 45 ms | 98.8% |

Those numbers partly reflect the imposed arrival kernel. They are a controllability/observability known-answer check, not a discovered biological integration law.

## Algebra: three different objects

`loop holonomy` is a useful research name, but the math has to be precise.

1. **Ordered route products:** `CBA` and `CAB` are two different three-hop paths; their Frobenius difference is `1.2300` for the default rotations. An order swap is *not by itself* a closed-loop holonomy measurement.
2. **Internal prediction error:** `epsilon = y_return - M @ r_internal`. This is an ordinary forward-model residual. It is **not** intrinsically a commutator, and learning `M` to predict a returning signal is not learning the inverse of the chain.
3. **Abstract closed group commutator:** `H = B^{-1} A^{-1} B A`. This is a true four-leg **mathematical** group loop, with `||H-I||_F = 1.2300` and `Tr(H)=2.2435`. A common orthogonal coordinate change conjugates `H`; its trace is invariant to floating-point tolerance. We have **not** built a neuronal connection on a fiber bundle or measured a physical Wilson loop.

Rotations about one common axis commute exactly and are the null. In `SO(3)` the group commutator has a fixed axis; an outgoing packet in that direction is blind to the AB/BA order difference. The chosen axis and measured blind-probe failure are recorded in the receipt.

## Can the receiver learn an expected return?

After fitting a 3×3 forward predictor on an AB regime, we **switch** the route to BA without telling the predictor. A normalized least-mean-squares (NLMS) outer-product rule is then allowed to adapt after each observed return. The underlying network's resident states continue to update.

At the end of the switch (final 60 probe events, mean across 12 seeds):

| Observer | Late prediction MSE |
| --- | ---: |
| Frozen AB predictor | **0.5021** |
| Online local predictor | **0.0085** |

The predictor learns an ordinary effective forward transfer. It has **not** identified each intervening neuron, inverted a unique network geometry, or earned superior performance over other adaptive filters. Reconstructing internal causes, in particular, would require additional observability assumptions and controls.

## Causality is harder than a stamped pulse

An outgoing state stamp does **not** remove endogenous-firing confounding by itself. We made two worlds with **identical passive observations**: one where output drives the return, and another where a common cause drives both. From passive pairs alone, the worlds are indistinguishable.

A controlled outgoing-packet flip while the external drive is held constant separates them: the causal return changes and the common-cause-only return does not. This is a known-answer **intervention fixture**, not a test of real causal identification in cortical loops. See [DESIGN.md](DESIGN.md).

## Quick start

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 python run_experiment.py --out results/receipt.json
```

Open `index.html` in a browser; no server, package, WebGL, CDN, or network access is needed. GitHub Pages has not been confirmed active: download the HTML file from the repository and open it locally to use the demo. Set packet to **Blind fixed packet**, try A→B vs B→A, then select **Informative fixed packet**. Click **Send a ping** several times to let the operator state and small forward predictor change. `Reset learning` restores both to the initial fixture. The browser lab is deterministic visualization of the structural mechanism, not the source of the held-out scores; the Python run generates the receipt.

## Genealogy

- [BrainAsInverseModelerV3](https://github.com/anttiluode/BrainAsInverseModelerV3): a synthetic state-dependent spike waveform carries predictive information beyond event time; did *not* model propagation across intervening synapses or a returning loop.
- [NotSoSimpleNeuron](https://github.com/anttiluode/NotSoSimpleNeuron): branch-local state changes the operator encountered by a subsequent event; passive linear morphology has an exactly equivalent modal realization.
- [SilentPing](https://github.com/anttiluode/SilentPing): a neutral probe reads a resident state; readout geometry and the probe's own write effect are load-bearing.
- [BrainAsInverseModelerV2](https://github.com/anttiluode/BrainAsInverseModelerV2): physical cable coordinates preserve some source history, but standard delay embeddings win the benchmark and hidden causes may be unidentifiable.
- [MovingProblem](https://github.com/anttiluode/MovingProblem): a *different* genuine holonomy test of eigenvector-frame orientation around a **parameter-space** loop; do not identify it with the neural return path.
- [SighImageSuper](https://github.com/anttiluode/SighImageSuper): self-probe prediction can become confounded when environmental activity correlates with the probe; a private perturbation can provide causal leverage under explicit assumptions.

## What remains open

The next meaningful experiment replaces the perfectly transmitted three-component packet by regenerated event times, bursts and synaptic-release dynamics; supplies no privileged AB/BA labels; learns a bounded observer using locally available information; and tests downstream prediction/control **against matched ordinary state-space system identification**. If it cannot beat or complement those baselines, that is the result.

**Summary:** ordered transformations exist; a paired ping/return exposes them; a forward predictor can adapt. Holonomy as a biological explanation, self-causal credit assignment, and a genuinely economical inverse model remain unproven.