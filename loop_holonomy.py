"""LoopHolonomyNeuron: abstract multi-hop state-bearing probe experiment.

SO(3) packet rotations are a convenient, exactly auditable toy for ordered,
state-conditioned transfer. They are NOT a model of regenerated axonal action
potential waveforms, neural gauge curvature, or a biological Wilson loop.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

DIM = 3
BASE_ANGLES = {'A': 0.94, 'B': -1.03, 'C': 0.24}
AXES = {'A': 0, 'B': 1, 'C': 2}
T_LAGS_MS = (4, 12, 24, 45)
ECHO_DIRECT = np.array([1., .65, .17, .025])
ECHO_LOOP = np.array([0., .16, .78, .98])


def rotation(axis: int, angle: float) -> np.ndarray:
    """Right-handed rotation about axis x=0,y=1,z=2."""
    c, s = np.cos(angle), np.sin(angle)
    if axis == 0:
        return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
    if axis == 1:
        return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    if axis == 2:
        return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    raise ValueError('axis must be 0,1,2')


def random_packet(rng: np.random.Generator) -> np.ndarray:
    p = rng.normal(size=3)
    return p / np.linalg.norm(p)


def fixed_transfers() -> tuple[np.ndarray, np.ndarray]:
    """Rightmost matrix acts first. Both paths contain A, B, C exactly once."""
    a, b, c = (rotation(AXES[k], BASE_ANGLES[k]) for k in ('A', 'B', 'C'))
    return c @ b @ a, c @ a @ b


def group_commutator(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """H = B^-1 A^-1 B A, a genuine closed four-leg abstract operator loop.

    Do not confuse H with the mismatch of the two three-hop neural paths.
    """
    return np.linalg.inv(b) @ np.linalg.inv(a) @ b @ a


def blind_packet() -> np.ndarray:
    """A fixed one-dimensional probe direction that is blind to the order swap."""
    ab, ba = fixed_transfers()
    _, _, vh = np.linalg.svd(ab - ba)
    p = vh[-1]
    return p / np.linalg.norm(p)


def internal_reference(packet: np.ndarray, lag_ms: float = 24., noise: float = 0.,
                       rng: np.random.Generator | None = None) -> np.ndarray:
    """Toy parallel internal cable: one coordinate per decaying mode.

    Not a biophysical bAP conduction-time model. Invertible for finite lag.
    """
    rates = np.exp(-lag_ms / np.array([25., 75., 180.]))
    r = rotation(2, .13) @ (rates * packet)
    if noise:
        if rng is None:
            raise ValueError('rng required if noise nonzero')
        r = r + rng.normal(0., noise, 3)
    return r


@dataclass
class StatefulLoop:
    """Three packet operators with locally evolving scalar resident states.

    The same incoming packet can produce a different outgoing packet when these
    states change. Each node is a stand-in for a circuit element, not a full neuron.
    """
    states: dict[str, float] = field(default_factory=lambda: {k: 0. for k in AXES})
    retain: float = .91
    write_gain: float = .12
    angle_gain: float = .22

    def send(self, packet: np.ndarray, order: str = 'AB', *, write: bool = True,
             jitter: dict[str, float] | None = None) -> tuple[np.ndarray, list[dict]]:
        if order not in ('AB', 'BA'):
            raise ValueError('order must be AB or BA')
        p = np.asarray(packet, dtype=float).copy()
        if p.shape != (3,):
            raise ValueError('packet must be a three-dimensional vector')
        itinerary = ('A', 'B', 'C') if order == 'AB' else ('B', 'A', 'C')
        visits = []
        for node in itinerary:
            before = p.copy()
            state = self.states[node]
            theta = BASE_ANGLES[node] + self.angle_gain * state
            if jitter is not None:
                theta += jitter.get(node, 0.)
            mat = rotation(AXES[node], theta)
            p = mat @ p
            if write:
                self.states[node] = (self.retain * state +
                                     self.write_gain * np.tanh(before[AXES[node]]))
            visits.append({'node': node, 'angle': float(theta),
                           'state_before': float(state), 'incoming': before,
                           'outgoing': p.copy()})
        return p, visits


def echo_observation(packet: np.ndarray, loop_return: np.ndarray,
                     rng: np.random.Generator, sigma: float = .065) -> np.ndarray:
    """Four imposed response windows; early direct and delayed loop components.

    Early 4-ms readout deliberately contains no loop information. The 24/45-ms
    bins expose the output of the external chain; this is a designed fixture.
    """
    direct = rotation(2, .23) @ packet
    echo = ECHO_DIRECT[:, None] * direct + ECHO_LOOP[:, None] * loop_return
    return echo + rng.normal(0., sigma, echo.shape)


def sample_dataset(seed: int, n_per_class: int, *, sigma: float = .065,
                   evolving: bool = False, fixed_probe: np.ndarray | None = None,
                   common_jitter: float = .06) -> dict[str, np.ndarray]:
    """Class-balanced, isotropic emitted states, independent class labels.

    Each class draws its own independent initial loop state and nuisance; no
    hidden class flag is fed to the observer. An entire four-bin response per
    probe is returned, plus its internal reference.
    """
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(2), n_per_class)
    rng.shuffle(labels)
    internal, emitted, echoes, returns = [], [], [], []
    circuit = StatefulLoop(write_gain=.025 if evolving else .12)
    for label in labels:
        if not evolving:
            circuit.states = {k: rng.normal(0., .16) for k in AXES}
        x = random_packet(rng) if fixed_probe is None else np.asarray(fixed_probe, dtype=float).copy()
        jitter = {k: rng.normal(0., common_jitter) for k in AXES}
        y, _ = circuit.send(x, 'AB' if label == 0 else 'BA', write=evolving, jitter=jitter)
        echo = echo_observation(x, y, rng, sigma)
        emitted.append(x)
        internal.append(internal_reference(x, noise=.012, rng=rng))
        echoes.append(echo)
        returns.append(y)
    return {'label': labels, 'internal': np.array(internal),
            'emitted': np.array(emitted), 'echo': np.array(echoes),
            'return': np.array(returns)}


def fit_conditional_transfer(data: dict, bins: tuple[int, ...], ridge: float = .08) -> np.ndarray:
    """Fit two class-conditioned forward models by regularized least squares.

    Outputs shape (2, n_bins*3, 3). Training labels are supplied only here;
    evaluation uses the learned models and an internal reference.
    """
    ref = data['internal']
    obs = data['echo'][:, bins, :].reshape(len(ref), -1)
    mats = []
    for label in (0, 1):
        sel = data['label'] == label
        x, y = ref[sel], obs[sel]
        mats.append(np.linalg.solve(x.T @ x + ridge*np.eye(3), x.T @ y).T)
    return np.stack(mats)


def classify_with_reference(data: dict, matrices: np.ndarray, bins: tuple[int, ...],
                            reference: np.ndarray | None = None) -> np.ndarray:
    ref = data['internal'] if reference is None else reference
    target = data['echo'][:, bins, :].reshape(len(ref), -1)
    predictions = np.stack([ref @ mat.T for mat in matrices], axis=1)
    score = np.sum((target[:, None, :] - predictions) ** 2, axis=-1)
    return np.argmin(score, axis=1)


def accuracy(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(pred == actual))


def return_only_classify(train: dict, test: dict, bins: tuple[int, ...]) -> np.ndarray:
    """Fair return-only Gaussian discriminant with class mean AND covariance.

    The exact generator's orthogonal transfer and isotropic probes imply identical
    return distributions in the two regimes; finite-sample fitting can fluctuate.
    """
    z = train['echo'][:, bins, :].reshape(len(train['label']), -1)
    q = test['echo'][:, bins, :].reshape(len(test['label']), -1)
    scores = []
    for cls in (0, 1):
        subset = z[train['label'] == cls]
        mu = subset.mean(axis=0)
        cov = np.cov(subset.T) + .03 * np.eye(subset.shape[1])
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise ValueError('non-positive covariance')
        diff = q - mu
        scores.append(np.sum(diff * np.linalg.solve(cov, diff.T).T, axis=1) + logdet)
    return np.argmin(np.stack(scores, axis=1), axis=1)


def online_adaptation(seed: int = 97, initial: int = 260, after: int = 170,
                      rate: float = .18, sigma: float = .045) -> dict:
    """Fit return model on AB; switch to BA; compare adaptive vs frozen.

    Baseline estimator is precisely NLMS (outer-product update), not a novel
    biological learning algorithm. The evolving material is a toy write trace.
    """
    rng = np.random.default_rng(seed)
    model = StatefulLoop(write_gain=.055, retain=.87, angle_gain=.28)
    xs, ys, regimes = [], [], []
    for i in range(initial + after):
        x = random_packet(rng)
        y, _ = model.send(x, order='AB' if i < initial else 'BA')
        ys.append(y + rng.normal(0., sigma, 3))
        xs.append(internal_reference(x, noise=.007, rng=rng))
        regimes.append(0 if i < initial else 1)
    x, y = np.array(xs), np.array(ys)
    fit = np.linalg.solve(x[:initial].T @ x[:initial] + .08 * np.eye(3),
                          x[:initial].T @ y[:initial]).T
    frozen = fit.copy()
    adaptive = fit.copy()
    fixed_errors, online_errors = [], []
    for i in range(initial, initial + after):
        fixed_errors.append(float(np.linalg.norm(y[i] - frozen @ x[i])**2))
        online_errors.append(float(np.linalg.norm(y[i] - adaptive @ x[i])**2))
        residue = y[i] - adaptive @ x[i]
        adaptive += rate * np.outer(residue, x[i]) / (.12 + np.dot(x[i], x[i]))
    return {'early_frozen_mse': float(np.mean(fixed_errors[:25])),
            'early_adaptive_mse': float(np.mean(online_errors[:25])),
            'late_frozen_mse': float(np.mean(fixed_errors[-60:])),
            'late_adaptive_mse': float(np.mean(online_errors[-60:])),
            'frozen_curve': fixed_errors, 'adaptive_curve': online_errors,
            'initial_matrix': fit.tolist(), 'final_adaptive_matrix': adaptive.tolist()}
