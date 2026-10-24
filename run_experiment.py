"""Frozen v0 experiment: randomized held-out tests, negative controls, receipt.

Run: python run_experiment.py --out results/receipt.json
The scenario is synthetic and explicitly supplied; no biological claim.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from loop_holonomy import (
    BASE_ANGLES, T_LAGS_MS, StatefulLoop, accuracy, blind_packet,
    classify_with_reference, fixed_transfers, fit_conditional_transfer,
    group_commutator, online_adaptation, random_packet, return_only_classify,
    rotation, sample_dataset,
)

SEEDS = (41, 57, 73, 89, 97, 113, 127, 151, 173, 191, 211, 233)


def summarize(rows):
    values = np.asarray(rows, float)
    return {'mean': float(values.mean()), 'min': float(values.min()),
            'max': float(values.max()), 'std': float(values.std(ddof=1)),
            'by_seed': values.tolist()}


def algebra():
    a = rotation(0, BASE_ANGLES['A'])
    b = rotation(1, BASE_ANGLES['B'])
    ab, ba = fixed_transfers()
    h = group_commutator(a, b)
    qc = rotation(0, .3) @ rotation(2, .7)
    hc = group_commutator(qc @ a @ qc.T, qc @ b @ qc.T)
    sameaxis = rotation(0, -.91)
    null = blind_packet()
    return {
        'ordered_transfer_frobenius_gap': float(np.linalg.norm(ab - ba)),
        'abstract_group_commutator_defect': float(np.linalg.norm(h - np.eye(3))),
        'commuting_same_axis_order_gap': float(np.linalg.norm(a@sameaxis - sameaxis@a)),
        'blind_packet_order_gap': float(np.linalg.norm((ab - ba)@null)),
        'group_commutator_trace': float(np.trace(h)),
        'trace_error_under_global_basis_change': float(abs(np.trace(h) - np.trace(hc))),
        'group_commutator_matrix_covariance_error': float(np.linalg.norm(qc @ h @ qc.T - hc)),
        'blind_packet': null.tolist(),
    }


def causal_intervention():
    """Observe equivalence, then break it by an independent outgoing intervention."""
    rng = np.random.default_rng(227)
    t, _ = fixed_transfers()
    base = np.array([random_packet(rng) for _ in range(240)])
    common_weather = rng.normal(0., .2, base.shape)
    # Natural baseline: emitter packet x=world drive z. Both constructions
    # give *exactly the same observables*: (x, y)=(z, Tz + weather).
    natural_causal = base @ t.T + common_weather
    natural_confounded = base @ t.T + common_weather
    # Under a controlled sign-flip to the emitted packet, the independently
    # maintained world drive z stays fixed. Counterfactual sharing of weather
    # isolates causal dependency in this finite-dimensional fixture.
    flipped_causal = -base @ t.T + common_weather
    flipped_confounded = base @ t.T + common_weather
    return {
        'natural_observational_difference': float(np.max(np.abs(natural_causal - natural_confounded))),
        'mean_causal_intervention_effect': float(np.mean(np.linalg.norm(flipped_causal - natural_causal,axis=1))),
        'mean_confounded_intervention_effect': float(np.mean(np.linalg.norm(flipped_confounded - natural_confounded,axis=1))),
        'assumptions': 'World state and background held fixed under intervention; toy do-operation, not biological causal identification.'
    }


def state_history():
    c = StatefulLoop()
    q = np.array([.45, -.17, .88]); q /= np.linalg.norm(q)
    initial, _ = c.send(q, write=False)
    pulse = np.array([.19, .83, -.52]); pulse /= np.linalg.norm(pulse)
    c.send(pulse, write=True)
    after, _ = c.send(q, write=False)
    frozen = StatefulLoop(write_gain=0.)
    frozen_initial, _ = frozen.send(q, write=False)
    frozen.send(pulse, write=True)
    frozen_after, _ = frozen.send(q, write=False)
    return {
        'same_probe_response_changed_after_one_prior_probe': float(np.linalg.norm(after - initial)),
        'no_write_same_probe_change': float(np.linalg.norm(frozen_after-frozen_initial)),
        'state_after_prior_probe': dict(c.states)
    }


def run(seeds=SEEDS, train_n=360, test_n=200):
    tracked = {k: [] for k in (
        'early_paired', 'late_paired', 'late_return_only',
        'late_shuffled_reference', 'full_paired', 'full_return_only',
        'blind_fixed_probe', 'informative_fixed_probe')}
    bin_curves = {str(k): [] for k in T_LAGS_MS}
    for seed in seeds:
        train = sample_dataset(seed, train_n)
        test = sample_dataset(seed + 9000, test_n)
        for k, time in enumerate(T_LAGS_MS):
            mats = fit_conditional_transfer(train, (k,))
            bin_curves[str(time)].append(accuracy(classify_with_reference(test,mats,(k,)),test['label']))
        early, late, full = (0,), (3,), (0,1,2,3)
        m_early = fit_conditional_transfer(train, early)
        m_late = fit_conditional_transfer(train, late)
        m_full = fit_conditional_transfer(train, full)
        idx = np.random.default_rng(seed+5544).permutation(len(test['internal']))
        tracked['early_paired'].append(accuracy(classify_with_reference(test,m_early,early),test['label']))
        tracked['late_paired'].append(accuracy(classify_with_reference(test,m_late,late),test['label']))
        tracked['late_return_only'].append(accuracy(return_only_classify(train,test,late),test['label']))
        tracked['late_shuffled_reference'].append(accuracy(
            classify_with_reference(test,m_late,late,test['internal'][idx]),test['label']))
        tracked['full_paired'].append(accuracy(classify_with_reference(test,m_full,full),test['label']))
        tracked['full_return_only'].append(accuracy(return_only_classify(train,test,full),test['label']))
        for key, direction in [('blind_fixed_probe',blind_packet()),
                               ('informative_fixed_probe',np.array([0.,0.,1.]))]:
            fixed_train = sample_dataset(seed,train_n,fixed_probe=direction)
            fixed_test = sample_dataset(seed+9000,test_n,fixed_probe=direction)
            mat = fit_conditional_transfer(fixed_train,late)
            tracked[key].append(accuracy(classify_with_reference(fixed_test,mat,late),fixed_test['label']))
    a = algebra()
    metrics = {k:summarize(v) for k,v in tracked.items()}
    adaptation = [online_adaptation(seed) for seed in seeds]
    adapt_metrics={k: summarize([v[k] for v in adaptation]) for k in (
        'early_frozen_mse','early_adaptive_mse','late_frozen_mse','late_adaptive_mse')}
    checks = {
        'G1_noncommuting_order_and_commuting_control': a['ordered_transfer_frobenius_gap']>.5 and a['commuting_same_axis_order_gap']<1.e-12,
        'G1_basis_invariant_abstract_group_commutator_trace': a['abstract_group_commutator_defect']>.2 and a['trace_error_under_global_basis_change']<1.e-12,
        'G2_heldout_late_paired_observer_beats_controls': metrics['late_paired']['mean']>.90 and metrics['late_paired']['mean']-max(metrics['late_return_only']['mean'],metrics['late_shuffled_reference']['mean'])>.28,
        'G3_no_short_latency_order_signal': abs(metrics['early_paired']['mean']-.5)<.08,
        'G3_full_echo_provides_its_own_reference': metrics['full_return_only']['mean']>.90,
        'G4_some_fixed_packets_are_blind_others_are_not': metrics['blind_fixed_probe']['mean']<.58 and metrics['informative_fixed_probe']['mean']>.9,
        'G5_online_predictor_recovers_after_switch': adapt_metrics['late_adaptive_mse']['mean'] < .15*adapt_metrics['late_frozen_mse']['mean'],
    }
    c = causal_intervention()
    checks['G6_observation_does_not_identify_cause_but_intervention_does'] = (
        c['natural_observational_difference']==0 and c['mean_causal_intervention_effect']>1.8 and c['mean_confounded_intervention_effect']==0)
    sh = state_history()
    checks['G7_prior_probe_changes_next_operator'] = sh['same_probe_response_changed_after_one_prior_probe']>.005 and sh['no_write_same_probe_change']==0
    return {'experiment':'LoopHolonomyNeuron v0 / synthetic ordered probe laboratory',
            'status':'simulation not biological validation',
            'settings':{'seeds':list(seeds),'training_per_class':train_n,'test_per_class':test_n,
                        'noise_sigma':.065,'time_bins_ms':T_LAGS_MS,'features':'3D unit-norm packet / 3D decaying internal reference',
                        'training':'two ridge-fitted class-conditioned forward maps; supervised network regime labels',
                        'model':'state-conditioned SO(3) rotations; delayed echo is an imposed kernel'},
            'algebra':a, 'accuracy':metrics,
            'accuracy_by_observation_window_ms':{k:summarize(v) for k,v in bin_curves.items()},
            'switch_adaptation':adapt_metrics,
            'single_switch_example':{'seed':seeds[0], 'frozen_mse':adaptation[0]['frozen_curve'],
                                     'adaptive_mse':adaptation[0]['adaptive_curve']},
            'causal_counterexample':c,
            'resident_state_write_test':sh,
            'predeclared_mechanism_checks':checks, 'all_checks_pass':bool(all(checks.values()))}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='results/receipt.json')
    args=ap.parse_args()
    output=run()
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('LoopHolonomyNeuron v0 / held-out synthetic checks')
    for k,v in output['predeclared_mechanism_checks'].items():
        print(('PASS' if v else 'FAIL'), k)
    print('PAIR late',round(output['accuracy']['late_paired']['mean'],4),
          'NO REF',round(output['accuracy']['late_return_only']['mean'],4))
    print('FULL ECHO without internal ref',round(output['accuracy']['full_return_only']['mean'],4))
    print('SWITCH late adaptive/frozen MSE',round(output['switch_adaptation']['late_adaptive_mse']['mean'],5),
          round(output['switch_adaptation']['late_frozen_mse']['mean'],5))
    print('WROTE',path)
    if not output['all_checks_pass']:
        raise SystemExit('One or more predeclared checks failed; see receipt')

if __name__=='__main__':
    main()
