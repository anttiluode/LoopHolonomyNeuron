import unittest
import numpy as np

from loop_holonomy import (
    StatefulLoop, accuracy, blind_packet, classify_with_reference, echo_observation,
    fit_conditional_transfer, fixed_transfers, group_commutator, internal_reference,
    random_packet, return_only_classify, rotation, sample_dataset,
)
from run_experiment import algebra, causal_intervention, state_history


class TestAlgebra(unittest.TestCase):
    def test_rotations_proper_and_norm_preserving(self):
        for ax in range(3):
            m = rotation(ax, 1.1)
            np.testing.assert_allclose(m.T @ m, np.eye(3), atol=1e-14)
            self.assertAlmostEqual(np.linalg.det(m), 1.)
    def test_noncommutativity_and_commuting_control(self):
        a, b = fixed_transfers()
        self.assertGreater(np.linalg.norm(a-b), 1.)
        self.assertLess(algebra()['commuting_same_axis_order_gap'],1e-12)
    def test_abstract_holonomy_covariance_and_trace(self):
        alg = algebra()
        self.assertLess(alg['trace_error_under_global_basis_change'], 1e-12)
        self.assertLess(alg['group_commutator_matrix_covariance_error'], 1e-12)
        self.assertGreater(alg['abstract_group_commutator_defect'],.1)
    def test_blind_pulse_and_informative_pulse(self):
        ab,ba = fixed_transfers()
        self.assertLess(np.linalg.norm((ab-ba)@blind_packet()),1e-12)
        self.assertGreater(np.linalg.norm((ab-ba)@np.array([0.,0.,1.])),.1)
    def test_stateful_history_vs_no_write(self):
        r=state_history()
        self.assertGreater(r['same_probe_response_changed_after_one_prior_probe'],.005)
        self.assertEqual(r['no_write_same_probe_change'],0.)
    def test_invalid_packets_and_orders(self):
        obj=StatefulLoop()
        with self.assertRaises(ValueError):obj.send(np.zeros(2))
        with self.assertRaises(ValueError):obj.send(np.zeros(3),'BAD')
    def test_causal_observation_counterexample(self):
        c=causal_intervention()
        self.assertEqual(c['natural_observational_difference'],0.)
        self.assertGreater(c['mean_causal_intervention_effect'],1.9)
        self.assertEqual(c['mean_confounded_intervention_effect'],0.)
    def test_response_early_bin_is_independent_of_loop(self):
        rng=np.random.default_rng(7)
        x=random_packet(rng)
        r1=echo_observation(x,np.array([0.,0.,1.]),rng,sigma=0.)
        r2=echo_observation(x,np.array([1.,0.,0.]),rng,sigma=0.)
        np.testing.assert_equal(r1[0],r2[0])
        self.assertGreater(np.linalg.norm(r1[-1]-r2[-1]),.5)


class TestLearning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train=sample_dataset(41,200)
        cls.test=sample_dataset(9041,120)
    def test_pairing_has_predictive_information(self):
        bins=(3,)
        fit=fit_conditional_transfer(self.train,bins)
        paired=accuracy(classify_with_reference(self.test,fit,bins),self.test['label'])
        unpaired=accuracy(return_only_classify(self.train,self.test,bins),self.test['label'])
        self.assertGreater(paired,.90)
        self.assertGreater(paired-unpaired,.28)
    def test_early_bin_carries_no_label(self):
        bins=(0,)
        fit=fit_conditional_transfer(self.train,bins)
        score=accuracy(classify_with_reference(self.test,fit,bins),self.test['label'])
        self.assertLess(abs(score-.5),.13)
    def test_observed_whole_trace_contains_own_reference(self):
        bins=(0,1,2,3)
        score=accuracy(return_only_classify(self.train,self.test,bins),self.test['label'])
        self.assertGreater(score,.9)
    def test_inverse_internal_reference_is_possible(self):
        p=np.array([.4,.5,.6]);a=internal_reference(p,lag_ms=30)
        b=internal_reference(p,lag_ms=30)
        np.testing.assert_equal(a,b)

if __name__=='__main__':
    unittest.main()
