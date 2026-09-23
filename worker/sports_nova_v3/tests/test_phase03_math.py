import unittest
import numpy as np
from worker.sports_nova_v3.distributions import sample_beta_binomial, sample_dirichlet_multinomial, sample_negative_binomial
from worker.sports_nova_v3.tests.test_v3_contracts import fixture
from worker.sports_nova_v3.allocation import allocate_opportunities

class Phase03MathTests(unittest.TestCase):
    def test_distribution_counts_are_nonnegative(self):
        rng = np.random.default_rng(2)
        self.assertGreaterEqual(sample_negative_binomial(5, 8, rng), 0)
        self.assertLessEqual(sample_beta_binomial(10, 3, 2, rng), 10)
        counts = sample_dirichlet_multinomial(17, np.array([.2, .3, .5]), 20, rng)
        self.assertEqual(int(counts.sum()), 17)
        self.assertTrue(np.all(counts >= 0))

    def test_allocation_conserves_residual_usage(self):
        state = fixture()
        rng = np.random.default_rng(4)
        result = allocate_opportunities(state, state.home.team_id, 11, 9, rng)
        self.assertEqual(sum(result.target_counts.values()) + result.untargeted, 11)
        self.assertEqual(sum(result.carry_counts.values()) + result.residual_carries, 9)

if __name__ == "__main__":
    unittest.main()
