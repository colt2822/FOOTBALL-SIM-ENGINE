import unittest
from worker.sports_nova_v3.tests.test_v3_contracts import fixture
from worker.sports_nova_v3.simulator import simulate_game
from worker.sports_nova_v3.config import MODEL_VERSION
from worker.sports_nova_v3.joint_queries import Predicate, query

class Phase06QueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = fixture()
        cls.batch = simulate_game(cls.state, 250, 44, MODEL_VERSION)

    def test_marginal_joint_and_conditional_use_same_paths(self):
        a = Predicate(scope="team", entity_id="TEST_HOME", stat="score", op=">", threshold=0)
        b = Predicate(scope="game", entity_id="TEST_GAME", stat="total", op=">", threshold=0)
        pa = query(self.batch, (a,))
        pab = query(self.batch, (a, b))
        self.assertGreaterEqual(pa.probability, pab.probability)
        conditional = query(self.batch, (a,), (b,))
        self.assertTrue(0 <= conditional.probability <= 1)

    def test_zero_condition_and_unknown_identity_fail_closed(self):
        impossible = Predicate(scope="team", entity_id="TEST_HOME", stat="score", op="<", threshold=-1)
        given = Predicate(scope="team", entity_id="TEST_HOME", stat="score", op="<", threshold=-1)
        result = query(self.batch, (impossible,), (given,))
        self.assertIsNone(result.probability)
        with self.assertRaises(ValueError):
            query(self.batch, (Predicate(scope="team", entity_id="NOPE", stat="score", op=">", threshold=0),))

if __name__ == "__main__":
    unittest.main()
