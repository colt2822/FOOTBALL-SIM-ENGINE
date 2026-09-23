import tempfile
import unittest
from worker.sports_nova_v3.tests.test_v3_contracts import fixture
from worker.sports_nova_v3.simulator import simulate_game
from worker.sports_nova_v3.config import MODEL_VERSION
from worker.sports_nova_v3.artifact_freeze import freeze_bundle, verify_frozen_artifact

class Phase09ArtifactTests(unittest.TestCase):
    def test_bundle_is_replayable_and_tamper_detected(self):
        samples = simulate_game(fixture(), 80, 22, MODEL_VERSION)
        with tempfile.TemporaryDirectory() as td:
            manifest = freeze_bundle(samples, samples.model, samples.runtime, td)
            self.assertTrue(verify_frozen_artifact(manifest))
            manifest["seed"] = 23
            self.assertFalse(verify_frozen_artifact(manifest))

if __name__ == "__main__":
    unittest.main()
