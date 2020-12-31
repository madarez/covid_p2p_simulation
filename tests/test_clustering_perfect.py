import unittest
import numpy as np
import covid19sim.frozen.clustering.perfect as clu
from tests.helpers import (
    MessagesContext,
    chronological_key
)


class PerfectClusteringTests(unittest.TestCase):
    # here, we will run random encounter scenarios and verify that cluster are always homogenous

    def setUp(self):
        self.context = MessagesContext()

    def test_linear_saturation_history(self):
        n_human = 200
        n_encounters = 15
        n_exposures = 50
        assert n_exposures <= n_human
        max_timestamp = 30
        curated_messages = []
        for _ in range(n_exposures):
            curated_messages.extend(
                self.context.generate_linear_saturation_risk_messages(
                    max_timestamp=max_timestamp,
                    exposure_timestamp=np.random.randint(max_timestamp),
                    n_encounters=n_encounters,
                    init_risk_level=0,
                    final_risk_level=15
                )
            )
        for _ in range(n_human - n_exposures):
            curated_messages.extend(
                self.context.generate_linear_saturation_risk_messages(
                    max_timestamp=max_timestamp,
                    exposure_timestamp=np.inf,
                    n_encounters=n_encounters,
                    init_risk_level=0,
                    final_risk_level=15
                )
            )
        curated_messages = sorted(curated_messages, key=chronological_key)
        cluster_manager = clu.PerfectClusterManager(
            max_history_ticks_offset=max_timestamp)
        cluster_manager.add_messages(curated_messages)
        for cluster in cluster_manager.clusters:
            self.assertTrue(cluster._is_homogenous())

    def test_random_history(self):
        n_human = 200
        n_encounters = 15
        n_messages = 15
        assert n_encounters <= n_messages
        max_timestamp = 30
        curated_messages = []

        for _ in range(n_human):
            curated_messages.extend(
                self.context.generate_random_messages(
                    max_timestamp=max_timestamp,
                    n_encounters=n_encounters,
                    n_messages=n_messages,
                    exposure_timestamp=np.random.randint(max_timestamp),
                )
            )
        curated_messages = sorted(curated_messages, key=chronological_key)
        cluster_manager = clu.PerfectClusterManager(
            max_history_ticks_offset=max_timestamp)
        cluster_manager.add_messages(curated_messages)
        for cluster in cluster_manager.clusters:
            self.assertTrue(cluster._is_homogenous())


if __name__ == "__main__":
    unittest.main()
