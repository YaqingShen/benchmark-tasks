import unittest
from arena_assets_fix import patch_registry


class AssetPathTests(unittest.TestCase):
    def test_unused_registry_assets_stay_lazy(self):
        class RemoteAssetPath:
            def __get__(self, instance, owner):
                raise AssertionError("unrelated remote asset was requested")
        class Microwave:
            usd_path = RemoteAssetPath()
        class StationAsset:
            usd_path = "https://omniverse-content-staging/IsaacLab/Arena/station.usd"
        class Registry:
            _components = {"microwave": Microwave, "station": StationAsset}
        self.assertEqual(patch_registry(Registry), 1)
        self.assertIn("omniverse-content-production", StationAsset.usd_path)
        self.assertIsInstance(vars(Microwave)["usd_path"], RemoteAssetPath)
