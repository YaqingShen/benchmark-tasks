"""arena_assets_fix: load Isaac Lab Arena 0.3.1 library assets from NVIDIA's production bucket.

Arena 0.3.1 points its object library at the *staging* S3 bucket
(``isaaclab_arena/assets/nucleus.py``: ``ARENA_NUCLEUS_DIR = ISAACLAB_NUCLEUS_DIR.replace("...production",
"...staging")``, marked TODO). The Isaac Sim 6.1 Arena folder has since been removed from staging and lives only in
production, so every library asset (the station's fruit, bowls, mugs, bottles) fails to load with a 404.

Importing this module from a task file rewrites the library's asset paths from staging to production, once
``yam_arena.scene`` (and with it Arena's object library) is imported. It changes nothing on disk, needs no Isaac
imports of its own, and is a no-op where the paths already point at production.

    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import arena_assets_fix  # noqa: F401
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys

STAGING = "omniverse-content-staging"
PRODUCTION = "omniverse-content-production"
TARGET = "yam_arena.scene"
_done = False


def _to_production(path):
    if isinstance(path, str) and STAGING in path and "/IsaacLab/Arena/" in path:
        return path.replace(STAGING, PRODUCTION)
    return path


def patch_registry(registry_cls) -> int:
    """Rewrite every registered asset's ``usd_path``; returns how many changed."""
    changed = 0
    for asset in registry_cls()._components.values():
        old = getattr(asset, "usd_path", None)
        new = _to_production(old)
        if new is not old:
            asset.usd_path = new
            changed += 1
    return changed


def _patch(scene_module) -> None:
    global _done
    if _done:
        return
    registry_cls = getattr(scene_module, "AssetRegistry", None)
    if registry_cls is None:
        return
    n = patch_registry(registry_cls)
    _done = True
    print(f"[arena_assets_fix] {n} library asset paths moved from staging to production", file=sys.stderr)


class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self.inner = inner

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        _patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _Loader(spec.loader)
        return spec


def install() -> None:
    if TARGET in sys.modules:
        _patch(sys.modules[TARGET])
    elif not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


install()
