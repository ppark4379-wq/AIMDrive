# The MORAI entry points use explicit modules and their verified optimizer loop.
# Keep the upstream model zoo / CARLA registration as the default behavior.
import os as _os
_morai_profile = _os.environ.get("LAVIS_IMPORT_PROFILE", "full") == "morai"
if _morai_profile:
    from pathlib import Path as _Path
    from lavis.common.registry import registry
    _library = _Path(__file__).resolve().parent
    registry.register_path("library_root", str(_library))
    registry.register_path("repo_root", str(_library.parent))
    registry.register_path("cache_root", str(_library.parent / ".runtime" / "cache"))
else:
    """
     Copyright (c) 2022, salesforce.com, inc.
     All rights reserved.
     SPDX-License-Identifier: BSD-3-Clause
     For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
    """

    import os
    import sys

    from omegaconf import OmegaConf

    from lavis.common.registry import registry

    from lavis.datasets.builders import *
    from lavis.models import *
    from lavis.processors import *
    from lavis.tasks import *


    root_dir = os.path.dirname(os.path.abspath(__file__))
    default_cfg = OmegaConf.load(os.path.join(root_dir, "configs/default.yaml"))

    registry.register_path("library_root", root_dir)
    repo_root = os.path.join(root_dir, "..")
    registry.register_path("repo_root", repo_root)
    cache_root = os.path.join(repo_root, default_cfg.env.cache_root)
    registry.register_path("cache_root", cache_root)

    registry.register("MAX_INT", sys.maxsize)
    registry.register("SPLIT_NAMES", ["train", "val", "test"])
