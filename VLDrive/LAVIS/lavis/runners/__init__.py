# The MORAI entry points use explicit modules and their verified optimizer loop.
# Keep the upstream model zoo / CARLA registration as the default behavior.
import os as _os
_morai_profile = _os.environ.get("LAVIS_IMPORT_PROFILE", "full") == "morai"
if _morai_profile:
    __all__ = []
else:
    """
     Copyright (c) 2022, salesforce.com, inc.
     All rights reserved.
     SPDX-License-Identifier: BSD-3-Clause
     For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
    """

    from lavis.runners.runner_base import RunnerBase
    from lavis.runners.runner_iter import RunnerIter

    __all__ = ["RunnerBase", "RunnerIter"]
