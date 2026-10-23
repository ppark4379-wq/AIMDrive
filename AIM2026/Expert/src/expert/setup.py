from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=[
        'expert_src',
        'expert_src.network',
        'expert_src.planning',
        'expert_src.obstacle',
        'expert_src.localization',
        'expert_src.config',
        'expert_src.control',
        'expert_src.path',
        'expert_src.path.lib',
        'expert_src.path.lib.class_defs',
        'expert_src.path.lib.save_load',
        'expert_src.path.lib.utils',
    ],
    package_dir={'expert_src': 'src'},
)

setup(**d)
