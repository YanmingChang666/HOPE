"""Installation script for the ``whole_body_tracking`` Isaac Lab extension."""

import os

import toml
from setuptools import setup

# Read the extension metadata (single source of truth for version / author / description).
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum runtime dependencies (Isaac Lab itself is provided by the base install).
INSTALL_REQUIRES = [
    "psutil",
    "onnx",
    "onnxscript",
    "pyyaml",
    # This fork targets Isaac Lab 2.1.0, whose isaaclab_rl pins rsl-rl-lib==2.3.1 (2.x tuple/tensor
    # obs API). rsl-rl-lib 3.x switched to a dict obs API + required cfg["obs_groups"], which is
    # INCOMPATIBLE with IsaacLab 2.1.0's RslRlVecEnvWrapper (it returns a tuple), so pin <3 to keep
    # `pip install -e` from upgrading rsl_rl and breaking export/play/train. On the 2.x line the
    # HOPEOnPolicyRunner._prepare_logging_writer override and obs_groups shim are simply unused
    # (training then logs via rsl_rl's default TensorBoard writer). 2.2.4 also works.
    "rsl-rl-lib>=2.2.1,<3",
]

setup(
    name="whole_body_tracking",
    packages=["whole_body_tracking"],
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="Apache-2.0",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 4.0.0",
    ],
    zip_safe=False,
)
