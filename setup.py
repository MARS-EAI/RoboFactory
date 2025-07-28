import argparse
import datetime
import sys
from datetime import date
from pathlib import Path

from setuptools import find_packages, setup

# update this version when a new official pypi release is made
__version__ = "0.0.0"


def get_package_version():
    return __version__


def get_nightly_version():
    today = date.today()
    now = datetime.datetime.now()
    timing = f"{now.hour:02d}{now.minute:02d}"
    return f"{today.year}.{today.month}.{today.day}.{timing}"


def get_python_version():
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def get_dependencies():
    install_requires = [
        "mani_skill"
    ]
    # NOTE (stao): until sapien is uploaded to pypi with mac support, users need to install manually below as so
    # f"sapien @ https://github.com/haosulab/SAPIEN/releases/download/nightly/sapien-3.0.0.dev20250303+291f6a77-{python_version}-{python_version}-macosx_12_0_universal2.whl;platform_system=='Darwin'"
    return install_requires


def parse_args(argv):
    parser = argparse.ArgumentParser(description="RoboFactory setup.py configuration")
    parser.add_argument(
        "--package_name",
        type=str,
        default="robofactory",
        choices=["robofactory", "robofactory-nightly"],
        help="the name of this output wheel. Should be either 'robofactory' or 'robofactory_nightly'",
    )
    return parser.parse_known_args(argv)


def main(argv):

    args, unknown = parse_args(argv)
    name = args.package_name
    is_nightly = name == "robofactory-nightly"

    this_directory = Path(__file__).parent
    long_description = (this_directory / "README.md").read_text(encoding="utf8")

    if is_nightly:
        version = get_nightly_version()
    else:
        version = get_package_version()

    sys.argv = [sys.argv[0]] + unknown
    print(sys.argv)
    setup(
        name=name,
        version=version,
        description="RoboFactory: Exploring Embodied Agent Collaboration with Compositional Constraints",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="RoboFactory contributors",
        url="https://github.com/MARS-EAI/RoboFactory",
        packages=find_packages(include=["robofactory*"]),
        python_requires=">=3.9",
        setup_requires=["setuptools>=62.3.0"],
        install_requires=get_dependencies(),
    )


if __name__ == "__main__":
    main(sys.argv[1:])