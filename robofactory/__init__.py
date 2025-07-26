import os
PACKAGE_DIR = os.path.dirname(__file__)
CONFIG_DIR = os.path.join(PACKAGE_DIR, 'configs')
ASSET_DIR = os.path.join(PACKAGE_DIR, 'assets')
DIR_MAP = {
    '${PACKAGE_DIR}': PACKAGE_DIR,
    '${ASSET_DIR}': ASSET_DIR,
    '${CONFIG_DIR}': CONFIG_DIR,
}

from .tasks import *
from .planner import *