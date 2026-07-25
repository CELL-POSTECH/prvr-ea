import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from prvr_compat import configure_cfg


def get_configs(dataset_name):
    base_name = dataset_name.replace('_clip', '')
    if base_name == 'tvr':
        import Configs.tvr as config
    elif base_name in ('act', 'msrvtt'):
        import Configs.act as config
    elif base_name == 'qvhighlight':
        import Configs.qvhighlight as config
    elif base_name == 'cha':
        import Configs.cha as config
    else:
        raise ValueError('Unsupported dataset: %s' % dataset_name)
    return configure_cfg(config.get_cfg_defaults(), dataset_name,
                         os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
