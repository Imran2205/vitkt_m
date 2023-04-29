from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.vittrack_baseline.config import cfg, update_config_from_file
from vot_path import base_path

def parameters(yaml_name: str):
    params = TrackerParams()
    prj_dir = base_path
    save_dir = base_path
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/vittrack_baseline_score/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    params.checkpoint = os.path.join(save_dir, "checkpoints/train/vittrack_baseline_score/%s/VITTRACK_BASELINE_SCORE_ep%04d.pth.tar" %
                                     (yaml_name, cfg.TEST.EPOCH))

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
