from pytracking.evaluation.environment import EnvSettings
from vot_path import base_path
import os
def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = ''
    settings.lasot_path = ''
    settings.network_path = os.path.join(base_path,'checkpoints/networks')    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.oxuva_path = ''
    settings.result_plot_path = '/home/zj/tracking/LTMU_Expansion/LTMU-master/Stark_keepTrack/pytracking/result_plots/'
    settings.results_path = '/home/zj/tracking/LTMU_Expansion/LTMU-master/Stark_keepTrack/pytracking/tracking_results/'    # Where to store tracking results
    settings.segmentation_path = '/home/zj/tracking/LTMU_Expansion/LTMU-master/Stark_keepTrack/pytracking/segmentation_results/'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''

    return settings

