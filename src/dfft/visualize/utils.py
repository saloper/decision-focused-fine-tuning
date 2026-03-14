import os
import re
from typing import List

def get_experiment_dirs(model_name: str, dataset_name: str, experiments_root: str = "results\\experiments", sort_by_time: bool = True) -> List[str]:
    """
    Returns a list of experiment run directories for a given model and dataset, sorted by timestamp in the folder name.
    Args:
        model_name: e.g. 'moment-base', 'moment-lora'
        dataset_name: e.g. 'NEMA'
        experiments_root: root experiments directory
        sort_by_time: if True, sorts by timestamp in folder name
    Returns:
        List of absolute paths to experiment run directories
    """
    base_dir = os.path.join(experiments_root, model_name, dataset_name)
    if not os.path.exists(base_dir):
        return []
    # Folders are named like 20260314_044216_lr0.001_bs32
    exp_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if sort_by_time:
        def extract_time(d):
            m = re.match(r"(\d{8}_\d{6})", os.path.basename(d))
            return m.group(1) if m else ""
        exp_dirs = sorted(exp_dirs, key=extract_time)
    return exp_dirs


def get_latest_experiment_dir(model_name: str, dataset_name: str, experiments_root: str = "results\\experiments") -> str:
    dirs = get_experiment_dirs(model_name, dataset_name, experiments_root, sort_by_time=True)
    return dirs[-1] if dirs else None
