import os
from os.path import join



def prepare_dirs(config):
    trial = get_trial_num(config)
    path = join(config.logs_dir, config.model, config.classification_way, config.folder_name, trial)
    if not exists(path):
        os.makedirs(path)
    if config.flush:
        shutil.rmtree(path)
        if not exists(path):
            os.makedirs(path)


def save_config(config):
    trial = get_trial_num(config)
    path = join(config.logs_dir, config.model, config.classification_way, config.folder_name, trial)
    filename = f'hyper_params_fold{config.fold_num}.json'
    param_path = join(path, filename)

    if isfile(param_path):
        os.remove(param_path)
    print("[*] Param & Model Checkpoint Dir: {}".format(path))
    all_params = config.__dict__
    with open(param_path, 'w') as fp:
        json.dump(all_params, fp, indent=4, sort_keys=False)
            

def load_config(model_dir, fold_num):
    filename = f'hyper_params_fold{fold_num}.json'
    param_path = join(model_dir, filename)
    with open(param_path, 'r+') as f:
        params = json.load(f)
    return params


def get_trial_num(config):
    trial_num = config.trial
    error_msg = "[!] model number must be >= 1."
    assert trial_num > 0, error_msg
    return 'trial_' + str(trial_num)

