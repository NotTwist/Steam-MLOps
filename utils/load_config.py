import yaml


def load_from_config():
    config_location = 'options/config.yaml'
    with open(config_location, 'r') as f:
        config = yaml.safe_load(f)
    return config
