import os 
import yaml
import logging
from typing import Dict, Any, List
logger = logging.getLogger(__name__)

CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__),'..', 'config'),'config.yaml')

def load_config():
    '''Load configuration from a YAML file.
    returns:
        dict: Configuration parameters
    '''
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f'Error loading configuration: {e}')
        return {}

def get_data_paths():
    config = load_config()
    return config.get('data_paths', {})

def get_columns():
    config = load_config()
    return config.get('columns', {})
    
def get_evaluation_config():
    config = load_config()
    return config.get('evaluation', {})

def get_deployment_config():
    config = load_config()
    return config.get('deployment', {})

def get_inference_config():
    config = load_config()
    return config.get('inference', {})

def get_logging_config():
    config = load_config()
    return config.get('logging', {})


def get_mlflow_config():
    config = load_config()
    return config.get('mlflow', {})

def get_environment_config():
    config = load_config()
    return config.get('environment', {})

def get_pipeline_config():
    config = load_config()
    return config.get('pipeline', {})