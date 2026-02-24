# auth_config.py
import os
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth


def get_authenticator():
    """
    Load user_config.yaml and construct a streamlit_authenticator.Authenticate instance.
    """
    config_path = os.path.join(os.path.dirname(__file__), "user_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
        # auto_hash=True by default: plain passwords in config will get hashed
    )
    return authenticator, config
