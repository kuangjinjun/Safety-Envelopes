from src.providers.lmstudio_provider import LMStudioProvider
import logging

log = logging.getLogger(__name__)

def get_client(provider_config):
    provider_type = provider_config.get('type')
    if not provider_type:
        raise ValueError(f"Provider config missing 'type' key: {provider_config}")

    log.info(f"Creating client for provider type: {provider_type}")

    if provider_type == "LMStudioProvider":
        return LMStudioProvider(provider_config)
    else:
        raise ValueError(f"Unknown provider type specified in config: {provider_type}")