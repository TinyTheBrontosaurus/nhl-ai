import confuse

# Singleton config for the app
cc_config = confuse.LazyConfig('cross-check', __name__)

filename: str = ''

log_name: str = 'shootout-training'