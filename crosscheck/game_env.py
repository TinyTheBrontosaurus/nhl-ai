from loguru import logger
import retro

class _Singleton:
    genv = None

def get_genv():
    """
    Get or create environment for this process. This is created once per process to save CPU
    :return: The environment
    """
    if _Singleton.genv is None:
        logger.debug("Creating Game env")
        _Singleton.genv = create_genv()
    return _Singleton.genv

def create_genv() -> retro.RetroEnv:
    """
    Create the environment.
    """
    env = retro.make('Nhl94-Genesis',
                     state=retro.State.NONE,
                     inttype=retro.data.Integrations.ALL)

    # # TODO: Wrap the env
    # if self._trainer_class.discretizer_class is not None:
    #     self.env = self._trainer_class.discretizer_class()(self.env)

    return env
