# import whisperx
from contrario.transcribe import Configuration, TranscriberModel
from injector import Module, provider, singleton

from rich import print


class WhisperXTranscriberModel(TranscriberModel):
    def __init__(self, model: str, config: Configuration) -> None:
        super().__init__(model)
        self._config = config
        # self._model = whisperx.load_model(model, config.device, compute_type=config.compute_type)


class TranscriberFactory(Module):
    def __init__(self) -> None:
        super().__init__()

    @provider
    @singleton
    def provide_model(self, config: Configuration) -> TranscriberModel:
        model = WhisperXTranscriberModel(model=config.model, config=config)
        return model
