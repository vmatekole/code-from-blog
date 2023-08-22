from injector import Injector, singleton
from contrario.transcribe import (
    Configuration,
    TranscriberModel,
    TranscriberService,
)
from contrario.transcribers import TranscriberFactory, WhisperXTranscriberModel

from rich import print


def configure_transcriber(binder):
    config = Configuration(
        provider="whisperx", model="medium", device="cpu", compute_type="int8"
    )
    binder.bind(Configuration, to=config, scope=singleton)


if __name__ == "__main__":
    injector: Injector = Injector([configure_transcriber, TranscriberFactory()])
    service: TranscriberService = injector.get(TranscriberService)
    # service.split_audio('/Users/PI/Downloads/joscha_1.m4a', 3600)
    service.diarize("/Users/PI/Downloads/joscha_1.m4a-0.wav")
