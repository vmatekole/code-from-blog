from injector import Injector, singleton
from contrario.transcribe import (
    Configuration,
    TranscriberModel,
    TranscriberService,
)
from contrario.transcribers import TranscriberFactory, WhisperXTranscriberModel

from rich import print
import pandas as pd


def configure_transcriber(binder):
    config = Configuration(
        provider='whisperx', model='medium', device='cpu', compute_type='int8'
    )
    binder.bind(Configuration, to=config, scope=singleton)


if __name__ == '__main__':
    injector: Injector = Injector([configure_transcriber, TranscriberFactory()])
    service: TranscriberService = injector.get(TranscriberService)
    # service.split_audio('/Users/PI/Downloads/joscha_1.m4a', 3600)
    # service.diarize("/Users/PI/Downloads/joscha_1.m4a-0.wav")
    df: pd.DataFrame = service.read_srt(
        '/Users/PI/code/code-from-blog/code/transcriber/diarised_txts/333-andrej-karpathy-tesla-ai-self-driving-optimus-aliens-and-agi-lex-fridman-podcast_DIARY.srt'
    )
    print(df.head())
    text = TranscriberService.write_transcript(df)
