from abc import ABC
from typing import Any
from injector import inject

from pydub import AudioSegment

from pyannote.audio import Pipeline


class Configuration:
    def __init__(self, **kwargs):
        self._readonly = False

        for k, v in kwargs.items():
            setattr(self, k, v)
        self._readonly = True

    def __setattr__(self, __name: str, __v: Any) -> None:
        if getattr(self, "_readonly", False):
            raise AttributeError("Transcriber configuration readonly")
        super().__setattr__(__name, __v)

    def __repr__(self) -> str:
        state = ""
        for k, v in self.__dict__.items():
            state += f"{k}: {v} "
        return state


class TranscriberModel(ABC):
    def __init__(self, model: str) -> None:
        self._model = model

        super().__init__()

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model


class TranscriberService:
    @inject
    def __init__(self, model: TranscriberModel):
        self._model = model

    def transcribe(self, audio_file: str) -> None:
        self._model.load_audio(audio_file)
        self._model.transcribe(audio_file)

    def split_audio(self, audio_file: str, seconds: int) -> None:
        sound = AudioSegment.from_file(audio_file, format="m4a")

        for i, chunk in enumerate(sound[:: seconds * 1000]):
            with open(f"{audio_file}-{i}.wav", "wb") as f:
                chunk.export(f, format="wav")

    def diarize(self, audio_file: str):
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token="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",
        )

        diarization = pipeline(audio_file)

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
