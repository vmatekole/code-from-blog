import re
from abc import ABC
from typing import Union

import pandas as pd
from injector import inject
from pyannote.audio import Pipeline
from pydub import AudioSegment


class Configuration:
    def __init__(self, **kwargs: dict[str, Union[str, int]]):
        self._readonly = False

        for k, v in kwargs.items():
            setattr(self, k, v)
        self._readonly = True

    def __setattr__(self, __name: str, __v) -> None:
        if getattr(self, '_readonly', False):
            raise AttributeError('Transcriber configuration readonly')
        super().__setattr__(__name, __v)

    def __repr__(self) -> str:
        state = ''
        for k, v in self.__dict__.items():
            state += f'{k}: {v} '
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
        #     self._model.load_audio(audio_file)
        #     self._model.transcribe(audio_file)
        pass

    def split_audio(self, audio_file: str, seconds: int) -> None:
        sound = AudioSegment.from_file(audio_file, format='m4a')

        for i, chunk in enumerate(sound[:: seconds * 1000]):
            with open(f'{audio_file}-{i}.wav', 'wb') as f:
                chunk.export(f, format='wav')

    def diarize(self, audio_file: str):
        pipeline = Pipeline.from_pretrained(
            'pyannote/speaker-diarization',
            use_auth_token='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',
        )

        diarization = pipeline(audio_file)

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            print(f'start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}')

    @classmethod
    def convert_to_milliseconds(cls, time: str) -> int:
        hours, minutes, seconds = time.split(':')
        seconds, milliseconds = map(int, seconds.split(','))
        return (
            int(hours) * 3600 + int(minutes) * 60 + int(seconds) * 1000
        ) + milliseconds

    @classmethod
    def parse_timings(cls, line: str) -> tuple[int, int]:

        if '-->' in line:
            start_time, end_time = map(str.strip, line.split('-->'))

            return (
                TranscriberService.convert_to_milliseconds(start_time),
                TranscriberService.convert_to_milliseconds(end_time),
            )
        return 0, 0

    @classmethod
    def parse_speaker_texts(cls, line: str) -> tuple[str, str]:
        if re.search(r'SPEAKER_\d+:\s*(.*)', line):
            speaker, text = map(str.strip, line.split(':'))
            return speaker, text
        return '', ''

    def read_srt(self, srt_file: str) -> pd.DataFrame:
        subtitle_data = []

        with open(srt_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()

            start_end_timings = list(
                filter(
                    lambda x: x != (0, 0),
                    map(TranscriberService.parse_timings, lines),
                )
            )
            speaker_texts = list(
                filter(
                    lambda x: x != ('', ''),
                    map(TranscriberService.parse_speaker_texts, lines),
                )
            )
            subtitle_data = list(zip(start_end_timings, speaker_texts))

            merged_tuple_list = [
                t_start_end_timings + t_speaker_texts
                for t_start_end_timings, t_speaker_texts in subtitle_data
            ]

            return pd.DataFrame(
                merged_tuple_list, columns=['start_time', 'end_time', 'speaker', 'text']
            )

    @classmethod
    def write_transcript(cls, raw_transcripts: pd.DataFrame) -> list[str]:

        lines = []
        curr_speaker: str = ''
        for r in raw_transcripts.itertuples(index=False):

            if curr_speaker != r.speaker:
                lines.append(f'\n\n{r.speaker}\n\n')
                curr_speaker: str = r.speaker.strip()

            lines.append(f'{r.text} ')

        with open('./temp.txt', mode='w', encoding='utf-8') as file:
            file.writelines(l for l in lines)
        return lines
