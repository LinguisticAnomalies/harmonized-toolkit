import csv
import os
import re
import json
from glob import glob
from tqdm import tqdm
import textgrid
import pydub


class ChaProcessor:
    def __init__(self, data_loc, txt_patterns, files):
        self.data_loc = data_loc
        self.txt_patterns = txt_patterns
        self.files = files


    def clean_text(self, text, speaker):
        """
        basic pre-processing for .cha transcripts
        :param text: the transcript for pre-processing
        :type text: str
        """
        pattern = re.search(r"(\d+_\d+)", text)
        start, end = 0, 0
        if pattern:
            start, end = map(int, pattern.group(1).split('_'))
            text = re.sub(pattern.group(1), '', text)
        text = re.sub(speaker, '', text)
        for pattern, replacement in self.txt_patterns.items():
            try:
                pattern = re.compile(pattern)
                text = re.sub(pattern, replacement, text)
            except re.error:
                pass
        return start, end, text.lower().strip()


    def clean_cha(self):
        """
        clean the .cha transcripts and save as .jsonl files to local folder
        """
        os.makedirs(self.data_loc['text_output_path'], exist_ok=True)
        audio_type = self.data_loc.get('audio_type', '')
        content_mark = self.data_loc.get("content", "")
        speaker = self.data_loc.get("speaker", "")
        for cha_file in tqdm(self.files, desc="Processing cha files"):
            records = []
            file_name, _ = os.path.splitext(os.path.basename(cha_file))
            file_name = str(file_name)
            output_file_path = os.path.join(
                    self.data_loc['text_output_path'], f"{file_name}.jsonl")
            if self.data_loc.get('audio_input_path', '') and \
                os.path.exists(os.path.join(self.data_loc['audio_input_path'], f"{file_name}{audio_type}")):
                audio_file = os.path.join(self.data_loc['audio_input_path'], f"{file_name}{audio_type}")
            else:
                audio_file = ""
            with open(cha_file, encoding='utf-8') as file_content:
                all_tran = file_content.read()
                if content_mark:
                    # no such content
                    try:
                        all_tran = re.search(
                            content_mark, all_tran, re.DOTALL).group()
                    except AttributeError:
                        all_tran = ""
                # for windows line breakers
                all_tran = re.sub(r"\n\s+", " ", all_tran)
                all_sents = all_tran.split("\n")
                for each_sent in all_sents:
                    if re.match(rf"\{speaker}:\s+", each_sent):
                        start, end, new_sent = self.clean_text(each_sent, rf"\{speaker}:")
                        if new_sent:
                            record = {
                                "start": start,
                                "end": end,
                                "text": new_sent,
                                "audio": audio_file}
                            records.append(record)
            with open(output_file_path, "w") as jsonl_file:
                for item in records:
                    json.dump(item, jsonl_file)
                    jsonl_file.write("\n")


class TextGridProcessor:
    def __init__(self, data_loc, txt_patterns, files, data_type):
        self.data_loc = data_loc
        self.txt_patterns = txt_patterns
        self.files = files
        self.data_type = data_type


    def clean_text(self, text):
        """
        basic pre-processing for CCC transcripts
        :param text: the CCC transcript for pre-processing
        :type text: str
        """
        for pattern, replacement in self.txt_patterns.items():
            text = re.sub(pattern, replacement, text)
        return text.lower().strip()


    def process_tier(self, tg, tg_file, audio_file):
        """
        Common logic for processing tiers
        """
        records = []
        file_name, _ = os.path.splitext(os.path.basename(tg_file))
        output_file_path = os.path.join(
            self.data_loc['text_output_path'], f"{file_name}.jsonl")
        for tier in tg:
            tier_name = tier.name.lower().strip()
            if file_name.lower() == tier_name:
                for item in tier:
                    if self.clean_text(item.mark):
                        record = {
                            "start": item.minTime*1000,
                            "end": item.maxTime*1000,
                            "text": self.clean_text(item.mark),
                            "audio": audio_file}
                        records.append(record)
        with open(output_file_path, "w") as jsonl_file:
            for item in records:
                json.dump(item, jsonl_file)
                jsonl_file.write("\n")


    def process_ccc_tier(self, tg, tg_file, speaker, audio_file):
        """
        special logic for preprocessing CCC corpus

        :param tg: the textgrid content
        :type tg: textgrid.textgrid.TextGrid
        :param tg_file: the path to the textgrid file
        :type tg_file: str
        :param speaker: the indicator (i.e., id, or particial name) of the speaker
        :type speaker: str
        :param audio_file: the location to the audio recording
        :type audio_file: str
        """
        records = []
        file_name, _ = os.path.splitext(os.path.basename(tg_file))
        output_file_path = os.path.join(
            self.data_loc['text_output_path'], f"{file_name}.jsonl")
        for tier in tg:
            tier_name = tier.name.lower().strip()
            if speaker.lower() in tier_name and \
                "transcript" not in tier_name:
                if tier_name.startswith("i "):
                    continue
                else:
                    for item in tier:
                        if self.clean_text(item.mark):
                            record = {
                                "start": item.minTime*1000,
                                "end": item.maxTime*1000,
                                "text": self.clean_text(item.mark),
                                "audio": audio_file}
                            records.append(record)
        with open(output_file_path, "w") as jsonl_file:
            for item in records:
                json.dump(item, jsonl_file)
                jsonl_file.write("\n")


    def clean_textgrid(self):
        os.makedirs(self.data_loc['text_output_path'], exist_ok=True)
        audio_type = self.data_loc.get('audio_type', '')
        self.files = [item for item in self.files if not item.split("/")[-1][0].isdigit()]
        for tg_file in tqdm(self.files, desc="Processing TextGrid files"):
            try:
                tg = textgrid.TextGrid.fromFile(tg_file)
                file_name, _ = os.path.splitext(os.path.basename(tg_file))
                audio_file = self.data_loc.get('audio_input_path', '') and \
                    os.path.join(self.data_loc['audio_input_path'], f"{file_name}{audio_type}")
                if not os.path.exists(audio_file):
                    audio_file = ""
                if self.data_type.lower() == "ccc":
                    speaker_name = file_name.split("_")[0]
                    self.process_ccc_tier(tg, tg_file, speaker_name, audio_file)
                else:
                    self.process_tier(tg, tg_file, audio_file)
            except ValueError:
                continue


class TextWrapperProcessor:
    def __init__(self, data_loc, txt_patterns):
        self.data_loc = data_loc
        self.txt_patterns = txt_patterns
        self.processor = None
        self.files = []
        self.data_type = self.data_loc.get('data_type', '')
        self.get_files()

    def get_files(self):
        if self.data_loc['format'] == '.cha':
            self.processor = ChaProcessor(
                data_loc=self.data_loc,
                txt_patterns=self.txt_patterns,
                files=self.files,
            )
        elif self.data_loc['format'] == '.TextGrid':
            self.processor = TextGridProcessor(
                data_loc=self.data_loc,
                txt_patterns=self.txt_patterns,
                files=self.files,
                data_type=self.data_type)
        else:
            raise ValueError(f"Unsupported format: {self.data_loc['format']}")
        self.processor.files = glob(
            os.path.join(self.data_loc['text_input_path'], f"*{self.data_loc['format']}"))


    def process(self):
        if isinstance(self.processor, ChaProcessor):
            self.processor.clean_cha()
        elif isinstance(self.processor, TextGridProcessor):
            self.processor.clean_textgrid()
        else:
            raise ValueError(f"Unsupported processor type: {type(self.processor)}")


class AudioProcessor:
    AUDIO_FORMAT = 'wav'
    def __init__(self, data_loc, sample_rate):
        self.data_loc = data_loc
        self.sample_rate = sample_rate
        self.text_files = []
        self.get_text_files()


    def get_text_files(self):
        pattern = os.path.join(self.data_loc['text_output_path'], "*.jsonl")
        self.text_files = glob(pattern)


    def process_audio(self):
        os.makedirs(self.data_loc['audio_output_path'], exist_ok=True)
        metadata = [['file_name', 'transcription']]

        for utter_file in tqdm(self.text_files, desc="Processing audio files"):
            file_name, _ = os.path.splitext(os.path.basename(utter_file))
            with open(utter_file, "r") as jsonl_f:
                utters = [json.loads(line) for line in jsonl_f]

                for i, record in enumerate(utters):
                    if record['audio'] and os.path.exists(record['audio']):
                        new_file_name = f"{file_name}_{i}.{self.AUDIO_FORMAT}"
                        new_file_path = os.path.join(
                            self.data_loc['audio_output_path'], new_file_name)
                        self.resample_and_slide(new_file_path, record)
                        metadata.append([new_file_name, record['text']])
                    else:
                        continue

        # Save metadata
        metadata_file = os.path.join(
            self.data_loc['audio_output_path'], "metadata.csv")
        with open(metadata_file, mode="w", newline="\n") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerows(metadata)

    def resample_and_slide(self, new_file_path, record):
        audio_file = record['audio']
        try:
            audio = pydub.AudioSegment.from_file(audio_file).set_frame_rate(self.sample_rate)
            sliced_audio = audio[record['start']:record['end']]
            sliced_audio.export(new_file_path, format=self.AUDIO_FORMAT)
        except FileNotFoundError:
            pass
        except pydub.exceptions.CouldntDecodeError:
            pass
