import csv
import os
import re
from glob import glob
from tqdm import tqdm
import textgrid
import jsonlines
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
            text = re.sub(pattern, replacement, text)
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
            file_name, _ = os.path.splitext(os.path.basename(cha_file))
            output_file_path = os.path.join(
                    self.data_loc['text_output_path'], f"{str(file_name)}.jsonl")
            audio_file = self.data_loc.get('audio_input_path', '') and \
                    os.path.join(self.data_loc['audio_input_path'], f"{file_name}{audio_type}") or ""
            with open(cha_file, encoding='utf-8') as file_content:
                all_tran = file_content.read()
                if content_mark:
                    # no such content
                    try:
                        all_tran = re.search(
                            content_mark, all_tran, re.DOTALL).group()
                    except AttributeError:
                        continue
                # for windows line breakers
                all_tran = re.sub(r"\r\n|\n|((\*|\%|\@)[A-Za-z]+\:)", r" \n\1", all_tran)
                all_sents = all_tran.split("\n")
                for each_sent in all_sents:
                    if re.match(rf"\*{speaker}:", each_sent):
                        start, end, new_sent = self.clean_text(each_sent, rf"\*{speaker}:")
                        if new_sent:
                            record = {
                                "start": start,
                                "end": end,
                                "text": new_sent,
                                "audio": audio_file}
                            with jsonlines.open(output_file_path, mode="a") as writer:
                                writer.write(record)


class TextGridProcessor:
    def __init__(self, data_loc, txt_patterns, files):
        self.data_loc = data_loc
        self.txt_patterns = txt_patterns
        self.files = files
    
    @staticmethod
    def get_key_by_value(search_dict, search_value):
        """
        Use next to get the first key where the value matches, or None if not found
        NOTE: assume that the value is unique

        :param search_dict: the dictionary to search
        :type search_dict: dict
        :param search_value: the value to search for the key
        :type search_value: str
        :return: the key, if not found, then return None
        :rtype: str/None
        """
        return next((key for key, value in search_dict.items() if \
                     value == search_value), None)
    
    def get_ccc_speakers(self):
        """
        Get speakers' name and ids from metadata

        :return: Speaker ids and speakers name
        :rtype: dict
        """
        ccc_spkrs = {}
        with open(self.data_loc["meta_file"], 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['corpus_name'] == "CCC" and row['role'] == "interviewee":
                    speaker_number = row['speaker_number']
                    speaker_name = row['name'].lower().strip()
                    ccc_spkrs[speaker_number] = speaker_name
        return ccc_spkrs


    def clean_text(self, text):
        """
        basic pre-processing for CCC transcripts
        :param text: the CCC transcript for pre-processing
        :type text: str
        """
        text = text.lower().strip()
        for pattern, replacement in self.txt_patterns.items():
            text = re.sub(pattern, replacement, text)
        text = text.strip()
        return text
    

    def process_tier(self, tg, spkrs, tg_file, spkrs_names, audio_file):
        """
        Common logic for processing tiers
        """
        for tier in tg:
            tier_name = tier.name.lower().strip() if spkrs else tier.name
            pid = self.get_key_by_value(spkrs, tier_name) if spkrs else None
            if spkrs and "transcript" not in tier_name and tier_name in spkrs_names and pid:
                output_file_path = os.path.join(
                    self.data_loc['text_output_path'], f"{str(pid)}.jsonl")
            elif not spkrs:
                file_name, _ = os.path.splitext(os.path.basename(tg_file))
                output_file_path = os.path.join(
                    self.data_loc['text_output_path'], f"{file_name}.jsonl")
            else:
                continue
            with jsonlines.open(output_file_path, mode="a") as writer:
                for item in tier:
                    if self.clean_text(item.mark):
                        record = {
                            "start": item.minTime*1000,
                            "end": item.maxTime*1000,
                            "text": self.clean_text(item.mark),
                            "audio": audio_file}
                        writer.write(record)
    

    def clean_textgrid(self):
        os.makedirs(self.data_loc['text_output_path'], exist_ok=True)
        spkrs = None
        # CCC speaker ids
        if "data_type" in self.data_loc and self.data_loc["data_type"].lower() == "ccc":
            spkrs = self.get_ccc_speakers()
        spkrs_names = set(spkrs.values()) if spkrs else set()
        audio_type = self.data_loc.get('audio_type', '')
        for tg_file in tqdm(self.files, desc="Processing TextGrid files"):
            try:
                tg = textgrid.TextGrid.fromFile(tg_file)
                file_name, _ = os.path.splitext(os.path.basename(tg_file))
                audio_file = self.data_loc.get('audio_input_path', '') and \
                    os.path.join(self.data_loc['audio_input_path'], f"{file_name}{audio_type}") or ""
                if os.path.exists(audio_file):
                    self.process_tier(tg, spkrs, tg_file, spkrs_names, audio_file)
            except ValueError:
                continue
    
class TextWrapperProcessor:
    def __init__(self, data_loc, txt_patterns):
        self.data_loc = data_loc
        self.txt_patterns = txt_patterns
        self.processor = None
        self.files = []
        self.get_files()

    def get_files(self):
        if self.data_loc['format'] == '.cha':
            self.processor = ChaProcessor(
                data_loc=self.data_loc,
                txt_patterns=self.txt_patterns,
                files=self.files)
        elif self.data_loc['format'] == '.TextGrid':
            self.processor = TextGridProcessor(
                data_loc=self.data_loc,
                txt_patterns=self.txt_patterns,
                files=self.files)
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

        for utter_file in self.text_files:
            file_name, _ = os.path.splitext(os.path.basename(utter_file))
            with jsonlines.open(utter_file, 'r') as jsonl_f:
                utters = [obj for obj in jsonl_f]

                for i, record in enumerate(utters):
                    new_file_name = f"{file_name}_{i}.{self.AUDIO_FORMAT}"
                    new_file_path = os.path.join(
                        self.data_loc['audio_output_path'], new_file_name)
                    self.resample_and_slide(new_file_path, record)
                    metadata.append([new_file_name, record['text']])

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
            print(f"Audio file not found: {audio_file}")
        except pydub.exceptions.CouldntDecodeError:
            print(pydub.utils.mediainfo(audio_file))