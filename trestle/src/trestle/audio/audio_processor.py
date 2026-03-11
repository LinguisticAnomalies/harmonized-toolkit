from pathlib import Path
import torchaudio

class AudioConvert:
    """
    Convert .mp3 audio to 16 kHZ WAV files

    Requirements:
        - torch == 2.8
        - torchaudio == 2.8 (with FFmpeg backend available)
    
    """
    def __init__(
            self,
            audio_root: Path,
            audio_out_root: Path,
            target_sr: int=16_000,
            source_format: str='mp3'):
        self.target_sr = target_sr
        self.audio_root = audio_root
        self.audio_out_root = audio_out_root
        self.target_format = 'wav'

        self.audio_out_root.mkdir(parents=True, exist_ok=True)
        self.audio_files = list(self.audio_root.rglob(f"*.{source_format}"))

        try:
            torchaudio.set_audio_backend("ffmpeg")
        except Exception:
            raise ValueError("Could not set torchaudio backend to 'ffmpeg'.")
    
    def convert_file(self):
        for raw_path in self.audio_files:
            out_file = raw_path.stem
            out_path = self.audio_out_root / out_file.with_suffix(f".{self.target_format}")
            
            try:
                waveform, sr = torchaudio.load(raw_path)

                # Downmix to mono if multi-channel
                if waveform.size(0) > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)  # (1, T)

                # Resample if needed
                if sr != self.target_sr:
                    waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)
                    sr = self.target_sr
                
                torchaudio.save(out_path, waveform, sr, format=self.target_format)
            except Exception as e:
                return False, f"failed {raw_path} -> {out_path}: {e.__class__.__name__}: {e}"