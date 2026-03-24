from pathlib import Path
import torchaudio


def clip_audio_batch(job):
    """
    Process all clips for ONE audio file.
    start/end are in milliseconds.
    """
    src, clips = job
    records = []

    try:
        info = torchaudio.info(src)
        sr = info.sample_rate
        total_frames = info.num_frames
    except RuntimeError:
        return records

    for clip in clips:
        start_ms = clip["start"]
        end_ms = clip["end"]

        start_frame = int(start_ms * sr / 1000)
        num_frames = int((end_ms - start_ms) * sr / 1000)

        if (
            num_frames <= 0
            or start_frame < 0
            or start_frame + num_frames > total_frames
        ):
            continue

        try:
            waveform, _ = torchaudio.load(
                src,
                frame_offset=start_frame,
                num_frames=num_frames,
            )
        except RuntimeError:
            continue

        if waveform.numel() == 0:
            continue

        torchaudio.save(clip["clip_path"], waveform, sr)

        records.append({
            "clip_path": str(clip["clip_path"]),
            "pid": clip["pid"],
            "text": clip["text"],
            "source_audio": str(src),
        })

    return records