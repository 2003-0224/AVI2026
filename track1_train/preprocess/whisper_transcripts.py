import os
import sys
import logging
from pathlib import Path
import whisper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[sys.stdout, logging.FileHandler('whisper_transcription.log', encoding='utf-8')]
)
logger = logging.getLogger("WhisperTranscriber")


class BatchTranscriber:
    def __init__(self, model_name_or_path: str, audio_folder: str, output_folder: str, device: str = None):
        self.audio_folder = Path(audio_folder).resolve()
        self.output_folder = Path(output_folder).resolve()

        if device is None:
            if whisper.torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(whisper.torch.backends, "mps") and whisper.torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        try:
            self.model = whisper.load_model(model_name_or_path, device=self.device)
            logger.info(f"Model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            sys.exit(1)

    def transcribe_all(self):
        if not self.audio_folder.exists():
            logger.error(f"Input directory not found: {self.audio_folder}")
            return

        audio_files = list(self.audio_folder.rglob("*.wav"))
        total_files = len(audio_files)

        if total_files == 0:
            logger.warning("No .wav files found.")
            return

        success_count = 0
        failed_count = 0

        for index, audio_path in enumerate(audio_files, 1):
            try:
                relative_path = audio_path.relative_to(self.audio_folder)
                output_path = self.output_folder / relative_path.with_suffix(".txt")
                output_path.parent.mkdir(parents=True, exist_ok=True)

                logger.info(f"[{index}/{total_files}] Transcribing: {relative_path}")

                result = self.model.transcribe(
                    str(audio_path),
                    temperature=0.0,
                    best_of=1,
                    beam_size=1
                )

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(result['text'].strip())

                success_count += 1

            except Exception as e:
                logger.error(f"Failed {audio_path.name}: {e}")
                failed_count += 1

        logger.info(f"Done. Total: {total_files} | Success: {success_count} | Failed: {failed_count}")


if __name__ == "__main__":
    transcriber = BatchTranscriber(
        model_name_or_path="./tiny.pt",
        audio_folder="./audio",
        output_folder="./output"
    )
    transcriber.transcribe_all()