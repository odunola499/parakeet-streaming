import queue

import numpy as np
import time
import onnxruntime as ort
from transformers import WhisperFeatureExtractor
from huggingface_hub import hf_hub_download

import torch

from parakeet.engine.sequence import Sequence


SMART_TURN_V3_REPO = "pipecat-ai/smart-turn-v3"
filename = "smart-turn-v3.2-cpu.onnx"
VAD_FRAME_SAMPLES = 512
VAD_THRESHOLD = 0.5
N_SECONDS = 8


class TurnDetection:
    def __init__(
        self,
        sr: int = 16000,
        speech_start_ms: float = 100,
        silence_end_ms: float = 100,
        reset_after_ms: int = 1000,
    ):

        self.speech_start_dur = int(sr * (speech_start_ms / 1000.0))
        self.silence_end_dur = int(sr * (silence_end_ms / 1000.0))
        self.reset_after_dur = int(sr * (reset_after_ms / 1000.0))
        self.sr = sr

        self._build_ort_session()
        self._build_vad_session()

    def _build_ort_session(self):
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            chosen = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            chosen = ["CPUExecutionProvider"]

        onnx_path = hf_hub_download(repo_id=SMART_TURN_V3_REPO, filename=filename)
        self.ort_session = ort.InferenceSession(
            onnx_path, sess_options=so, providers=chosen
        )
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            "openai/whisper-tiny"
        )

    def _build_vad_session(self):
        torch.set_num_threads(1)
        self.vad, _utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )

    @torch.no_grad()
    def infer_vad(self, seq: Sequence, samples: np.ndarray) -> float:
        if samples.size == 0:
            return 0.0

        with seq.lock:
            if seq.vad_buf.size == 0:
                seq.vad_buf = samples
            else:
                seq.vad_buf = np.concatenate([seq.vad_buf, samples], axis=0)

            frame_count = seq.vad_buf.shape[0] // VAD_FRAME_SAMPLES
            if frame_count == 0:
                return 0.0
            frames = np.split(
                seq.vad_buf[: frame_count * VAD_FRAME_SAMPLES],
                frame_count,
            )
            seq.vad_buf = seq.vad_buf[frame_count * VAD_FRAME_SAMPLES :]

        probs = [self.vad(torch.from_numpy(frame), self.sr).item() for frame in frames]
        return float(np.mean(probs)) if probs else 0.0

    def infer_turn(self, array):
        inputs = self.feature_extractor(
            array,
            sampling_rate=self.sr,
            return_tensors="np",
            padding="max_length",
            max_length=N_SECONDS * self.sr,
            truncation=True,
            do_normalize=True,
        )["input_features"].astype(np.float32)

        outputs = self.ort_session.run(None, {"input_features": inputs})
        probability = outputs[0][0].item()
        prediction = 1 if probability > 0.5 else 0

        return {"prediction": prediction, "probability": probability}

    def __call__(self, seq: Sequence):
        try:
            chunk = seq.td_queue.get(timeout=0.01)
        except queue.Empty:
            with seq.lock:
                seq.last_state = None
            return

        chunk_size = chunk.shape[0]
        vad_prob = self.infer_vad(seq, chunk)
        is_speech = vad_prob >= VAD_THRESHOLD
        now = time.time()

        with seq.lock:
            if is_speech:
                seq.speech_run += chunk_size
                seq.silence_run = 0
            else:
                seq.silence_run += chunk_size
                seq.speech_run = 0

            if not seq.turn_active and seq.speech_run >= self.speech_start_dur:
                seq.turn_active = True
                seq.td_array[:] = 0.0
                seq.speech_run = 0
                seq.last_endpoint_check = 0
                if seq.last_state != "speech":
                    seq.turn_position = "start_of_utterance"
                    seq.last_state = "speech"

            if seq.turn_active:
                seq.td_array[:-chunk_size] = seq.td_array[chunk_size:]
                seq.td_array[-chunk_size:] = chunk

                if is_speech and seq.last_state != "speech":
                    seq.turn_position = "running"
                    seq.last_state = "speech"

                if (not is_speech) and seq.last_state != "silence":
                    seq.last_state = "silence"

            if (
                seq.silence_run >= self.silence_end_dur
                and (now - seq.last_endpoint_check) >= 0.1
            ):
                seq.last_endpoint_check = now

                pred = self.infer_turn(seq.td_array)
                if pred["prediction"] == 1:
                    seq.turn_position = "end_of_utterance"
                    seq.turn_active = False
                    seq.td_array[:] = 0.0
                    seq.speech_run = 0
                    seq.silence_run = 0
                    seq.last_state = None

                else:
                    seq.turn_position = "pause"

                if seq.silence_run >= self.reset_after_dur:
                    seq.turn_active = False
                    seq.td_array[:] = 0.0
                    seq.speech_run = 0
                    seq.silence_run = 0
                    seq.last_state = None
