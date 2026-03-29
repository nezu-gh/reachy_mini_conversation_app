#!/usr/bin/env python3
"""Smart-turn VAD server — predicts whether a user's speech turn is complete.

Uses an ONNX model with Whisper feature extraction to classify the last
8 seconds of audio as "complete" or "incomplete".  Runs as a lightweight
Flask server on the inference VM.

Setup:
    pip install flask onnxruntime transformers soundfile librosa numpy

    # Download or train the smart-turn ONNX model and place it at MODEL_PATH.
    # The model expects Whisper log-mel features (80 bins, 8s chunk) and
    # outputs a single sigmoid probability (>0.5 = turn complete).

Usage:
    python scripts/smart_turn_server.py                  # default port 7863
    SMART_TURN_PORT=7863 python scripts/smart_turn_server.py

Then set in the robot's .env:
    SMART_TURN_URL=http://192.168.178.155:7863
"""

import os
import io
import base64
import logging

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart-turn")

MODEL_PATH = os.environ.get(
    "SMART_TURN_MODEL",
    os.path.expanduser("~/models/smart-turn-v3/smart-turn-v3.1-gpu.onnx"),
)
PORT = int(os.environ.get("SMART_TURN_PORT", "7863"))
THRESHOLD = float(os.environ.get("SMART_TURN_THRESHOLD", "0.5"))
CHUNK_SECONDS = 8
TARGET_SR = 16000


def _load_model():
    """Load ONNX model and Whisper feature extractor."""
    import onnxruntime as ort
    from transformers import WhisperFeatureExtractor

    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    session = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=providers)
    extractor = WhisperFeatureExtractor(chunk_length=CHUNK_SECONDS)
    logger.info("Loaded model from %s (providers: %s)", MODEL_PATH, session.get_providers())
    return session, extractor


def _truncate(audio: np.ndarray, n_seconds: int = CHUNK_SECONDS) -> np.ndarray:
    max_samples = n_seconds * TARGET_SR
    if len(audio) > max_samples:
        return audio[-max_samples:]
    if len(audio) < max_samples:
        return np.pad(audio, (max_samples - len(audio), 0), mode="constant")
    return audio


def _predict(session, extractor, audio: np.ndarray) -> dict:
    audio = _truncate(audio)
    inputs = extractor(
        audio,
        sampling_rate=TARGET_SR,
        return_tensors="np",
        padding="max_length",
        max_length=CHUNK_SECONDS * TARGET_SR,
        truncation=True,
        do_normalize=True,
    )
    features = inputs.input_features.squeeze(0).astype(np.float32)
    features = np.expand_dims(features, axis=0)
    outputs = session.run(None, {"input_features": features})
    prob = float(outputs[0][0].item())
    prediction = 1 if prob > THRESHOLD else 0
    return {
        "prediction": prediction,
        "probability": prob,
        "status": "complete" if prediction == 1 else "incomplete",
    }


def create_app():
    from flask import Flask, request, jsonify

    session, extractor = _load_model()
    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "healthy", "model": os.path.basename(MODEL_PATH)})

    @app.route("/predict", methods=["POST"])
    def predict():
        import soundfile as sf
        import librosa

        data = request.get_json()
        if not data or "audio_base64" not in data:
            return jsonify({"error": "Provide audio_base64 field"}), 400

        try:
            raw = base64.b64decode(data["audio_base64"])
            audio, sr = sf.read(io.BytesIO(raw))
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sr != TARGET_SR:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
            audio = audio.astype(np.float32)
            peak = np.max(np.abs(audio))
            if peak > 1.0:
                audio /= peak
            result = _predict(session, extractor, audio)
            return jsonify(result)
        except Exception as e:
            logger.exception("Prediction failed")
            return jsonify({"error": str(e)}), 500

    @app.route("/predict_raw", methods=["POST"])
    def predict_raw():
        """Accept raw float32 PCM bytes (no WAV header)."""
        import librosa

        audio_data = request.data
        if not audio_data:
            return jsonify({"error": "No audio data"}), 400
        try:
            sr = request.args.get("sample_rate", TARGET_SR, type=int)
            audio = np.frombuffer(audio_data, dtype=np.float32)
            if sr != TARGET_SR:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
            peak = np.max(np.abs(audio))
            if peak > 1.0:
                audio /= peak
            result = _predict(session, extractor, audio)
            return jsonify(result)
        except Exception as e:
            logger.exception("Prediction failed")
            return jsonify({"error": str(e)}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    logger.info("Smart-turn VAD server starting on :%d (threshold=%.2f)", PORT, THRESHOLD)
    app.run(host="0.0.0.0", port=PORT, debug=False)
