"""
ECG CNN Engine module (Alias & backward compatibility layer for ecg_engine.py).
"""
from backend.app.engines.ecg_engine import ECGCNNEngine, ecg_engine, ecg_cnn_engine

__all__ = ["ECGCNNEngine", "ecg_engine", "ecg_cnn_engine"]
