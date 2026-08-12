import os
from pathlib import Path
from cryptography.fernet import Fernet
from backend.app.core.config import settings

class EncryptionManager:
    def __init__(self, key_path: str = None):
        if key_path is None:
            key_path = str(settings.SECRET_KEY_PATH)
        self.key_path = key_path
        self.key = self._load_or_create_key(key_path)
        self.cipher = Fernet(self.key)

    def _load_or_create_key(self, key_path: str) -> bytes:
        if os.path.exists(key_path):
            with open(key_path, "rb") as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_path, "wb") as f:
                f.write(key)
            return key

    def encrypt_and_save(self, file_bytes: bytes, save_path: Path) -> Path:
        encrypted_data = self.cipher.encrypt(file_bytes)
        encrypted_path = save_path.with_suffix(".enc")
        with open(encrypted_path, "wb") as f:
            f.write(encrypted_data)
        return encrypted_path

    def decrypt_file(self, encrypted_path: Path) -> bytes:
        with open(encrypted_path, "rb") as f:
            encrypted_data = f.read()
        return self.cipher.decrypt(encrypted_data)
