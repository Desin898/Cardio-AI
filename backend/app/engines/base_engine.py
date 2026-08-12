from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseMLEngine(ABC):
    @abstractmethod
    def load_models(self) -> None:
        """Load required machine learning models and artifacts."""
        pass

    @abstractmethod
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run model inference on structured input data."""
        pass
