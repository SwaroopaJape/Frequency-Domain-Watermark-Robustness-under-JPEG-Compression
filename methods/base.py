from abc import ABC, abstractmethod
import numpy as np

# abstract base class for all watermarking methods
class Watermarker(ABC):
    
    # embed / extract operate on single-channel float32 images
    @abstractmethod
    def embed(self, image: np.ndarray, watermark: np.ndarray,alpha: float = 0.1,) -> np.ndarray:
        # paticular functions will overide this function it's own embedding function
        pass

    @abstractmethod
    def extract(self, watermarked: np.ndarray, original: np.ndarray, n_bits: int, alpha: float = 0.1,) -> np.ndarray:
        # particular functions will overide this function it's own extraction (non-blind extraction) function
        pass

    @property
    def name(self) -> str:
        # returns the class name
        return self.__class__.__name__
