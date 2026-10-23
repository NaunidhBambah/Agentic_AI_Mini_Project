"""Embedding functions for ChromaDB (MiniLM default, offline hashing fallback)."""
import hashlib
import math
import re
import shutil
from functools import lru_cache
from pathlib import Path

from . import config

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("""a an the is are was were be been to of in on for and or with by as at from
that this these those it its what why how when which who whom do does did can could should
would will i me my we you your he she they them our us about into than then so if not no
please explain tell give difference between vs""".split())


class SimpleHashEmbedding:
    DIM = 512

    @staticmethod
    def name():
        return "simple-hash"

    def _vec(self, text):
        tech = {t.lower() for t in re.findall(r"[A-Za-z0-9]+", text)
                if re.search(r"[A-Z].*[A-Z]|[a-z][A-Z]|\d", t)}
        words = [w for w in _WORD.findall(text.lower()) if w not in _STOP]
        words = [w[:-1] if len(w) > 4 and w.endswith("s") else w for w in words]
        feats = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
        vec = [0.0] * self.DIM
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            vec[h % self.DIM] += 3.0 if f in tech else (1.0 if "_" not in f else 0.5)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def __call__(self, input):  # noqa: A002
        return [self._vec(t) for t in input]

    def embed_documents(self, input):  # noqa: A002
        return self(input)

    def embed_query(self, input):  # noqa: A002
        return self(input)

    def is_legacy(self):
        return False

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(cfg):
        return SimpleHashEmbedding()

    def default_space(self):
        return "cosine"

    def supported_spaces(self):
        return ["cosine"]


# The MiniLM model is downloaded ONCE into the project's models/ folder and reused on every start
# (it also travels with the project when you copy the folder to another laptop).
MODEL_DIR = config.ROOT / "models" / "all-MiniLM-L6-v2"
_FILES = ["config.json", "model.onnx", "special_tokens_map.json", "tokenizer_config.json",
          "tokenizer.json", "vocab.txt"]
_HOME_CACHE = Path.home() / ".cache" / "chroma" / "onnx_models" / "all-MiniLM-L6-v2"


def _complete(folder: Path) -> bool:
    return all((folder / "onnx" / f).exists() for f in _FILES)


def model_cached() -> bool:
    """True when no download is needed (model already in models/ or in Chroma's home cache)."""
    return config.EMBEDDINGS != "default" or _complete(MODEL_DIR) or _complete(_HOME_CACHE)


def _prepare_model_dir():
    if not _complete(MODEL_DIR) and _complete(_HOME_CACHE):     # reuse an earlier download, no internet
        shutil.copytree(_HOME_CACHE / "onnx", MODEL_DIR / "onnx", dirs_exist_ok=True)
        print(f"[DoubtLoop] Copied the embedding model from {_HOME_CACHE} to {MODEL_DIR}")
    elif not _complete(MODEL_DIR):
        print(f"[DoubtLoop] Downloading the embedding model once (~80 MB) to {MODEL_DIR} ...")


def _minilm():
    """Chroma's default MiniLM, but with ONE model kept in memory.

    Chroma's own DefaultEmbeddingFunction creates a new model object on every call, so the 80 MB ONNX
    model was re-loaded from disk for every search (3-4 times per doubt). This keeps one loaded copy.
    The name stays "default", so indexes built earlier still open.
    """
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

    class _Model(ONNXMiniLM_L6_V2):
        DOWNLOAD_PATH = MODEL_DIR

    class CachedMiniLM(DefaultEmbeddingFunction):
        def __init__(self):
            self._model = _Model()

        def __call__(self, input):  # noqa: A002
            return self._model(input)

        @staticmethod
        def build_from_config(cfg):
            return get_embedding()[0]

    return CachedMiniLM()


@lru_cache(maxsize=1)
def get_embedding():
    if config.EMBEDDINGS == "default":
        try:
            _prepare_model_dir()
            ef = _minilm()
            ef(["warm up"])
            return ef, "minilm"
        except Exception as exc:
            print(f"[DoubtLoop] MiniLM embeddings unavailable ({exc}); using offline 'simple' embeddings.")
    return SimpleHashEmbedding(), "simple"


def thresholds():
    _, backend = get_embedding()
    if backend == "simple":
        return {"retrieval": 0.12, "ta": 0.55, "assignment": 0.50}
    return {"retrieval": config.MIN_RETRIEVAL_SIM, "ta": config.TA_MATCH_SIM,
            "assignment": config.ASSIGNMENT_MATCH_SIM}