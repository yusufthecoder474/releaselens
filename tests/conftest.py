"""Point the app at temp DB/Chroma paths before any app module is
imported, so tests never touch the real releaselens.db or chroma_db.
pytest imports conftest before test modules, so these env vars win.
"""
import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="releaselens-test-")
os.environ["RELEASELENS_DB"] = os.path.join(_tmp, "test.db")
os.environ["RELEASELENS_CHROMA"] = os.path.join(_tmp, "chroma")
os.environ.pop("LLM_API_KEY", None)
