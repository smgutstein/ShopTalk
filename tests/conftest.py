"""Shared test stubs for optional runtime dependencies.

The unit tests exercise pure project code and should not require FAISS or
LangChain to be installed just to import modules under ``server``.
"""

import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if "faiss" not in sys.modules:
    fake_faiss = types.ModuleType("faiss")

    def _missing_read_index(path):
        raise RuntimeError("faiss.read_index was not monkeypatched for this test")

    fake_faiss.read_index = _missing_read_index
    sys.modules["faiss"] = fake_faiss


if "langchain_classic.schema" not in sys.modules:
    fake_langchain = types.ModuleType("langchain_classic")
    fake_schema = types.ModuleType("langchain_classic.schema")

    class _Message:
        def __init__(self, content):
            self.content = content

        def __eq__(self, other):
            return self.__class__ is other.__class__ and self.content == other.content

        def __repr__(self):
            return f"{self.__class__.__name__}(content={self.content!r})"

    class AIMessage(_Message):
        pass

    class HumanMessage(_Message):
        pass

    class SystemMessage(_Message):
        pass

    fake_schema.AIMessage = AIMessage
    fake_schema.HumanMessage = HumanMessage
    fake_schema.SystemMessage = SystemMessage
    fake_langchain.schema = fake_schema

    sys.modules["langchain_classic"] = fake_langchain
    sys.modules["langchain_classic.schema"] = fake_schema
