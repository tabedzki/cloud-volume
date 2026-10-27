import importlib
import sys

import numpy as np
import pytest

try:
  import simdjson as _simdjson
except ImportError:
  _simdjson = None

BACKENDS = ["orjson"]
if _simdjson is not None:
  BACKENDS.append("simdjson")

@pytest.fixture(params=BACKENDS)
def fastjson(request, monkeypatch):
  """fastjson imported with pysimdjson either hidden or available."""
  if request.param == "orjson":
    monkeypatch.setitem(sys.modules, "simdjson", None) # import raises ImportError
  import cloudvolume.fastjson as module
  module = importlib.reload(module)
  assert (module.simdjson is None) == (request.param == "orjson")
  yield module
  monkeypatch.undo()
  importlib.reload(module)

CONTENT = b'{"1": [0, 0, 0, 1, 1, 1], "22": [2, 3, 4, 5, 6, 7]}'

@pytest.mark.parametrize("content", [CONTENT, CONTENT.decode("utf8")])
def test_loads(fastjson, content):
  res = fastjson.loads(content)
  assert type(res) is dict
  assert res == {"1": [0, 0, 0, 1, 1, 1], "22": [2, 3, 4, 5, 6, 7]}

def test_loads_empty_object(fastjson):
  assert fastjson.loads(b"{}") == {}

def test_keys(fastjson):
  parser = fastjson.SelectiveParser()
  assert set(parser.keys(CONTENT)) == {"1", "22"}

def test_keys_empty_object(fastjson):
  parser = fastjson.SelectiveParser()
  assert len(list(parser.keys(b"{}"))) == 0

def test_get_int64_array(fastjson):
  parser = fastjson.SelectiveParser()
  res = parser.get_int64_array(CONTENT, "22")
  assert res.dtype == np.int64
  assert np.array_equal(res, [2, 3, 4, 5, 6, 7])

def test_get_int64_array_missing_key(fastjson):
  parser = fastjson.SelectiveParser()
  assert parser.get_int64_array(CONTENT, "3") is None
  assert parser.get_int64_array(b"{}", "1") is None

def test_get_int64_array_reuses_parser(fastjson):
  # pysimdjson raises if a parser is reused while a parsed
  # object is still alive; this must work across many documents.
  parser = fastjson.SelectiveParser()
  for _ in range(5):
    assert parser.get_int64_array(CONTENT, "1") is not None
    assert parser.get_int64_array(CONTENT, "3") is None
    assert len(list(parser.keys(CONTENT))) == 2
