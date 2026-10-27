"""
JSON decoding for spatial index files and mesh manifests.

pysimdjson is an optional speedup (pip install cloud-volume[simdjson]).
When it is installed, every function here uses it. Otherwise they fall
back to orjson, which cloudvolume already depends on.

pysimdjson's lazy parser can list an object's keys, or read one value,
without decoding the whole document, which makes spatial index queries
faster. orjson has to decode the whole document.
"""
import numpy as np
import orjson

try:
  import simdjson
except ImportError:
  simdjson = None

loads = simdjson.loads if simdjson is not None else orjson.loads

class SelectiveParser:
  """
  Reads only the keys, or a single value, of JSON objects.
  With pysimdjson the rest of the document is never decoded;
  with orjson the whole document is.

  Reuse one instance across many documents. Not thread safe.
  """
  def __init__(self):
    self._parser = simdjson.Parser() if simdjson is not None else None

  def keys(self, content):
    """Return the keys of the JSON object in content."""
    if self._parser is None:
      return orjson.loads(content).keys()
    return self._parser.parse(content).keys()

  def get_int64_array(self, content, key):
    """
    Return the value at key in the JSON object in content as
    an int64 numpy array, or None if key is not present.
    """
    if self._parser is None:
      value = orjson.loads(content).get(key)
      if value is None:
        return None
      return np.array(value, dtype=np.int64)

    # The parsed object must be released before the parser is
    # reused, or pysimdjson raises a RuntimeError.
    obj = self._parser.parse(content)
    try:
      if key not in obj:
        return None
      return np.frombuffer(obj[key].as_buffer(of_type="i"), dtype=np.int64)
    finally:
      del obj
