#!/usr/bin/env python3
"""Restricted unpickler for PyTorch zip checkpoints.

The TinyStories checkpoint's data.pkl references exactly four globals
(verified with pickletools): collections.OrderedDict, torch.FloatStorage,
torch.BoolStorage, torch._utils._rebuild_tensor_v2. This loader whitelists
exactly those; anything else raises. No checkpoint code is ever executed:
storages are read as raw bytes from the zip and rebuilt as numpy arrays.
"""
import io
import pickle
import zipfile

import numpy as np

_DTYPES = {"float32": np.float32, "bool": np.bool_}


class _Storage:
    def __init__(self, raw: bytes, dtype: str):
        self.raw = raw
        self.dtype = dtype


class _RestrictedUnpickler(pickle.Unpickler):
    def __init__(self, file, zf: zipfile.ZipFile, prefix: str):
        super().__init__(file)
        self.zf = zf
        self.prefix = prefix

    def persistent_load(self, pid):
        # pid = ('storage', <storage class>, key, location, numel)
        kind, cls, key, _loc, _numel = pid
        if kind != "storage":
            raise ValueError(f"unexpected persistent id {pid!r}")
        name = getattr(cls, "__name__", str(cls))
        dtype = "float32" if "Float" in name else "bool"
        raw = self.zf.read(f"{self.prefix}/data/{key}")
        return _Storage(raw, dtype)

    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            from collections import OrderedDict
            return OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild_tensor_v2
        if module == "torch" and name in ("FloatStorage", "BoolStorage"):
            # Referenced as the storage class inside persistent ids; the
            # unpickler only needs a stand-in class object for identity.
            return type(name, (), {})
        raise ValueError(f"forbidden global in checkpoint: {module}.{name}")


def _rebuild_tensor_v2(storage: _Storage, offset, size, stride, *_extra):
    dt = _DTYPES[storage.dtype]
    arr = np.frombuffer(storage.raw, dtype=dt)
    count = int(np.prod(size)) if len(size) else 1
    arr = arr[offset:offset + count]
    if len(size) == 0:
        return arr.reshape(())
    # Rebuild with the recorded element strides (row-major in practice).
    byte_strides = tuple(s * arr.dtype.itemsize for s in stride)
    return np.lib.stride_tricks.as_strided(arr, shape=size, strides=byte_strides).copy()


def load_state_dict(path: str) -> dict:
    zf = zipfile.ZipFile(path)
    names = zf.namelist()
    pkl_name = next(n for n in names if n.endswith("data.pkl"))
    prefix = pkl_name[: -len("/data.pkl")]
    with zf.open(pkl_name) as f:
        data = f.read()
    up = _RestrictedUnpickler(io.BytesIO(data), zf, prefix)
    return up.load()


if __name__ == "__main__":
    import sys
    sd = load_state_dict(sys.argv[1] if len(sys.argv) > 1 else "pytorch_model.bin")
    total = 0
    for k, v in sd.items():
        n = int(np.prod(v.shape)) if v.shape else 1
        total += n
        print(f"{k:60s} {str(v.shape):28s} {v.dtype}")
    print(f"TOTAL params+buffers: {total:,}")
