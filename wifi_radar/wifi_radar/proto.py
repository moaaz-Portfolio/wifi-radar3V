import struct
import numpy as np

MAGIC = b"CSI1"
_HDR = struct.Struct("<4sIIbB")


def parse(pkt: bytes):
    if len(pkt) < _HDR.size:
        return None
    magic, seq, ts, rssi, n = _HDR.unpack_from(pkt)


    if magic != MAGIC or n != 64 or len(pkt) != _HDR.size + n:
        return None
    amp = np.frombuffer(pkt, dtype=np.uint8, count=n, offset=_HDR.size)
    return seq, ts, rssi, amp


def build(seq, ts_ms, rssi, amp) -> bytes:
    amp = np.asarray(amp, dtype=np.uint8)
    return _HDR.pack(MAGIC, seq & 0xFFFFFFFF, ts_ms & 0xFFFFFFFF, rssi, len(amp)) + amp.tobytes()
