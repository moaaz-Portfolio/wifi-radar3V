import sys

image, model, offset = sys.argv[1], sys.argv[2], int(sys.argv[3], 0)
data = bytearray(open(image, "rb").read())
blob = open(model, "rb").read()
if len(data) > offset:
    if any(b != 0xFF for b in data[offset:offset + len(blob)]) and bytes(data[offset:offset + len(blob)]) != blob:
        raise SystemExit("flash image already uses the model region")
else:
    data.extend(b"\xff" * (offset - len(data)))
data[offset:offset + len(blob)] = blob
open(image, "wb").write(data)
if bytes(open(image, "rb").read()[offset:offset + len(blob)]) != blob:
    raise SystemExit("model was not written into the image")
print("model added at", hex(offset), len(blob), "bytes")
