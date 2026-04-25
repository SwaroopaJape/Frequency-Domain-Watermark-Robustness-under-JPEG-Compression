import numpy as np

# generate a 1d binary array of +/- 1 values from text
def generate(text: str = "TEAM-THE_MORPHIX-IIT-GOA") -> np.ndarray:
    raw = text.encode("utf-8")
    bits = []
    for byte in raw:
        for i in range(7, -1, -1):  # MSB first
            bits.append(1.0 if (byte >> i) & 1 else -1.0)
    return np.array(bits, dtype=np.float32)

# decode a +/- 1 bit array back to a string
def decode(bits: np.ndarray, strip_errors: bool = True) -> str:
    b = (bits > 0).astype(np.uint8)
    n_bytes = len(b) // 8
    chars = []
    for i in range(n_bytes):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | int(b[i * 8 + j])
        if strip_errors:
            try:
                chars.append(bytes([byte]).decode("utf-8"))
            except UnicodeDecodeError:
                chars.append("?")
        else:
            chars.append(chr(byte))
    return "".join(chars)
