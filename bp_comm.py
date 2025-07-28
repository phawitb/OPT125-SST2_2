# bp_comm.py  -- TCP helpers (JSON + chunked file transfer) + return size
import socket, struct, json, os, tempfile, time

CHUNK = 4 * 1024 * 1024  # 4 MB

def recvall(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf

def send_json(sock: socket.socket, obj: dict):
    data = json.dumps(obj).encode("utf-8")
    sock.sendall(struct.pack("!I", len(data)))
    sock.sendall(data)

def recv_json(sock: socket.socket) -> dict:
    raw_len = recvall(sock, 4)
    if not raw_len:
        raise ConnectionError("closed while reading json length")
    length = struct.unpack("!I", raw_len)[0]
    data = recvall(sock, length)
    return json.loads(data.decode("utf-8"))

def send_file(sock: socket.socket, path: str) -> (int, float):
    size = os.path.getsize(path)
    sock.sendall(struct.pack("!Q", size))
    t0 = time.perf_counter()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            sock.sendall(chunk)
    t = time.perf_counter() - t0
    return size, t  # bytes, seconds

def recv_file(sock: socket.socket, save_to: str) -> (int, float):
    size_data = recvall(sock, 8)
    if not size_data:
        raise ConnectionError("closed while reading file size")
    size = struct.unpack("!Q", size_data)[0]

    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(save_to) or ".")
    t0 = time.perf_counter()
    with os.fdopen(fd, "wb") as f:
        remaining = size
        while remaining:
            chunk = sock.recv(min(CHUNK, remaining))
            if not chunk:
                raise ConnectionError("closed during file recv")
            f.write(chunk)
            remaining -= len(chunk)
    t = time.perf_counter() - t0
    os.replace(tmp, save_to)
    return size, t
