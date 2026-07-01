from __future__ import annotations

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

from cascadia_cluster.object_store import ObjectStoreClient, ObjectStoreConfig


class _ObjectHandler(BaseHTTPRequestHandler):
    objects: ClassVar[dict[str, bytes]] = {}

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_HEAD(self) -> None:
        self.send_response(200)
        self.end_headers()

    def do_PUT(self) -> None:
        length = int(self.headers["Content-Length"])
        self.objects[self.path] = self.rfile.read(length)
        self.send_response(200)
        self.end_headers()

    def do_GET(self) -> None:
        value = self.objects[self.path]
        self.send_response(200)
        self.send_header("Content-Length", str(len(value)))
        self.end_headers()
        self.wfile.write(value)


def test_file_upload_and_download_stream_without_read_bytes(tmp_path: Path) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ObjectHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        source = tmp_path / "large.bin"
        payload = bytes(range(251)) * 24_000
        source.write_bytes(payload)
        client = ObjectStoreClient(
            ObjectStoreConfig(
                endpoint=f"http://127.0.0.1:{server.server_port}",
                access_key="access",
                secret_key="secret",
            )
        )
        expected = hashlib.sha256(payload).hexdigest()
        assert client.put_file("inputs", "sha256/data.bin", source) == expected
        destination = tmp_path / "downloaded.bin"
        assert client.download("inputs", "sha256/data.bin", destination) == expected
        assert destination.read_bytes() == payload
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
