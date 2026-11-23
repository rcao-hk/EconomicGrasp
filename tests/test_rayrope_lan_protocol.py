import json
import socket
import struct
import unittest

import numpy as np

from demo.rayrope_lan.protocol import (
    decode_npz,
    encode_npz,
    recv_frame,
    send_frame,
)


class RayRoPEProtocolTests(unittest.TestCase):
    def test_npz_round_trip_without_pickle(self):
        arrays = {
            "rgb": np.zeros((12, 20, 3), dtype=np.uint8),
            "depth": np.full((12, 20), 1200, dtype=np.uint16),
            "metadata_json": np.frombuffer(
                json.dumps({"request_id": "unit-test"}).encode("utf-8"),
                dtype=np.uint8,
            ),
        }
        decoded = decode_npz(encode_npz(arrays))
        np.testing.assert_array_equal(decoded["rgb"], arrays["rgb"])
        np.testing.assert_array_equal(decoded["depth"], arrays["depth"])
        self.assertEqual(
            json.loads(decoded["metadata_json"].tobytes().decode("utf-8")),
            {"request_id": "unit-test"},
        )

    def test_length_prefixed_socket_round_trip(self):
        sender, receiver = socket.socketpair()
        try:
            payload = encode_npz({"value": np.arange(17, dtype=np.float32)})
            send_frame(sender, payload)
            received = recv_frame(receiver, max_bytes=1024 * 1024)
            decoded = decode_npz(received)
            np.testing.assert_array_equal(
                decoded["value"], np.arange(17, dtype=np.float32)
            )
        finally:
            sender.close()
            receiver.close()

    def test_oversized_frame_is_rejected_before_payload_read(self):
        sender, receiver = socket.socketpair()
        try:
            sender.sendall(struct.pack("!Q", 4096))
            with self.assertRaisesRegex(ValueError, "too large"):
                recv_frame(receiver, max_bytes=128)
        finally:
            sender.close()
            receiver.close()


if __name__ == "__main__":
    unittest.main()
