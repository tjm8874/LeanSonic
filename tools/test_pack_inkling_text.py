import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from pack_inkling_text import attention_matrices, pack


class PackInklingTextTest(unittest.TestCase):
    def test_copy_retained_tensors_and_metadata(self):
        header = {"__metadata__": {"format": "pt"}}
        for name, start in (("model.llm.embed.weight", 0), ("model.visual.proj", 4),
                            ("model.llm.unembed.weight", 8)):
            header[name] = {"dtype": "BF16", "shape": [2], "data_offsets": [start, start + 4]}
        encoded = json.dumps(header).encode()
        encoded += b" " * (-len(encoded) % 8)
        original = struct.pack("<Q", len(encoded)) + encoded + b"aaaabbbbcccc"
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "source", Path(tmp) / "output"
            src.write_bytes(original)
            removed, saved = pack(src, dst)
            self.assertEqual(removed, 1)
            packed = dst.read_bytes()
            self.assertEqual(saved, len(original) - len(packed))
            offset = struct.unpack("<Q", packed[:8])[0] + 8
            self.assertEqual(offset % 8, 0)
            self.assertEqual(packed[offset:], b"aaaacccc")
            del header["model.visual.proj"]
            header["model.llm.unembed.weight"]["data_offsets"] = [4, 8]
            self.assertEqual(json.loads(packed[8:offset]), header)
            with self.assertRaises(FileExistsError):
                pack(src, dst)
            self.assertEqual(src.read_bytes(), original)
            self.assertEqual(dst.read_bytes(), packed)
            src.write_bytes(original[:-1])
            incomplete = Path(tmp) / "incomplete"
            with self.assertRaises(ValueError):
                pack(src, incomplete)
            self.assertFalse(incomplete.exists())

    @patch("pack_inkling_text.build_packer", return_value="inkling_pack_q8")
    def test_native_head_size_and_payload_order(self, _build):
        rows, padded_rows, dim = 200_058, 201_024, 2_304
        original_bytes = padded_rows * dim * 2
        code_bytes = rows * dim
        plane_bytes = code_bytes + rows * (dim // 64) * 4
        header = {
            "model.llm.embed.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
            "model.llm.unembed.weight": {
                "dtype": "BF16", "shape": [padded_rows, dim], "data_offsets": [2, 2 + original_bytes]},
            "tail": {"dtype": "U8", "shape": [2], "data_offsets": [2 + original_bytes, 4 + original_bytes]},
        }
        encoded = json.dumps(header).encode()
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "source", Path(tmp) / "packed"
            with src.open("wb") as f:
                f.write(struct.pack("<Q", len(encoded)) + encoded + b"AAHH")
                f.seek(original_bytes - 2, 1)
                f.write(b"ZZ")

            def emit_planes(args, *, check, stdin, stdout):
                self.assertEqual(stdin.read(2), b"HH")
                stdout.write(b"C")
                stdout.seek(plane_bytes - 2, 1)
                stdout.write(b"S")

            with patch("pack_inkling_text.subprocess.run"):
                with self.assertRaisesRegex(ValueError, "payload length"):
                    pack(src, dst, prepack_head=True)
            self.assertFalse(dst.exists())
            with patch("pack_inkling_text.subprocess.run", side_effect=emit_planes):
                removed, saved = pack(src, dst, prepack_head=True)
            self.assertEqual(removed, 0)
            self.assertEqual(saved, src.stat().st_size - dst.stat().st_size)
            with dst.open("rb") as f:
                size, = struct.unpack("<Q", f.read(8))
                output = json.loads(f.read(size))
                self.assertNotIn("model.llm.unembed.weight", output)
                self.assertEqual(output["leansonic.unembed_q8_64.codes"]["data_offsets"], [2, 2 + code_bytes])
                self.assertEqual(output["leansonic.unembed_q8_64.scales"]["data_offsets"], [2 + code_bytes, 2 + plane_bytes])
                self.assertEqual(f.read(3), b"AAC")
                f.seek(-3, 2)
                self.assertEqual(f.read(), b"SZZ")

    def test_packed_and_incompressible_scales_keep_following_tensor_aligned(self):
        catalog = {"expert.a": ("U8", [256]), "expert.b": ("U8", [256])}
        records = [("model.llm.embed.weight", "BF16", [1], b"EE"),
                   ("model.llm.unembed.weight", "BF16", [1], b"HH"),
                   ("expert.a", "U8", [256], bytes([110] * 255 + [4])),
                   ("expert.b", "U8", [256], bytes(range(256))),
                   ("tail", "F32", [1], b"TAIL")]
        payload = bytearray()
        header = {}
        for name, dtype, shape, data in records:
            start = len(payload)
            payload.extend(data)
            header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(payload)]}
        encoded = json.dumps(header).encode()
        with tempfile.TemporaryDirectory() as tmp, patch("pack_inkling_text.expert_scale_tensors", return_value=catalog):
            src, dst = Path(tmp) / "source", Path(tmp) / "packed"
            src.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
            removed, saved = pack(src, dst, prepack_scales=True)
            self.assertEqual(removed, 0)
            self.assertEqual(saved, src.stat().st_size - dst.stat().st_size)
            with dst.open("rb") as f:
                h = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
                data = f.read()
            self.assertNotIn("expert.a", h)
            a, b = h["expert.a.range_v1"]["data_offsets"]
            self.assertEqual((h["expert.a.range_v1"]["shape"], data[a:a+2]), ([56], bytes([1, 110])))
            self.assertEqual(data[a+16:b], bytes(31) + bytes([128, 4]) + bytes(7))
            a, b = h["expert.b"]["data_offsets"]
            self.assertEqual(data[a:b], bytes(range(256)))
            self.assertEqual(data[b:], b"TAIL")
            self.assertEqual(b % 4, 0)

    def test_prefix_embedding_conversion_and_padding(self):
        raw = bytes([42, 63]) * 256
        data = raw + b"HEAD"
        header = {"model.llm.embed.weight": {"dtype": "BF16", "shape": [8, 32], "data_offsets": [0, len(raw)]},
                  "model.llm.unembed.weight": {"dtype": "BF16", "shape": [2], "data_offsets": [len(raw), len(data)]}}
        encoded = json.dumps(header).encode()
        with tempfile.TemporaryDirectory() as tmp, patch("pack_inkling_text.EMBED_SHAPE", [8, 32]):
            src, dst = Path(tmp) / "source", Path(tmp) / "packed"
            src.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
            removed, saved = pack(src, dst, prepack_embedding=True)
            self.assertEqual((removed, saved), (0, src.stat().st_size - dst.stat().st_size))
            with dst.open("rb") as f:
                h = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
                payload = f.read()
            self.assertNotIn("model.llm.embed.weight", h)
            self.assertEqual(h["model.llm.embed.weight.prefix_v1"]["shape"], [328])
            self.assertEqual(payload, struct.pack("<I", 320) + bytes([42])*256 + bytes(68) + b"HEAD")

    def test_attention_projection_catalog(self):
        matrices = attention_matrices()
        self.assertEqual(len(matrices), 176)
        self.assertEqual(matrices["model.llm.layers.0.attn.wo_ud.weight"][:3], (2304, 2304, 3072))
        for layer in range(24):
            for key in ("k_sconv", "v_sconv"):
                name = f"model.llm.layers.{layer}.attn.{key}.weight_proj"
                if layer in (5, 11, 17, 23):
                    self.assertEqual(matrices[name][:3], (32, 32, 2304))
                else:
                    self.assertNotIn(name, matrices)

    def test_missing_projection_does_not_publish_a_shard(self):
        header = {
            "model.llm.embed.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
            "model.llm.unembed.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [2, 4]},
        }
        encoded = json.dumps(header).encode()
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source", Path(tmp) / "output"
            source.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(4))
            with self.assertRaisesRegex(ValueError, "missing source projection"):
                pack(source, output, prepack_attention=True)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
