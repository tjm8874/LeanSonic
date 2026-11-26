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
