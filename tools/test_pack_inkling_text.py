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

    def test_rare_embedding_values_keep_raw_tensor_and_conflicts_do_not_publish(self):
        raw = bytes(range(64))
        header = {"model.llm.embed.weight": {"dtype": "BF16", "shape": [4, 8], "data_offsets": [0, 64]},
                   "model.llm.unembed.weight": {"dtype": "BF16", "shape": [2], "data_offsets": [64, 68]}}
        with tempfile.TemporaryDirectory() as tmp, patch("pack_inkling_text.EMBED_SHAPE", [4, 8]):
            src, dst = Path(tmp) / "source", Path(tmp) / "raw"
            def write_source(extra=b""):
                encoded = json.dumps(header).encode()
                src.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw + b"HEAD" + extra)
            write_source()
            pack(src, dst, prepack_embedding=True)
            with dst.open("rb") as f:
                info = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
                self.assertEqual(info["model.llm.embed.weight"], header["model.llm.embed.weight"])
                self.assertEqual(f.read(), raw + b"HEAD")
            header["model.llm.embed.weight.prefix_v1"] = {"dtype": "U8", "shape": [8], "data_offsets": [68, 76]}
            write_source(bytes(8))
            other = Path(tmp) / "conflict"
            with self.assertRaisesRegex(ValueError, "ambiguous Inkling input embedding"):
                pack(src, other, prepack_embedding=True)
            self.assertFalse(other.exists())

    def test_entropy_scales_choose_each_plane_and_reject_collisions(self):
        scales = {"s13": ("F8_E4M3", [1024]), "s2": ("U8", [256])}
        payload = b"EEHH" + bytes([43]) * 1024 + bytes(range(256)) + b"TAIL"
        header = {
            "model.llm.embed.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
            "model.llm.unembed.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [2, 4]},
            "s13": {"dtype": "F8_E4M3", "shape": [1024], "data_offsets": [4, 1028]},
            "s2": {"dtype": "U8", "shape": [256], "data_offsets": [1028, 1284]},
            "tail": {"dtype": "F32", "shape": [1], "data_offsets": [1284, 1288]},
        }
        with tempfile.TemporaryDirectory() as tmp, patch("pack_inkling_text.expert_scale_tensors", return_value=scales):
            src, dst = Path(tmp) / "source", Path(tmp) / "packed"
            def write_input():
                encoded = json.dumps(header).encode()
                src.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
            write_input()
            pack(src, dst, entropy_scales=True)
            with dst.open("rb") as f:
                out = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
                result = f.read()
            self.assertEqual(out["s13.rans_v1"]["shape"], [536])
            self.assertNotIn("s13", out)
            self.assertEqual(result[out["s2"]["data_offsets"][0]:], bytes(range(256)) + b"TAIL")
            for suffix in (".range_v1", ".rans_v1"):
                header["s13" + suffix] = {"dtype": "U8", "shape": [0], "data_offsets": [1288, 1288]}
                write_input()
                bad = Path(tmp) / ("bad" + suffix)
                with self.assertRaisesRegex(ValueError, "conflicting source expert scales"):
                    pack(src, bad, entropy_scales=True)
                self.assertFalse(bad.exists())
                del header["s13" + suffix]

    def test_mxfp8_encoding_is_optional_per_matrix(self):
        weights = {"mx.a": [1024], "mx.b": [256]}
        data = b"EEHH" + bytes([63])*1024 + bytes(range(256)) + b"TAIL"
        header = {
            "model.llm.embed.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
            "model.llm.unembed.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [2, 4]},
            "mx.a": {"dtype": "F8_E4M3", "shape": [1024], "data_offsets": [4, 1028]},
            "mx.b": {"dtype": "F8_E4M3", "shape": [256], "data_offsets": [1028, 1284]},
            "end": {"dtype": "F32", "shape": [1], "data_offsets": [1284, 1288]}}
        with tempfile.TemporaryDirectory() as tmp, patch("pack_inkling_text.mxfp8_expert_weights", return_value=weights):
            src, dst = Path(tmp)/"src", Path(tmp)/"dst"
            encoded = json.dumps(header).encode()
            src.write_bytes(struct.pack("<Q",len(encoded)) + encoded + data)
            pack(src,dst,prepack_mxfp8=True)
            with dst.open("rb") as f:
                new = json.loads(f.read(struct.unpack("<Q",f.read(8))[0]))
                payload = f.read()
            self.assertNotIn("mx.a", new)
            self.assertEqual(new["mx.a.rans_v1"]["shape"], [536])
            self.assertEqual(new["mx.b"]["dtype"], "F8_E4M3")
            self.assertEqual(payload[540:], bytes(range(256)) + b"TAIL")

    def test_q8_dictionary_from_bf16_and_native_are_identical(self):
        name = "projection.weight"
        prefix = name + ".q8_64."
        # A frequent scale compresses; 36 distinct scales in one short row expand.
        for blocks, use_dict in [([0x3f80] * 36, True), (list(range(0x3f80, 0x3fa4)), False)]:
            values = b"".join(struct.pack("<H", block) * 64 for block in blocks)
            header = {
                "model.llm.embed.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
                "model.llm.unembed.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [2, 4]},
                name: {"dtype": "BF16", "shape": [1, 2304], "data_offsets": [4, 4 + len(values)]},
            }
            with self.subTest(use_dict=use_dict), tempfile.TemporaryDirectory() as tmp, patch(
                "pack_inkling_text.attention_matrices", return_value={name: (1, 1, 2304, prefix)}
            ):
                src, raw, direct, staged = (Path(tmp) / n for n in ("src", "raw", "direct", "staged"))
                encoded = json.dumps(header).encode()
                src.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"EEHH" + values)
                pack(src, raw, prepack_attention=True)
                pack(src, direct, prepack_attention=True, compact_q8_scales=True)
                pack(raw, staged, compact_q8_scales=True)
                self.assertEqual(direct.read_bytes(), staged.read_bytes())
                with direct.open("rb") as f:
                    tensors = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
                self.assertEqual(prefix + "scales.dict_v1" in tensors, use_dict)
                self.assertEqual(prefix + "scales" in tensors, not use_dict)

    def test_refinement_changes_only_attention_output_and_combines_with_dictionary(self):
        names = ["model.llm.layers.0.attn.wo_ud.weight", "model.llm.layers.0.attn.wq_du.weight"]
        catalog = {name: (1, 1, 2304, name + ".q8_64.") for name in names}
        block = [(0x3d80 + 7 * i % 350) | (0x8000 if i % 2 else 0) for i in range(64)]
        matrix = struct.pack("<64H", *block) * 36
        header = {
            "model.llm.embed.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
            "model.llm.unembed.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [2, 4]},
        }
        for k, name in enumerate(names):
            pos = 4 + k * len(matrix)
            header[name] = {"dtype": "BF16", "shape": [1, 2304], "data_offsets": [pos, pos + len(matrix)]}
        with tempfile.TemporaryDirectory() as tmp, patch("pack_inkling_text.attention_matrices", return_value=catalog):
            src, stock, refined, compact, staged = (Path(tmp) / n for n in ("src", "stock", "refined", "compact", "staged"))
            encoded = json.dumps(header).encode()
            src.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"EEHH" + matrix * 2)
            with self.assertRaisesRegex(ValueError, "requires --prepack-attention"):
                pack(src, refined, refine_output=True)
            pack(src, stock, prepack_attention=True)
            pack(src, refined, prepack_attention=True, refine_output=True)
            pack(src, compact, prepack_attention=True, refine_output=True, compact_q8_scales=True)
            pack(refined, staged, compact_q8_scales=True)
            self.assertEqual(compact.read_bytes(), staged.read_bytes())
            def planes(path):
                with path.open("rb") as f:
                    tensors = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
                    payload = f.read()
                return {name: payload[slice(*info["data_offsets"])] for name, info in tensors.items()}
            a, b = planes(stock), planes(refined)
            for name, data in a.items():
                if name.startswith(names[0] + "."):
                    self.assertNotEqual(data, b[name])
                else:
                    self.assertEqual(data, b[name])
            scale = names[0] + ".q8_64.scales"
            self.assertEqual(a[scale], bytes.fromhex("63b1583b") * 36)
            self.assertEqual(b[scale], bytes.fromhex("67b3593b") * 36)

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
