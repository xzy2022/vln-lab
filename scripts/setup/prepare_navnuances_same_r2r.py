#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import unicodedata
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANNOTATION_ROOT = REPO_ROOT / "data" / "navnuances" / "annotations" / "NavNuances"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "navnuances" / "same" / "R2R"
DEFAULT_SPLITS = ("DC", "LR", "RR", "VM", "NU")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert NavNuances R2R splits into SAME-readable R2R encoded JSON files."
    )
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=DEFAULT_ANNOTATION_ROOT,
        help="Directory containing R2R_DC.json, R2R_LR.json, ...",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated R2R_*_enc.json files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="NavNuances split suffixes to convert.",
    )
    parser.add_argument(
        "--standard",
        choices=("auto", "include", "skip"),
        default="auto",
        help=(
            "How to handle standard R2R_val_unseen. 'auto' includes it when "
            "annotation-root/R2R_val_unseen.json exists."
        ),
    )
    parser.add_argument(
        "--tokenizer-name",
        default="bert-base-uncased",
        help="HuggingFace tokenizer name used when transformers is available.",
    )
    parser.add_argument(
        "--vocab-file",
        type=Path,
        help="Optional bert-base-uncased vocab.txt fallback when transformers is unavailable.",
    )
    parser.add_argument(
        "--max-instr-len",
        type=int,
        default=0,
        help="Optional maximum encoded instruction length. 0 keeps the full encoding.",
    )
    parser.add_argument(
        "--no-repair-singleton-path",
        action="store_true",
        help=(
            "Do not duplicate one-viewpoint paths. By default singleton paths are duplicated "
            "in the SAME copy so SAME can finish its own R2R metric/export step."
        ),
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=None,
        help="Pretty-print JSON with this indent. Default writes compact JSON.",
    )
    return parser.parse_args()


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def whitespace_tokenize(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return text.split()


def is_whitespace(char: str) -> bool:
    if char in " \t\n\r":
        return True
    return unicodedata.category(char) == "Zs"


def is_control(char: str) -> bool:
    if char in "\t\n\r":
        return False
    return unicodedata.category(char) in ("Cc", "Cf")


def is_punctuation(char: str) -> bool:
    codepoint = ord(char)
    if (33 <= codepoint <= 47) or (58 <= codepoint <= 64) or (91 <= codepoint <= 96) or (123 <= codepoint <= 126):
        return True
    return unicodedata.category(char).startswith("P")


def clean_text(text: str) -> str:
    output = []
    for char in text:
        codepoint = ord(char)
        if codepoint in (0, 0xFFFD) or is_control(char):
            continue
        if is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    return "".join(char for char in text if unicodedata.category(char) != "Mn")


def split_on_punctuation(text: str) -> list[str]:
    output: list[list[str]] = []
    current: list[str] = []
    for char in text:
        if is_punctuation(char):
            if current:
                output.append(current)
                current = []
            output.append([char])
        else:
            current.append(char)
    if current:
        output.append(current)
    return ["".join(chars) for chars in output]


def tokenize_chinese_chars(text: str) -> str:
    output = []
    for char in text:
        codepoint = ord(char)
        is_cjk = (
            (0x4E00 <= codepoint <= 0x9FFF)
            or (0x3400 <= codepoint <= 0x4DBF)
            or (0x20000 <= codepoint <= 0x2A6DF)
            or (0x2A700 <= codepoint <= 0x2B73F)
            or (0x2B740 <= codepoint <= 0x2B81F)
            or (0x2B820 <= codepoint <= 0x2CEAF)
            or (0xF900 <= codepoint <= 0xFAFF)
            or (0x2F800 <= codepoint <= 0x2FA1F)
        )
        if is_cjk:
            output.extend((" ", char, " "))
        else:
            output.append(char)
    return "".join(output)


class BasicWordpieceTokenizer:
    def __init__(self, vocab_file: Path) -> None:
        self.vocab = self.load_vocab(vocab_file)
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"

    @staticmethod
    def load_vocab(vocab_file: Path) -> dict[str, int]:
        vocab: dict[str, int] = {}
        with vocab_file.open("r", encoding="utf-8") as handle:
            for index, token in enumerate(handle):
                vocab[token.rstrip("\n")] = index
        for special in ("[UNK]", "[CLS]", "[SEP]"):
            if special not in vocab:
                raise ValueError(f"{vocab_file} does not contain required token {special}")
        return vocab

    def basic_tokenize(self, text: str) -> list[str]:
        text = clean_text(text)
        text = tokenize_chinese_chars(text)
        output_tokens: list[str] = []
        for token in whitespace_tokenize(text):
            token = strip_accents(token.lower())
            output_tokens.extend(split_on_punctuation(token))
        return whitespace_tokenize(" ".join(output_tokens))

    def wordpiece_tokenize(self, token: str) -> list[str]:
        if len(token) > 100:
            return [self.unk_token]

        sub_tokens = []
        start = 0
        while start < len(token):
            end = len(token)
            current_substr = None
            while start < end:
                substr = token[start:end]
                if start > 0:
                    substr = f"##{substr}"
                if substr in self.vocab:
                    current_substr = substr
                    break
                end -= 1

            if current_substr is None:
                return [self.unk_token]

            sub_tokens.append(current_substr)
            start = end

        return sub_tokens

    def encode(self, text: str) -> list[int]:
        tokens = [self.cls_token]
        for token in self.basic_tokenize(text):
            tokens.extend(self.wordpiece_tokenize(token))
        tokens.append(self.sep_token)
        return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]


def iter_hf_vocab_candidates(tokenizer_name: str) -> Iterable[Path]:
    cache_roots = []
    for raw_root in (
        os.environ.get("HF_HOME"),
        os.environ.get("TRANSFORMERS_CACHE"),
        Path.home() / ".cache" / "huggingface",
    ):
        if raw_root:
            cache_roots.append(Path(raw_root).expanduser())

    model_dir = f"models--{tokenizer_name.replace('/', '--')}"
    for cache_root in cache_roots:
        hub_dir = cache_root / "hub" / model_dir / "snapshots"
        if hub_dir.exists():
            for candidate in sorted(hub_dir.glob("*/vocab.txt"), reverse=True):
                yield candidate
        direct_candidate = cache_root / model_dir / "vocab.txt"
        if direct_candidate.exists():
            yield direct_candidate


def build_encoder(tokenizer_name: str, vocab_file: Path | None):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        tokenizer = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if tokenizer is not None:
        return lambda text: tokenizer.encode(text, add_special_tokens=True)

    if vocab_file is None:
        vocab_file = next(iter_hf_vocab_candidates(tokenizer_name), None)

    if vocab_file is None or not vocab_file.exists():
        raise RuntimeError(
            "transformers is not installed and no local vocab.txt was found. "
            "Install transformers in the SAME environment or pass --vocab-file."
        )

    fallback = BasicWordpieceTokenizer(vocab_file)
    print(f"Using fallback BERT vocab: {repo_rel(vocab_file)}")
    return fallback.encode


def normalize_split(split: str) -> str:
    split = split.strip()
    if split.upper().startswith("R2R_"):
        split = split[4:]
    if split.upper().endswith(".JSON"):
        split = split[:-5]
    return split.upper()


def write_json(path: Path, payload, indent: int | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=indent)


def prepare_item(item: dict, encode, max_instr_len: int, repair_singleton_path: bool) -> tuple[dict, int]:
    new_item = dict(item)
    instructions = new_item.get("instructions")
    if instructions is None and "instruction" in new_item:
        instructions = [new_item["instruction"]]
        new_item["instructions"] = instructions
    if not isinstance(instructions, list) or not all(isinstance(text, str) for text in instructions):
        raise ValueError(f"path_id={new_item.get('path_id')} has invalid instructions")

    encodings = [encode(instruction) for instruction in instructions]
    if max_instr_len > 0:
        encodings = [encoding[:max_instr_len] for encoding in encodings]
    new_item["instr_encodings"] = encodings

    repaired = 0
    path = new_item.get("path")
    if repair_singleton_path and isinstance(path, list) and len(path) == 1:
        new_item["path"] = [path[0], path[0]]
        repaired = 1

    return new_item, repaired


def prepare_items(
    data: list[dict],
    encode,
    max_instr_len: int,
    repair_singleton_path: bool,
) -> tuple[list[dict], int]:
    repaired_count = 0
    prepared_data = []
    for item in data:
        prepared_item, repaired = prepare_item(
            item,
            encode=encode,
            max_instr_len=max_instr_len,
            repair_singleton_path=repair_singleton_path,
        )
        prepared_data.append(prepared_item)
        repaired_count += repaired
    return prepared_data, repaired_count


def maybe_prepare_standard_split(args: argparse.Namespace, encode) -> int:
    if args.standard == "skip":
        return 0

    output_path = args.output_dir.resolve() / "R2R_val_unseen_enc.json"
    annotation_path = args.annotation_root.resolve() / "R2R_val_unseen.json"
    if not annotation_path.exists():
        if args.standard == "include":
            raise FileNotFoundError(f"Missing standard R2R annotation file: {repo_rel(annotation_path)}")
        print(f"Skipped standard R2R_val_unseen: missing {repo_rel(annotation_path)}")
        return 0

    with annotation_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{repo_rel(annotation_path)} must contain a JSON list")

    prepared_data, repaired_count = prepare_items(
        data,
        encode=encode,
        max_instr_len=args.max_instr_len,
        repair_singleton_path=False,
    )
    write_json(output_path, prepared_data, indent=args.indent)
    print(
        f"Wrote {repo_rel(output_path)}: {len(prepared_data)} standard R2R items "
        f"from {repo_rel(annotation_path)}, {repaired_count} singleton paths repaired"
    )
    return len(data)


def main() -> None:
    args = parse_args()
    annotation_root = args.annotation_root.resolve()
    output_dir = args.output_dir.resolve()
    splits = [normalize_split(split) for split in args.splits]
    encode = build_encoder(args.tokenizer_name, args.vocab_file)

    output_dir.mkdir(parents=True, exist_ok=True)

    total_items = 0
    total_repaired = 0
    for split in splits:
        input_path = annotation_root / f"R2R_{split}.json"
        output_path = output_dir / f"R2R_{split}_enc.json"
        if not input_path.exists():
            raise FileNotFoundError(f"Missing NavNuances annotation file: {repo_rel(input_path)}")

        with input_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError(f"{repo_rel(input_path)} must contain a JSON list")

        prepared_data, repaired_count = prepare_items(
            data,
            encode=encode,
            max_instr_len=args.max_instr_len,
            repair_singleton_path=not args.no_repair_singleton_path,
        )

        write_json(output_path, prepared_data, indent=args.indent)

        total_items += len(prepared_data)
        total_repaired += repaired_count
        print(
            f"Wrote {repo_rel(output_path)}: {len(prepared_data)} items, "
            f"{repaired_count} singleton paths repaired"
        )

    standard_items = maybe_prepare_standard_split(args, encode)
    print(
        f"Done: {total_items + standard_items} items "
        f"({total_items} NavNuances + {standard_items} standard R2R), "
        f"{total_repaired} singleton paths repaired"
    )


if __name__ == "__main__":
    main()
