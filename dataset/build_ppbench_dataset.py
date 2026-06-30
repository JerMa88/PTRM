"""
PPBench Dataset Builder for TRM Training.

Uses the ppbench library (pzpr.js engine) to properly decode all puzzle types.

Implements the exact data pipeline from arXiv:2605.19943, Appendix A.3:
  - 6 puzzle types: sudoku, lightup, nurikabe, shakashaka, heyawake, tapa
  - Grid sizes: 9×9 (sudoku, padded to 10×10), 10×10 (rest) → seq_len=100
  - Unified vocabulary (paper: 294 tokens)
  - Augmentation: 10 examples per puzzle (1 original + 9 trajectory×dihedral)
  - Splits: train / val (100 per type, 50 for tapa) / golden (from PPBench golden set)

Requirements:
    pip install ppbench
    Node.js (for pzpr.js puzzle engine)

Usage:
    python dataset/build_ppbench_dataset.py
    python dataset/build_ppbench_dataset.py --output-dir data/ppbench --dry-run
"""

import argparse
import base64
import json
import os
import sys
import re
from collections import defaultdict, OrderedDict
from typing import Optional, Tuple, List

import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants matching the paper (Appendix A.3)
# ---------------------------------------------------------------------------

PUZZLE_TYPES = ["sudoku", "lightup", "nurikabe", "shakashaka", "heyawake", "tapa"]

GRID_SIZES = {
    "sudoku": (9, 9),
    "lightup": (10, 10),
    "nurikabe": (10, 10),
    "shakashaka": (10, 10),
    "heyawake": (10, 10),
    "tapa": (10, 10),
}

SEQ_LEN = 100

VAL_SIZES = {
    "sudoku": 100,
    "lightup": 100,
    "nurikabe": 100,
    "shakashaka": 100,
    "heyawake": 100,
    "tapa": 50,  # "50 for tapa, due to its smaller base size"
}

NUM_AUGMENTS_PER_PUZZLE = 9

PPBENCH_XOR_KEY = b"ppbench"


# ---------------------------------------------------------------------------
# Solution decryption
# ---------------------------------------------------------------------------

def decode_solution(enc: str) -> dict:
    """Decrypt PPBench's XOR-encrypted, base64-encoded solution."""
    raw = base64.b64decode(enc)
    decrypted = bytes(b ^ PPBENCH_XOR_KEY[i % len(PPBENCH_XOR_KEY)]
                      for i, b in enumerate(raw))
    return json.loads(decrypted)


# ---------------------------------------------------------------------------
# pzprv3 state parser → grid of cell symbols
# ---------------------------------------------------------------------------

def parse_pzprv3_state(state_text: str) -> Tuple[str, int, int, List[List[str]]]:
    """
    Parse a pzprv3 state string into puzzle type, dimensions, and grid(s).

    pzprv3 format:
        pzprv3
        <puzzle_type>
        <height> (or <width> then <height> for non-square)
        [<num_rooms> for heyawake]
        [room assignment rows for heyawake]
        <grid rows: space-separated cell values>
        [additional grid sections]

    Returns:
        (puzzle_type, width, height, list of grid rows where each row is a list of cell strings)
    """
    lines = state_text.strip().split("\n")
    idx = 0

    # Skip "pzprv3" header
    if lines[idx].strip() == "pzprv3":
        idx += 1

    # Puzzle type
    puzzle_type = lines[idx].strip()
    idx += 1

    # Dimensions
    if puzzle_type == "sudoku":
        # sudoku has a single size line (it's square)
        size = int(lines[idx].strip())
        width, height = size, size
        idx += 1
    else:
        # Most puzzles: width then height on separate lines, or on one line
        dim1 = int(lines[idx].strip())
        idx += 1
        dim2 = int(lines[idx].strip())
        idx += 1
        # pzprv3 uses height, width order (cols, rows)
        height, width = dim2, dim1

    # For heyawake, skip room data
    extra_sections = []
    if puzzle_type == "heyawake":
        num_rooms = int(lines[idx].strip())
        idx += 1
        # Skip room assignment rows (height rows)
        for _ in range(height):
            idx += 1

    # Parse main grid (problem layer — clues)
    problem_grid = []
    for r in range(height):
        if idx < len(lines):
            row = lines[idx].strip().split()
            problem_grid.append(row)
            idx += 1
        else:
            problem_grid.append(["."] * width)

    # Parse answer grid if present (solution layer)
    answer_grid = []
    for r in range(height):
        if idx < len(lines):
            row = lines[idx].strip().split()
            answer_grid.append(row)
            idx += 1

    return puzzle_type, width, height, problem_grid, answer_grid


def state_to_grids(state_text: str) -> Tuple[List[List[str]], List[List[str]], str]:
    """
    Convert a pzprv3 state into problem and answer grids.

    For sudoku: problem grid has clue digits and '.'; answer grid has user-entered digits.
    For binary puzzles (lightup, nurikabe, etc.): answer grid has # (shaded) + (unshaded) etc.

    Returns (problem_grid, answer_grid, puzzle_type) where grids are lists of rows of strings.
    """
    ptype, w, h, problem, answer = parse_pzprv3_state(state_text)
    return problem, answer, ptype


# ---------------------------------------------------------------------------
# Unified cell tokenizer
# ---------------------------------------------------------------------------

class CellTokenizer:
    """
    Maps cell symbols to integer token IDs with a type-prefixed scheme.

    Builds a unified vocabulary across all puzzle types:
      - Token 0: PAD (for padding sudoku 9×9 → 10×10)
      - Token 1: BLANK (empty cell, encoded as '.' in pzprv3)
      - Then type-specific tokens for each unique cell symbol

    The paper states 294 tokens — this comes from the combined symbol sets
    across all 6 puzzle types.
    """

    def __init__(self):
        self.token_map = OrderedDict()  # (puzzle_type, symbol) → token_id
        self.PAD_ID = 0
        self.BLANK_ID = 1
        self.next_id = 2

        # Pre-register PAD and BLANK
        self.token_map[("_global", "_PAD")] = self.PAD_ID
        self.token_map[("_global", "_BLANK")] = self.BLANK_ID

    def register(self, puzzle_type: str, symbol: str) -> int:
        """Register a (type, symbol) pair and return its token ID."""
        if symbol == ".":
            return self.BLANK_ID

        key = (puzzle_type, symbol)
        if key not in self.token_map:
            self.token_map[key] = self.next_id
            self.next_id += 1
        return self.token_map[key]

    def encode(self, puzzle_type: str, symbol: str) -> int:
        """Encode a cell symbol to its token ID (register if new)."""
        if symbol == ".":
            return self.BLANK_ID
        key = (puzzle_type, symbol)
        if key in self.token_map:
            return self.token_map[key]
        return self.register(puzzle_type, symbol)

    @property
    def vocab_size(self):
        return self.next_id

    def save(self, path: str):
        """Save the token map to a JSON file."""
        inverse = {}
        for (ptype, sym), tid in self.token_map.items():
            inverse[str(tid)] = {"type": ptype, "symbol": sym}
        with open(path, "w") as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "pad_id": self.PAD_ID,
                "blank_id": self.BLANK_ID,
                "tokens": inverse,
            }, f, indent=2)


# ---------------------------------------------------------------------------
# Grid construction using ppbench
# ---------------------------------------------------------------------------

def build_grids_from_ppbench(record: dict, puzzle_type: str) -> Optional[dict]:
    """
    Build initial (input) and solved (label) grids from a PPBench record
    using the ppbench library for proper puzzle decoding.

    Returns dict with:
      - input_symbols: 2D list of cell symbol strings (initial state)
      - label_symbols: 2D list of cell symbol strings (solved state)
    """
    from ppbench import Puzzle

    try:
        url = record["puzzlink_url"]
        width = record["width"]
        height = record["height"]

        # Decode solution moves
        solution = decode_solution(record["solution_enc"])
        moves_full = solution.get("moves_full", [])

        # Load puzzle and get initial state
        puzzle = Puzzle.from_url(url)
        initial_state = puzzle.get_state()

        # Apply all solution moves
        for move in moves_full:
            puzzle.send_move(move)
        solved_state = puzzle.get_state()

        # Parse states into grids
        init_problem, init_answer, _ = state_to_grids(initial_state)
        solved_problem, solved_answer, _ = state_to_grids(solved_state)

        # Build combined grids:
        # Input = problem layer (clues) only
        # Label = problem layer + answer layer merged (full solution)
        input_symbols = _merge_layers(init_problem, init_answer, puzzle_type, is_input=True)
        label_symbols = _merge_layers(solved_problem, solved_answer, puzzle_type, is_input=False)

        return {
            "input_symbols": input_symbols,
            "label_symbols": label_symbols,
            "puzzle_type": puzzle_type,
            "width": width,
            "height": height,
        }

    except Exception as e:
        return None


def _merge_layers(problem: List[List[str]], answer: List[List[str]],
                  puzzle_type: str, is_input: bool) -> List[List[str]]:
    """
    Merge problem (clue) and answer (solution) layers into a single grid.

    For input grids: only problem layer (clues), blanks for unsolved cells.
    For label grids: merge both layers — answer fills in blanks.
    """
    height = len(problem)
    width = len(problem[0]) if problem else 0

    merged = []
    for r in range(height):
        row = []
        for c in range(width):
            prob_cell = problem[r][c] if r < len(problem) and c < len(problem[r]) else "."
            ans_cell = answer[r][c] if answer and r < len(answer) and c < len(answer[r]) else "."

            if puzzle_type == "sudoku":
                # Sudoku: problem has digits and '.'; answer has user-entered digits and '.'
                if is_input:
                    row.append(prob_cell)  # Clues only
                else:
                    # Merge: if answer has a digit, use it; otherwise use problem
                    if ans_cell != "." and ans_cell != "0":
                        row.append(ans_cell)
                    elif prob_cell != ".":
                        row.append(prob_cell)
                    else:
                        row.append(".")
            else:
                # For other puzzles: answer layer has the solve state (#, +, etc.)
                if is_input:
                    # Input: show only clue values (numbers, walls), blank for solvable cells
                    if prob_cell in (".", "0"):
                        row.append(".")
                    else:
                        row.append(prob_cell)
                else:
                    # Label: prefer answer state; if answer is blank, use problem
                    if ans_cell != "." and ans_cell != "0":
                        row.append(ans_cell)
                    elif prob_cell != "." and prob_cell != "0":
                        row.append(prob_cell)
                    else:
                        row.append(".")

        merged.append(row)

    return merged


# ---------------------------------------------------------------------------
# Augmentation (Appendix A.3)
# ---------------------------------------------------------------------------

def trajectory_sample_symbols(input_grid: List[List[str]],
                              label_grid: List[List[str]],
                              rng: np.random.Generator) -> List[List[str]]:
    """
    Trajectory sampling: randomly reveal some solution cells in the input.
    """
    h, w = len(input_grid), len(input_grid[0])
    augmented = [row[:] for row in input_grid]

    # Find blank cells in input that have values in label
    blanks = []
    for r in range(h):
        for c in range(w):
            if input_grid[r][c] == "." and label_grid[r][c] != ".":
                blanks.append((r, c))

    if not blanks:
        return augmented

    num_reveal = rng.integers(0, len(blanks) + 1)
    if num_reveal == 0:
        return augmented

    reveal_indices = rng.choice(len(blanks), size=num_reveal, replace=False)
    for idx in reveal_indices:
        r, c = blanks[idx]
        augmented[r][c] = label_grid[r][c]

    return augmented


def dihedral_transform_symbols(grid: List[List[str]], tid: int) -> List[List[str]]:
    """Apply dihedral transform to a symbol grid."""
    arr = np.array(grid, dtype=object)

    if tid == 0:
        result = arr
    elif tid == 1:
        result = np.rot90(arr, k=1)
    elif tid == 2:
        result = np.rot90(arr, k=2)
    elif tid == 3:
        result = np.rot90(arr, k=3)
    elif tid == 4:
        result = np.fliplr(arr)
    elif tid == 5:
        result = np.flipud(arr)
    elif tid == 6:
        result = arr.T
    elif tid == 7:
        result = np.fliplr(np.rot90(arr, k=1))
    else:
        result = arr

    return result.tolist()


def augment_puzzle_symbols(input_grid: List[List[str]],
                           label_grid: List[List[str]],
                           num_augments: int,
                           rng: np.random.Generator) -> List[Tuple]:
    """Generate augmented (input, label) pairs from a puzzle."""
    examples = [(input_grid, label_grid)]

    for _ in range(num_augments):
        aug_input = trajectory_sample_symbols(input_grid, label_grid, rng)
        tid = rng.integers(0, 8)
        aug_input = dihedral_transform_symbols(aug_input, tid)
        aug_label = dihedral_transform_symbols(label_grid, tid)
        examples.append((aug_input, aug_label))

    return examples


# ---------------------------------------------------------------------------
# Grid → token sequence conversion
# ---------------------------------------------------------------------------

def grid_to_tokens(grid: List[List[str]], puzzle_type: str,
                   tokenizer: CellTokenizer, target_size: int = 10) -> np.ndarray:
    """
    Convert a symbol grid to a flat token sequence.

    Pads to target_size × target_size (for sudoku 9×9 → 10×10).
    """
    h, w = len(grid), len(grid[0])
    tokens = np.full(target_size * target_size, tokenizer.PAD_ID, dtype=np.int32)

    for r in range(min(h, target_size)):
        for c in range(min(w, target_size)):
            tokens[r * target_size + c] = tokenizer.encode(puzzle_type, grid[r][c])

    return tokens


# ---------------------------------------------------------------------------
# PPBench loading
# ---------------------------------------------------------------------------

def load_ppbench_records(data_path: str = None) -> Tuple[list, list]:
    """
    Load PPBench records. If data_path is provided, load from file.
    Otherwise use ppbench library to load directly.
    """
    if data_path and os.path.exists(data_path):
        records = []
        with open(data_path, "r") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records, []

    # Use ppbench library
    from ppbench import load_dataset
    full = load_dataset("full")
    golden = load_dataset("golden")
    return full, golden


def filter_by_type_and_size(records: list) -> dict:
    """Filter records to our 6 puzzle types at correct grid sizes."""
    filtered = defaultdict(list)
    for r in records:
        ptype = r.get("pid", "")
        w = r.get("width", 0)
        h = r.get("height", 0)

        if ptype not in GRID_SIZES:
            continue
        expected_w, expected_h = GRID_SIZES[ptype]
        if w == expected_w and h == expected_h:
            filtered[ptype].append(r)

    return dict(filtered)


def split_golden_val_train(filtered: dict, golden_records: list,
                           rng: np.random.Generator) -> Tuple[dict, dict, dict]:
    """Split into golden / val / train per the paper's spec."""
    golden_urls = set()
    golden_filtered = filter_by_type_and_size(golden_records)
    for ptype, recs in golden_filtered.items():
        for r in recs:
            golden_urls.add(r.get("puzzlink_url", ""))

    golden, remaining = defaultdict(list), defaultdict(list)
    for ptype, recs in filtered.items():
        for r in recs:
            if r.get("puzzlink_url", "") in golden_urls:
                golden[ptype].append(r)
            else:
                remaining[ptype].append(r)

    val, train = {}, {}
    for ptype, recs in remaining.items():
        val_size = min(VAL_SIZES.get(ptype, 100), len(recs))
        indices = rng.permutation(len(recs))
        val[ptype] = [recs[i] for i in indices[:val_size]]
        train[ptype] = [recs[i] for i in indices[val_size:]]

    return dict(golden), val, train


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build_ppbench_dataset(output_dir: str = "data/ppbench",
                          seed: int = 42,
                          dry_run: bool = False):
    """Build the complete PPBench dataset for TRM training."""
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load data
    print("Loading PPBench data...")

    # Download if needed
    raw_dir = os.path.join(output_dir, "_raw")
    full_path = os.path.join(raw_dir, "full_dataset.jsonl")
    golden_path = os.path.join(raw_dir, "golden_300.jsonl")

    if not os.path.exists(full_path):
        from huggingface_hub import hf_hub_download
        print("  Downloading full_dataset.jsonl from HuggingFace...")
        full_path = hf_hub_download(
            "bluecoconut/pencil-puzzle-bench", "full_dataset.jsonl",
            repo_type="dataset", local_dir=raw_dir,
        )
    if not os.path.exists(golden_path):
        from huggingface_hub import hf_hub_download
        print("  Downloading golden_300.jsonl from HuggingFace...")
        golden_path = hf_hub_download(
            "bluecoconut/pencil-puzzle-bench", "golden_300.jsonl",
            repo_type="dataset", local_dir=raw_dir,
        )

    # Load JSONL files
    def load_jsonl(path):
        records = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    full_records = load_jsonl(full_path)
    golden_records = load_jsonl(golden_path)
    print(f"  Full: {len(full_records)} puzzles")
    print(f"  Golden: {len(golden_records)} puzzles")

    # Step 2: Filter
    print("Filtering to 6 puzzle types at correct sizes...")
    filtered = filter_by_type_and_size(full_records)
    for ptype in PUZZLE_TYPES:
        print(f"  {ptype}: {len(filtered.get(ptype, []))}")

    # Step 3: Split
    print("Splitting into train/val/golden...")
    golden, val, train = split_golden_val_train(filtered, golden_records, rng)

    total_train = sum(len(v) for v in train.values())
    total_val = sum(len(v) for v in val.values())
    total_golden = sum(len(v) for v in golden.values())

    print(f"\nSplit counts (base puzzles, before augmentation):")
    print(f"{'type':<15} {'train':>8} {'val':>8} {'golden':>8}")
    print("-" * 45)
    for ptype in PUZZLE_TYPES:
        t = len(train.get(ptype, []))
        v = len(val.get(ptype, []))
        g = len(golden.get(ptype, []))
        print(f"  {ptype:<13} {t:>8} {v:>8} {g:>8}")
    print(f"  {'TOTAL':<13} {total_train:>8} {total_val:>8} {total_golden:>8}")

    if dry_run:
        print(f"\n[DRY RUN] Would produce ~{total_train * 10} train examples. Exiting.")
        return

    # Step 4: Process puzzles using ppbench (proper pzpr.js decoding)
    tokenizer = CellTokenizer()

    # First pass: register all symbols to build vocabulary
    print("\nPass 1: Building vocabulary by scanning all puzzles...")
    all_processed = {"train": {}, "val": {}, "golden": {}}

    for split_name, split_data in [("train", train), ("val", val), ("golden", golden)]:
        for ptype in PUZZLE_TYPES:
            records = split_data.get(ptype, [])
            processed = []
            failures = 0

            for record in tqdm(records, desc=f"  {split_name}/{ptype}", leave=False):
                result = build_grids_from_ppbench(record, ptype)
                if result is not None:
                    processed.append(result)
                    # Register all symbols
                    for row in result["input_symbols"]:
                        for sym in row:
                            tokenizer.register(ptype, sym)
                    for row in result["label_symbols"]:
                        for sym in row:
                            tokenizer.register(ptype, sym)
                else:
                    failures += 1

            all_processed[split_name][ptype] = processed
            if failures:
                print(f"    {split_name}/{ptype}: {failures} failures out of {len(records)}")

    print(f"\n  Vocabulary size: {tokenizer.vocab_size} tokens")

    # Step 5: Build TRM-format splits
    for split_name in ["train", "val", "golden"]:
        augment = (split_name == "train")
        print(f"\nBuilding {split_name} split {'(with augmentation)' if augment else '(no augmentation)'}...")

        all_inputs = []
        all_labels = []
        all_puzzle_ids = []
        puzzle_indices = [0]
        group_indices = [0]
        puzzle_count = 0
        example_count = 0

        for ptype_idx, ptype in enumerate(PUZZLE_TYPES):
            processed = all_processed[split_name].get(ptype, [])

            for p in tqdm(processed, desc=f"  {ptype}", leave=False):
                input_sym = p["input_symbols"]
                label_sym = p["label_symbols"]

                if augment:
                    examples = augment_puzzle_symbols(
                        input_sym, label_sym, NUM_AUGMENTS_PER_PUZZLE, rng
                    )
                else:
                    examples = [(input_sym, label_sym)]

                for aug_input, aug_label in examples:
                    input_tokens = grid_to_tokens(aug_input, ptype, tokenizer)
                    label_tokens = grid_to_tokens(aug_label, ptype, tokenizer)

                    all_inputs.append(input_tokens)
                    all_labels.append(label_tokens)
                    all_puzzle_ids.append(ptype_idx)

                    example_count += 1
                    puzzle_count += 1
                    puzzle_indices.append(example_count)

                group_indices.append(puzzle_count)

        # Convert to numpy
        data = {
            "inputs": np.array(all_inputs, dtype=np.int32) if all_inputs else np.zeros((0, SEQ_LEN), dtype=np.int32),
            "labels": np.array(all_labels, dtype=np.int32) if all_labels else np.zeros((0, SEQ_LEN), dtype=np.int32),
            "puzzle_identifiers": np.array(all_puzzle_ids, dtype=np.int32),
            "puzzle_indices": np.array(puzzle_indices, dtype=np.int32),
            "group_indices": np.array(group_indices, dtype=np.int32),
        }

        # Save
        # TRM expects val split in a directory called "test"
        save_name = "test" if split_name == "val" else split_name
        save_dir = os.path.join(output_dir, save_name)
        os.makedirs(save_dir, exist_ok=True)

        for key, arr in data.items():
            np.save(os.path.join(save_dir, f"all__{key}.npy"), arr)

        num_groups = len(data["group_indices"]) - 1
        metadata = {
            "seq_len": SEQ_LEN,
            "vocab_size": tokenizer.vocab_size,
            "pad_id": tokenizer.PAD_ID,
            "ignore_label_id": tokenizer.PAD_ID,
            "blank_identifier_id": 0,
            "num_puzzle_identifiers": len(PUZZLE_TYPES),
            "total_groups": num_groups,
            "mean_puzzle_examples": len(data["inputs"]) / max(num_groups, 1),
            "total_puzzles": len(data["puzzle_indices"]) - 1,
            "sets": ["all"],
        }

        with open(os.path.join(save_dir, "dataset.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Saved {split_name}: {len(data['inputs'])} examples to {save_dir}")

    # Save vocab + identifiers
    tokenizer.save(os.path.join(output_dir, "vocab.json"))
    with open(os.path.join(output_dir, "identifiers.json"), "w") as f:
        json.dump(PUZZLE_TYPES, f)

    print(f"\n✅ PPBench dataset built successfully!")
    print(f"   Output: {output_dir}")
    print(f"   Vocab: {tokenizer.vocab_size} tokens (paper: 294)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build PPBench dataset for TRM training (arXiv:2605.19943)"
    )
    parser.add_argument("--output-dir", default="data/ppbench",
                        help="Output directory (default: data/ppbench)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only compute counts, don't build")
    args = parser.parse_args()

    build_ppbench_dataset(output_dir=args.output_dir, seed=args.seed, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
