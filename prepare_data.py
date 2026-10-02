"""Prepare local 1M/10M training files and shared validation/test splits from FineWeb."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer


TRAIN_SEQUENCES = {'1M': 488, '10M': 4880}
HELD_OUT_SEQUENCES = 544


def encode_split(documents, tokenizer, min_tokens):
    tokens = []
    end = tokenizer.token_to_id('<|endoftext|>')
    while len(tokens) < min_tokens:
        tokens.append(end)
        tokens.extend(tokenizer.encode(next(documents)['text']).ids)
    usable = len(tokens) // 2049 * 2049
    return torch.tensor(tokens[:usable], dtype=torch.int32).view(-1, 2049)


def save_split(path, rows, count):
    if len(rows) < count:
        raise ValueError(f'{path}: need {count} sequences, found {len(rows)}')
    # Clone the prefix so torch.save does not serialize the larger pool's storage.
    selected = rows[:count].clone()
    if path.exists():
        previous = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(previous, torch.Tensor) or not torch.equal(previous, selected):
            raise FileExistsError(f'{path} contains different data; choose a new --output directory.')
    else:
        torch.save(selected, path)
    print(f'{path}: {count} sequences, {count * 2048:,} prediction tokens', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--tokens', choices=['1M', '10M', 'both'], default='both',
        help='Training files to create; validation and test are shared.',
    )
    parser.add_argument('--output', type=Path, default=Path('data'))
    parser.add_argument('--parquet', nargs='+', help='Optional local FineWeb shards, in original stream order.')
    args = parser.parse_args()

    from datasets import load_dataset

    args.output.mkdir(parents=True, exist_ok=True)
    if args.parquet:
        corpus = load_dataset('parquet', data_files=args.parquet, split='train', streaming=True)
    else:
        corpus = load_dataset(
            'HuggingFaceFW/fineweb', name='sample-10BT', split='train', streaming=True,
        )

    tokenizer_path = Path(__file__).parent / 'tokenizer' / 'tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    documents = iter(corpus)
    for _ in range(200_000):
        next(documents)

    # For the paper experiments, validation and test documents followed a
    # ~20.5M-token training pool. The 1M and 10M runs used prefixes of that pool.
    # Read through the same pool here so validation and test start at the same
    # documents, but save only the requested training prefixes below.
    training = encode_split(documents, tokenizer, 20_500_000)
    budgets = TRAIN_SEQUENCES if args.tokens == 'both' else {args.tokens: TRAIN_SEQUENCES[args.tokens]}
    for budget, count in budgets.items():
        save_split(args.output / f'train_{budget}.pt', training, count)
    del training

    for split in ('val', 'test'):
        rows = encode_split(documents, tokenizer, 1_150_000)
        save_split(args.output / f'{split}.pt', rows, HELD_OUT_SEQUENCES)

    files = {}
    for name in ('train_1M.pt', 'train_10M.pt', 'val.pt', 'test.pt'):
        path = args.output / name
        if path.exists():
            rows = torch.load(path, map_location='cpu', weights_only=True)
            files[name] = dict(
                sequences=len(rows),
                prediction_tokens=len(rows) * 2048,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )

    metadata = dict(
        corpus='HuggingFaceFW/fineweb',
        subset='sample-10BT',
        skipped_documents=200_000,
        tokenizer_sha256=hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        files=files,
    )
    (args.output / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
