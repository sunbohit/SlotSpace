"""GPU-only offline paraphrasing; importable without installing vLLM."""

import argparse
from itertools import islice
import json
from pathlib import Path

from ..artifacts import logged_run, new_directory, read_jsonl, sha256, write_json

PROMPT = (
    "Rewrite the sentence below in 3 different ways. Keep its meaning. "
    "Keep each of these words exactly as written, each exactly once: {tokens}. "
    "Vary the wording and the sentence structure. Return one rewrite per line, "
    "with no numbering and no extra text.\n\nSentence: {text}"
)


def paraphrase_batch(samples: list[dict], llm, params) -> list[dict]:
    messages = [[{"role": "user", "content": PROMPT.format(
        tokens=", ".join(sample["tokens"]), text=sample["text"])}] for sample in samples]
    outputs = llm.chat(messages, params)
    if len(outputs) != len(samples):
        raise ValueError("Paraphraser returned a different number of samples")
    result = []
    for sample, output in zip(samples, outputs):
        if not output.outputs:
            raise RuntimeError(f"No paraphrase completion for {sample['sample_id']}")
        lines = [line.strip() for line in output.outputs[0].text.splitlines() if line.strip()]
        result.append({**sample, "rewrites": lines})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    from vllm import LLM, SamplingParams

    config = {"input_sha256": sha256(args.input), "model": args.model, "seed": 13,
              "temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_tokens": 256, "max_model_len": 4096}
    with logged_run(Path("runs"), "m5-paraphrase", config):
        with new_directory(args.out) as stage:
            llm = LLM(model=args.model, max_model_len=4096, seed=13)
            params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=256, seed=13)
            source, count = iter(read_jsonl(args.input)), 0
            with (stage / "paraphrases.jsonl").open("x", encoding="utf-8", newline="\n") as dest:
                while batch := list(islice(source, args.batch_size)):
                    for row in paraphrase_batch(batch, llm, params):
                        dest.write(json.dumps(row, ensure_ascii=False) + "\n")
                        count += 1
                    print(f"{count} original samples paraphrased", flush=True)
            write_json(stage / "manifest.json", {**config, "samples": count})


if __name__ == "__main__":
    main()
