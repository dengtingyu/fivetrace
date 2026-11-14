"""Actually train class-conditional causal-LM LoRA adapters on disjoint data."""
import gc
import json
import math
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from ai_detector.config import ARTIFACT_DIR, ROOT, SEED
from ai_detector.expanded_data import PREPARED, clean_text, read_units, sample_units

OUT = ARTIFACT_DIR / "five_feature" / "Qwen3-1.7B-LoRA-v2"
MODEL = ROOT / "models" / "Qwen3-1.7B-Base"


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def train_adapter(name, train_rows, dev_rows, tokenizer, batch_size=2, accumulation=4):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    directory = OUT / "adapters" / name
    directory.mkdir(parents=True, exist_ok=True)
    selected = {"train": [{k: v for k, v in r.items() if k != "text"} for r in train_rows],
                "dev": [{k: v for k, v in r.items() if k != "text"} for r in dev_rows]}
    save_json(directory / "sample_manifest.json", selected)

    def encode(row):
        ids = tokenizer.encode(clean_text(row["text"]), add_special_tokens=False)
        if len(ids) > 256:
            rng = random.Random(row["id"])
            start = rng.randrange(len(ids)-256+1)
            ids = ids[start:start+256]
        return ids

    train_ids, dev_ids = [list(map(encode, values)) for values in [train_rows, dev_rows]]

    def collate(examples):
        length = max(len(x) for x in examples)
        inputs = torch.full((len(examples), length), tokenizer.eos_token_id, dtype=torch.long)
        masks = torch.zeros_like(inputs)
        labels = torch.full_like(inputs, -100)
        for i, ids in enumerate(examples):
            inputs[i, :len(ids)] = torch.tensor(ids)
            labels[i, :len(ids)] = torch.tensor(ids)
            masks[i, :len(ids)] = 1
        return {"input_ids": inputs.cuda(), "attention_mask": masks.cuda(), "labels": labels.cuda()}

    loader = DataLoader(train_ids, batch_size=batch_size, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(SEED), num_workers=0)
    dev_loader = DataLoader(dev_ids, batch_size=batch_size, collate_fn=collate, num_workers=0)
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), local_files_only=True,
                torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
                        target_modules=["q_proj", "v_proj"], lora_dropout=.05, bias="none")
    model = get_peft_model(model, config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n, p in trainable)
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=2e-4, weight_decay=.01)
    steps = math.ceil(len(loader)/accumulation)
    warmup = max(1, steps//10)
    history, best, best_step = [], float("inf"), 0
    started = time.monotonic()
    trained_tokens = 0

    def evaluate():
        model.eval()
        total_loss, total_tokens = 0., 0
        with torch.inference_mode():
            for batch in dev_loader:
                tokens = int((batch["labels"][:, 1:] != -100).sum())
                loss = float(model(**batch).loss)
                total_loss += loss*tokens
                total_tokens += tokens
        model.train()
        return total_loss/total_tokens

    initial_dev = evaluate()
    print(f"{name}: initial dev NLL={initial_dev:.4f}; {len(train_rows)} training units; {steps} optimizer steps", flush=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running, running_batches, step = 0., 0, 0
    torch.cuda.reset_peak_memory_stats()
    for batch_index, batch in enumerate(loader, start=1):
        loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss: {name}, batch={batch_index}")
        (loss/accumulation).backward()
        running += float(loss.detach())
        running_batches += 1
        trained_tokens += int((batch["labels"][:, 1:] != -100).sum())
        if batch_index % accumulation and batch_index != len(loader):
            continue
        step += 1
        torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.)
        lr = 2e-4*(step/warmup if step <= warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1, steps-warmup))))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        record = {"adapter": name, "step": step, "total_steps": steps, "train_nll": running/running_batches,
                  "lr": lr, "elapsed_seconds": time.monotonic()-started, "trained_tokens": trained_tokens}
        if step % 10 == 0 or step == steps:
            print(json.dumps(record), flush=True)
        if step % 75 == 0 or step == steps:
            record["dev_nll"] = evaluate()
            if record["dev_nll"] < best:
                best, best_step = record["dev_nll"], step
                model.save_pretrained(str(directory), safe_serialization=True)
            print(f"{name}: step {step}, dev NLL={record['dev_nll']:.4f}, best={best:.4f}", flush=True)
        history.append(record)
        save_json(OUT / "training_state.json", {"stage": "lora_training", "updated_at": now(), **record})
        running, running_batches = 0., 0
    change_norm = sum(float(p.detach().float().norm()) for n, p in trainable if "lora_B" in n)
    assert change_norm > 0
    result = {"adapter": name, "train_units": len(train_rows), "dev_units": len(dev_rows),
              "trainable_parameters": sum(p.numel() for _, p in trainable), "initial_dev_nll": initial_dev,
              "best_dev_nll": best, "best_step": best_step, "total_steps": steps, "trained_tokens": trained_tokens,
              "nonzero_lora_B_norm_sum": change_norm, "peak_cuda_GB": torch.cuda.max_memory_allocated()/1024**3,
              "elapsed_seconds": time.monotonic()-started, "history": history,
              "config": {"rank": 8, "alpha": 16, "target_modules": ["q_proj", "v_proj"],
                         "max_tokens": 256, "batch_size": batch_size, "gradient_accumulation": accumulation,
                         "dtype": "bfloat16", "epochs": 1, "peak_learning_rate": 2e-4},
              "objective": "teacher-forced causal LM on class-specific real text; no label words or classifier prompts in LM targets"}
    save_json(directory / "training_metrics.json", result)
    del optimizer, model, trainable, loader, dev_loader
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    rows = read_units()
    train = sample_units(rows, "lora_train", per_class=3000)
    dev = sample_units(rows, "lora_dev", per_class=160)
    assert not ({r["group"] for r in train} & {r["group"] for r in dev})
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    results = []
    for name, label in [("ai", 1), ("human", 0)]:
        complete = OUT / "adapters" / name / "training_metrics.json"
        if complete.exists():
            previous = json.loads((complete.parent / "sample_manifest.json").read_text(encoding="utf-8"))
            assert [r["id"] for r in previous["train"]] == [r["id"] for r in train if r["label"] == label], "Existing adapter was trained on different sample IDs"
            assert [r["id"] for r in previous["dev"]] == [r["id"] for r in dev if r["label"] == label], "Existing adapter used different dev IDs"
            results.append(json.loads(complete.read_text(encoding="utf-8")))
            print(f"{name}: reusing completed adapter", flush=True)
            continue
        results.append(train_adapter(name, [r for r in train if r["label"] == label],
                                     [r for r in dev if r["label"] == label], tokenizer))
    save_json(OUT / "lora_training.json", {"completed_at": now(), "adapters": results,
              "data_manifest": str(PREPARED / "manifest.json"), "external_test_used_for_training": False})
    save_json(OUT / "training_state.json", {"stage": "lora_completed", "updated_at": now()})
    print("Both actual LoRA adapters trained and saved", flush=True)


if __name__ == "__main__":
    main()
