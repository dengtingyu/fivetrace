"""Local-only aggregate evaluation for the three pinned recent English detectors.

Install the separate evaluation requirements in a separate virtual environment.
Use --prepare for a separate, explicit download-and-verification-only phase.
Evaluation uses verified local assets. Individual texts, labels, scores and IDs
are never uploaded or written out.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import urllib.request

import torch
from sklearn.metrics import f1_score, roc_auc_score
from transformers import AutoTokenizer

RECIPES = {'desklib': {'mode': 'author_first_768', 'threshold': 0.5, 'comparison': '>=', 'max_tokens': 768, 'padding': 'max_length', 'precision': 'float32', 'score': 'sigmoid(single mean-pooled classifier logit)'}, 'tropa': {'mode': 'author_sentence_chunks_768', 'threshold': 0.9755912190656111, 'comparison': '>=', 'max_tokens': 768, 'padding': 'longest', 'precision': 'float32', 'score': 'token-length-weighted mean of 1-P(human)', 'sentence_split': '[.!?][\\"\\\'\\)\\]]*\\s+|\\n\\s*\\n', 'minimum_words_advisory': 50}, 'conpara': {'mode': 'author_normalized_first_512', 'threshold': 'checkpoint.threshold_margin', 'comparison': '>', 'max_tokens': 512, 'precision': 'float32', 'score': 'AI logit minus human logit; author ConPara.score with Unicode normalization and whitespace collapse', 'minimum_words_advisory': 75, 'checkpoint_loading': 'weights_only=True, strict=True'}}

PINNED = {'desklib': {'repo': 'desklib/ai-text-detector-v1.01', 'revision': '5fdea974cd4287c61674951ec78803aa274e2fb7', 'year': 2025, 'directory': 'Desklib-v1.01', 'weight_sha256': 'c024a1704c65f5a4bffeda58745c58fc0ed67d6ca07b158b068a257238815265', 'weight_bytes': 1736100972, 'recipe': {'mode': 'author_first_768', 'threshold': 0.5, 'comparison': '>=', 'max_tokens': 768, 'padding': 'max_length', 'precision': 'float32', 'score': 'sigmoid(single mean-pooled classifier logit)'}, 'implementation_files': {'added_tokens.json': 'dc046d04c9b0ada7ae6f1dc89c465801799acdf0c9a6aab8c15a1b2d5ca4e91f', 'config.json': 'a9512ad54a256ca2e26bb16336cfcaa7c93d071a8667c659044003ef3973a9e2', 'README.md': 'a9d2aee79852340b6420838fe2af0864f16245c950a128c428f842bca41181ff', 'special_tokens_map.json': '9463f61e1b109a8eb4688b829260d7c6b1e6dff04c98ff7269bb89e2b92369b9', 'tokenizer.json': '5124ef2ead1a10a717703bc436de7f353da76d6340e4587719b42b1693707964', 'tokenizer_config.json': 'e6c4c771911c211618a2d46488dc7e9e499f051773cd4ff4faeca6fe55fdf569'}, 'download_files': {'added_tokens.json': {'bytes': 23, 'sha256': 'dc046d04c9b0ada7ae6f1dc89c465801799acdf0c9a6aab8c15a1b2d5ca4e91f'}, 'config.json': {'bytes': 890, 'sha256': 'a9512ad54a256ca2e26bb16336cfcaa7c93d071a8667c659044003ef3973a9e2'}, 'model.safetensors': {'bytes': 1736100972, 'sha256': 'c024a1704c65f5a4bffeda58745c58fc0ed67d6ca07b158b068a257238815265'}, 'README.md': {'bytes': 6856, 'sha256': 'a9d2aee79852340b6420838fe2af0864f16245c950a128c428f842bca41181ff'}, 'special_tokens_map.json': {'bytes': 286, 'sha256': '9463f61e1b109a8eb4688b829260d7c6b1e6dff04c98ff7269bb89e2b92369b9'}, 'spm.model': {'bytes': 2464616, 'sha256': 'c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd'}, 'tokenizer.json': {'bytes': 8656624, 'sha256': '5124ef2ead1a10a717703bc436de7f353da76d6340e4587719b42b1693707964'}, 'tokenizer_config.json': {'bytes': 1315, 'sha256': 'e6c4c771911c211618a2d46488dc7e9e499f051773cd4ff4faeca6fe55fdf569'}}}, 'tropa': {'repo': 'wasitaigeneratedcom/tropa-mini-ai-text-detector', 'revision': '10fecd0856ed65ddfe92b88416b0a1e1350d8fb8', 'year': 2026, 'directory': 'Tropa-mini', 'weight_sha256': '4a1561fadf44ec72934edd6158ff8c76e9388ade1384dbeeea6eb15f93251087', 'weight_bytes': 1736113280, 'recipe': {'mode': 'author_sentence_chunks_768', 'threshold': 0.9755912190656111, 'comparison': '>=', 'max_tokens': 768, 'padding': 'longest', 'precision': 'float32', 'score': 'token-length-weighted mean of 1-P(human)', 'sentence_split': '[.!?][\\"\\\'\\)\\]]*\\s+|\\n\\s*\\n', 'minimum_words_advisory': 50}, 'implementation_files': {'config.json': 'deac399da91c0cc96f5570fa5eae7cb716e32f6feb00860ce3e42c84f86c01a3', 'README.md': '06883a4824ed9c2f5f99be978d90cf7c33bbf4cf88e31650d0029c354ec9e2ae', 'serving_head.json': '098ce573e07d8d99221d9eac41f7acae7b324071490a06023f837efe43df4cab', 'tokenizer.json': '2c070ff353c36f85e20919da787a30e1e8e52b626d0e026e044f373378f6d0bf', 'tokenizer_config.json': 'd8089020d294f20cb21eca751b5f7f68a0b62ea2716a78576cbd587605382432'}, 'download_files': {'config.json': {'bytes': 1146, 'sha256': 'deac399da91c0cc96f5570fa5eae7cb716e32f6feb00860ce3e42c84f86c01a3'}, 'model.safetensors': {'bytes': 1736113280, 'sha256': '4a1561fadf44ec72934edd6158ff8c76e9388ade1384dbeeea6eb15f93251087'}, 'NOTICE': {'bytes': 401, 'sha256': '4fe6835a2928bcfd7467888a9422d805dfa2fe6364437c1af01cf1945c3edd57'}, 'README.md': {'bytes': 10071, 'sha256': '06883a4824ed9c2f5f99be978d90cf7c33bbf4cf88e31650d0029c354ec9e2ae'}, 'serving_head.json': {'bytes': 380, 'sha256': '098ce573e07d8d99221d9eac41f7acae7b324071490a06023f837efe43df4cab'}, 'tokenizer.json': {'bytes': 8339875, 'sha256': '2c070ff353c36f85e20919da787a30e1e8e52b626d0e026e044f373378f6d0bf'}, 'tokenizer_config.json': {'bytes': 532, 'sha256': 'd8089020d294f20cb21eca751b5f7f68a0b62ea2716a78576cbd587605382432'}}}, 'conpara': {'repo': 'SES-Lab-OTH/deberta-conpara', 'revision': 'a5f896f20f14259cf8a966fa4ae2012bd8d80164', 'year': 2026, 'directory': 'ConPara', 'weight_file': 'rawguard.pt', 'github_revision': 'b6388ca8b95b75c057f5e6dfa3ee5b7c3f92eced', 'base_revision': '64a8c8eab3e352a784c658aef62be1662607476f', 'weight_sha256': 'a0a4eaf8914471b76102703e9de25ab5e7d49a49e3d5b529036ef4f61e72ab7c', 'weight_bytes': 1738327272, 'recipe': {'mode': 'author_normalized_first_512', 'threshold': 'checkpoint.threshold_margin', 'comparison': '>', 'max_tokens': 512, 'precision': 'float32', 'score': 'AI logit minus human logit; author ConPara.score with Unicode normalization and whitespace collapse', 'minimum_words_advisory': 75, 'checkpoint_loading': 'weights_only=True, strict=True'}, 'implementation_files': {'conpara.py': 'f9358b70807b895bd6511cc8912cb22764521a5da617bd934318b86f38f35ad9', 'README.md': '5b29c9e7542ab05e82c62119affdfda2439c2a796df46e2813affdce91d87b75', 'unicode_preprocessing_v2.py': 'b93a6c43cb061d833d1f0cfca3d0e2f5000672e4a214740a34be75c4386384be', 'base/config.json': 'ddec8b81d079d218ce9e54fc0af5d1d5937d6d53b5d42e70c1f251a1cebc830d', 'base/tokenizer_config.json': '3f3978e0c036f2c2588cac34a6047cbb0af0b0dc1814254e291028529805496d'}, 'download_files': {'conpara.py': {'bytes': 8756, 'sha256': 'f9358b70807b895bd6511cc8912cb22764521a5da617bd934318b86f38f35ad9'}, 'LICENSE': {'bytes': 1069, 'sha256': '53131ec3b5b026a586fe30677d5bae0832cbc37ac299b2bced5a705daff83a53'}, 'rawguard.pt': {'bytes': 1738327272, 'sha256': 'a0a4eaf8914471b76102703e9de25ab5e7d49a49e3d5b529036ef4f61e72ab7c'}, 'README.md': {'bytes': 4983, 'sha256': '5b29c9e7542ab05e82c62119affdfda2439c2a796df46e2813affdce91d87b75'}, 'unicode_preprocessing_v2.py': {'bytes': 20135, 'sha256': 'b93a6c43cb061d833d1f0cfca3d0e2f5000672e4a214740a34be75c4386384be'}, 'base/config.json': {'bytes': 580, 'sha256': 'ddec8b81d079d218ce9e54fc0af5d1d5937d6d53b5d42e70c1f251a1cebc830d'}, 'base/spm.model': {'bytes': 2464616, 'sha256': 'c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd'}, 'base/tokenizer_config.json': {'bytes': 52, 'sha256': '3f3978e0c036f2c2588cac34a6047cbb0af0b0dc1814254e291028529805496d'}}}}


def sentence_token_chunks(tokenizer, text, max_tokens=768):
    """Sentence-aligned, disjoint original token slices; hard split long sentences.

    Use original offsets instead of decoding/re-encoding fragments so every input
    token has exactly one contribution. Add specials within the 768-token budget.
    """
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=False)
    ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    capacity = max_tokens - tokenizer.num_special_tokens_to_add(pair=False)
    if not ids:
        raise ValueError('No scoreable tokens')
    boundaries = [m.end() for m in re.finditer(RECIPES['tropa']['sentence_split'], text)]
    # A boundary lies after the last token ending before its whitespace ends.
    cuts = set()
    j = 0
    for boundary in boundaries:
        while j < len(offsets) and offsets[j][1] <= boundary:
            j += 1
        if j:
            cuts.add(j)
    start = 0
    chunks = []
    while start < len(ids):
        end = min(start + capacity, len(ids))
        if end < len(ids):
            candidates = [p for p in cuts if start < p <= end]
            if candidates:
                end = max(candidates)
        # Transformers 5's fast tokenizer no longer exposes the legacy helper.
        # Both pinned DeBERTa tokenizers use the same verified CLS/SEP template.
        assert tokenizer.num_special_tokens_to_add(pair=False) == 2
        sequence = [tokenizer.cls_token_id] + ids[start:end] + [tokenizer.sep_token_id]
        assert len(sequence) <= max_tokens
        chunks.append((sequence, end - start))
        start = end
    assert sum(n for _, n in chunks) == len(ids)
    return chunks, len(ids)


def load_pooled_model(directory, labels, device='cuda'):
    import torch
    from torch import nn
    from transformers import AutoConfig, AutoModel
    from safetensors.torch import load_file

    class PooledDetector(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = AutoModel.from_config(AutoConfig.from_pretrained(directory, local_files_only=True))
            self.classifier = nn.Linear(self.model.config.hidden_size, labels)

        def forward(self, input_ids, attention_mask):
            hidden = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
            return self.classifier(pooled)

    model = PooledDetector()
    model.load_state_dict(load_file(str(directory / 'model.safetensors')), strict=True)
    return model.float().to(device).eval()


def pooled_probabilities(logits, name):
    import torch
    if name == 'desklib':
        assert logits.shape[-1] == 1
        return torch.sigmoid(logits[:, 0])
    assert name == 'tropa' and logits.shape[-1] == 4
    # The author calls .item() before Python's 1.0 - P(human). Preserve that
    # double-precision subtraction to avoid extra FP32 ties in AUROC scores.
    return 1.0 - torch.softmax(logits, dim=-1)[:, 0].double()


def load_numeric_checkpoint(path):
    """Restricted PyTorch loading with explicit numeric NumPy metadata types."""
    import torch
    import numpy as np
    # The official checkpoint includes numeric NumPy scaler/index metadata.
    # Permit only its numeric-array reconstruction types, keeping arbitrary
    # pickle globals and unrestricted weights_only=False disabled.
    core=np._core if hasattr(np,'_core') else np.core
    safe=[np.ndarray,np.dtype,
          (core.multiarray._reconstruct,'numpy.core.multiarray._reconstruct'),
          (core.multiarray._reconstruct,'numpy._core.multiarray._reconstruct')]
    safe.extend({type(np.dtype(t)) for t in [np.float32,np.float64,np.int32,np.int64,np.uint32,np.uint64,np.bool_]})
    with torch.serialization.safe_globals(safe):
        return torch.load(path,map_location='cpu',weights_only=True,mmap=True)


def load_author_conpara(directory, device='cuda'):
    """Unmodified author scoring/normalization, pinned local Base configuration.

    The loading path uses safe torch.load rather than the author's unrestricted
    pickle default; the network, preprocessing and decision rule are unchanged.
    """
    from transformers import AutoTokenizer
    import os
    for module_name, filename in [('unicode_preprocessing_v2','unicode_preprocessing_v2.py'),
                                   ('fivetrace_pinned_conpara','conpara.py')]:
        spec=importlib.util.spec_from_file_location(module_name,directory/filename)
        module=importlib.util.module_from_spec(spec)
        sys.modules[module_name]=module
        spec.loader.exec_module(module)
    checkpoint=load_numeric_checkpoint(directory/'rawguard.pt')
    state=checkpoint['model_state_dict']
    assert not any(k.startswith('feat_proj') for k in state)
    # The pinned tokenizer resolver mishandles absolute non-ASCII Windows paths.
    # A relative path addresses the identical verified files without changing IDs.
    try:base=os.path.relpath(directory/'base',Path.cwd())
    except ValueError:base=str(directory/'base')
    model=module.NoFeat(backbone=base,config_only=True)
    model.load_state_dict(state,strict=True)
    model.float().to(device).eval()
    tokenizer=AutoTokenizer.from_pretrained(base,use_fast=True,local_files_only=True)
    threshold=float(checkpoint['threshold_margin'])
    assert threshold==2.9423828125
    detector=module.ConPara(model=model,tokenizer=tokenizer,threshold=threshold,device=device,meta={})
    return module,detector

def file_sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_checkpoint(name, directory):
    pin = PINNED[name]
    path = directory / pin.get('weight_file','model.safetensors')
    if path.stat().st_size != pin['weight_bytes'] or file_sha(path) != pin['weight_sha256']:
        raise ValueError('Checkpoint does not match the pinned public revision.')
    for filename, metadata in pin['download_files'].items():
        if file_sha(directory / filename) != metadata['sha256']:
            raise ValueError('Model configuration/code differs from the pinned author release: ' + filename)


def prepare_local(name, directory):
    """Explicit download-only phase; do not execute downloaded native code."""
    from huggingface_hub import hf_hub_download
    pin=PINNED[name]
    directory.mkdir(parents=True,exist_ok=True)
    for filename, metadata in pin['download_files'].items():
        path=directory/filename
        if path.exists() and file_sha(path)==metadata['sha256']:
            continue
        if name=='conpara' and filename in ['conpara.py','unicode_preprocessing_v2.py','LICENSE']:
            relative=filename if filename=='LICENSE' else 'src/'+filename
            url='https://raw.githubusercontent.com/SES-Lab-OTH/deberta-conpara/'+pin['github_revision']+'/'+relative
            with urllib.request.urlopen(url,timeout=60) as response:
                content=response.read(metadata['bytes']+1)
            if len(content)!=metadata['bytes'] or hashlib.sha256(content).hexdigest()!=metadata['sha256']:
                raise ValueError('Pinned GitHub source hash mismatch: '+filename)
            path.write_bytes(content)
        else:
            base=name=='conpara' and filename.startswith('base/')
            repo='microsoft/deberta-v3-large' if base else pin['repo']
            revision=pin['base_revision'] if base else pin['revision']
            target=directory/'base' if base else directory
            hf_hub_download(repo,filename=filename.removeprefix('base/') if base else filename,
                            revision=revision,local_dir=target)
        if path.stat().st_size!=metadata['bytes'] or file_sha(path)!=metadata['sha256']:
            raise ValueError('Downloaded file does not match pinned artifact: '+filename)
        print(json.dumps(dict(verified=filename)),flush=True)


def aggregate_metrics(labels, scores, decisions):
    humans = [i for i, y in enumerate(labels) if y == 0]
    ais = [i for i, y in enumerate(labels) if y == 1]
    both = bool(humans and ais)
    return dict(records=len(labels), human=len(humans), ai=len(ais),
                human_FPR=sum(decisions[i] for i in humans)/len(humans) if humans else None,
                AI_recall=sum(decisions[i] for i in ais)/len(ais) if ais else None,
                macro_F1=float(f1_score(labels, decisions, average='macro')) if both else None,
                AUROC=float(roc_auc_score(labels, scores)) if both else None)


def evaluate_local(name, directory, source, device):
    verify_checkpoint(name, directory)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    recipe = RECIPES[name]
    tokenizer = None
    if name == 'conpara':
        module, model = load_author_conpara(directory, device=device)
        tokenizer = model.tokenizer
    else:
        model = load_pooled_model(directory, 1 if name == 'desklib' else 4, device=device)
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    labels, scores, decisions = [], [], []
    skipped_short = skipped_language = capped = below_recommendation = 0
    with source.open(encoding='utf-8') as stream, torch.inference_mode():
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            text, label = record['text'], record['label']
            if not isinstance(text, str) or type(label) is not int or label not in (0, 1):
                raise ValueError('Each JSONL record needs a string text and integer label 0/1.')
            if record.get('language', 'en') != 'en':
                skipped_language += 1
                continue
            if sum(not c.isspace() for c in text) < 100:
                skipped_short += 1
                continue
            if name == 'conpara':
                score=float(model.score([text],batch_size=1,normalise_input=True)[0])
                decision=int(score>model.threshold)
                capped+=int(len(tokenizer(module.normalise(text),add_special_tokens=True,truncation=False)['input_ids'])>512)
                below_recommendation+=int(len(text.split())<75)
            elif name == 'desklib':
                encoded = tokenizer(text, padding='max_length', max_length=768, truncation=True, return_tensors='pt')
                logits = model(encoded['input_ids'].to(device), encoded['attention_mask'].to(device))
                score = float(pooled_probabilities(logits, name)[0])
                decision = int(score >= recipe['threshold'])
                capped += int(len(tokenizer(text, add_special_tokens=True, truncation=False)['input_ids']) > 768)
            else:
                chunks, total = sentence_token_chunks(tokenizer, text)
                weighted = 0.
                for ids, length in chunks:
                    inp = torch.tensor([ids], device=device)
                    logits = model(inp, torch.ones_like(inp))
                    weighted += float(pooled_probabilities(logits, name)[0]) * length
                score = weighted / total
                decision = int(score >= recipe['threshold'])
                below_recommendation += int(len(text.split()) < 50)
            if not math.isfinite(score):
                raise ValueError('Nonfinite model score')
            labels.append(label); scores.append(score); decisions.append(decision)
            if len(labels) % 100 == 0:
                print(json.dumps(dict(processed=len(labels))), file=sys.stderr, flush=True)
    return dict(detector=name, checkpoint=PINNED[name]['repo'], revision=PINNED[name]['revision'],
                recipe=recipe, device=device, **aggregate_metrics(labels, scores, decisions),
                skipped_short=skipped_short, skipped_non_english=skipped_language,
                capped_inputs=capped, below_word_recommendation=below_recommendation,
                note='Author/native thresholds, not recalibrated on these inputs. Scores are evidence, not authorship proof. No per-record output.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detector', choices=list(PINNED), required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--prepare',action='store_true',help='Download and verify pinned model files only; do not run inference.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.prepare:
        prepare_local(args.detector,args.model_dir)
        return
    if args.input is None or args.output is None:
        parser.error('--input and --output are required for evaluation')
    result = evaluate_local(args.detector, args.model_dir, args.input, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
