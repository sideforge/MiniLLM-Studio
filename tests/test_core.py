"""Schnelle Tests für Tokenizer, Modell und Trainings-Schritt (kein Internet nötig)."""
import numpy as np
import torch

from minillm.generate import filter_logits, generate_text
from minillm.model import GPT, GPTConfig
from minillm.tokenizer import EOT, BPETokenizer

CORPUS = ["the cat sat on the mat. the dog sat on the log.\n" * 20,
          "    def add(a, b):\n        return a + b\n" * 10, "Grüezi! Äpfel & Öl 123 🙂"]


def test_tokenizer_roundtrip():
    tok = BPETokenizer.train(CORPUS, 320)
    assert 257 < tok.vocab_size <= 320   # stoppt früher, wenn kein Paar mehr 2x vorkommt
    for text in CORPUS + ["völlig neuer Text mit Ümläuten 🙂", "", f"a{EOT}b"]:
        assert tok.decode(tok.encode(text)) == text


def test_tokenizer_compresses():
    tok = BPETokenizer.train(CORPUS, 400)
    text = CORPUS[0]
    assert len(tok.encode(text)) < len(text.encode()) / 2


def test_tokenizer_save_load(tmp_path):
    tok = BPETokenizer.train(CORPUS, 300)
    tok.save(tmp_path / "t.json")
    tok2 = BPETokenizer.load(tmp_path / "t.json")
    assert tok.encode(CORPUS[0]) == tok2.encode(CORPUS[0])


def small_model():
    return GPT(GPTConfig(vocab_size=300, block_size=16, n_layer=2, n_head=2, n_embd=32))


def test_model_shapes_and_params():
    cfg = GPTConfig(vocab_size=300, block_size=16, n_layer=2, n_head=2, n_embd=32)
    m = GPT(cfg)
    x = torch.randint(0, 300, (3, 16))
    logits, loss = m(x, x)
    assert logits.shape == (3, 16, 300)
    assert abs(loss.item() - np.log(300)) < 0.5          # ungelernt: Loss ≈ ln(Vokabular)
    assert m.num_params() == GPT.estimate_params(cfg)
    assert sum(m.param_breakdown().values()) == m.num_params()


def test_attention_is_causal():
    m = small_model().eval()
    x = torch.randint(0, 300, (1, 10))
    _, _, atts = m(x, return_attn=True)
    for a in atts:
        assert torch.allclose(a.sum(-1), torch.ones(1, 2, 10), atol=1e-5)   # Zeilen summieren zu 1
        assert torch.triu(a[0, 0], diagonal=1).sum() == 0                  # Zukunft ist maskiert


def test_future_tokens_do_not_change_past_logits():
    m = small_model().eval()
    a = torch.randint(0, 300, (1, 12))
    b = a.clone()
    b[0, 8:] = (b[0, 8:] + 1) % 300
    la, _ = m(a)
    lb, _ = m(b)
    assert torch.allclose(la[0, :8], lb[0, :8], atol=1e-5)


def test_model_can_overfit_one_batch():
    torch.manual_seed(0)
    m = small_model()
    opt = m.configure_optimizer(1e-2, 0.0)
    x = torch.randint(0, 300, (4, 16))
    first = None
    for _ in range(60):
        _, loss = m(x, x.roll(-1, 1))
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.3


def test_sampling_filters():
    logits = torch.tensor([5.0, 4.0, 1.0, 0.0, -1.0])
    p = filter_logits(logits, 1.0, 2, 1.0)
    assert (p > 0).sum() == 2 and abs(p.sum().item() - 1) < 1e-6
    p = filter_logits(logits, 1.0, 0, 0.5)
    assert p[0] > 0 and p[4] == 0
    assert filter_logits(logits, 0.1, 0, 1.0)[0] > 0.99


def test_generate_text_runs():
    tok = BPETokenizer.train(CORPUS, 300)
    m = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=16, n_layer=1, n_head=2, n_embd=32))
    out = generate_text(m, tok, "the cat", max_new_tokens=10, seed=1)
    assert out.startswith("the cat")


# ----------------------------------------------------------------- Gerät / GPU
def test_device_selection_falls_back_to_cpu():
    from minillm import device as dev
    assert dev.pick_device("cpu").type == "cpu"
    d = dev.pick_device("auto")                      # auf einem Rechner ohne GPU: CPU, sonst die GPU
    assert d.type in ("cpu", "cuda", "xpu", "mps")
    assert dev.describe(torch.device("cpu"))["type"] == "cpu"
    assert dev.amp_dtype(torch.device("cpu")) is None
    with dev.autocast(torch.device("cpu"), True):    # auf der CPU ein No-Op
        pass


def test_find_max_batch_on_cpu_returns_start():
    from minillm import device as dev
    assert dev.find_max_batch(small_model, 16, 300, torch.device("cpu"), start=16) == 16


def test_gpu_batcher_indexing_matches_cpu_path(tmp_path):
    """Der GPU-Weg (Daten als Tensor, Fenster per Index) muss dasselbe liefern wie der CPU-Weg (hier auf CPU simuliert)."""
    from minillm.data import Batcher
    ids = (np.arange(500) % 251).astype(np.uint16)
    ids.tofile(tmp_path / "train.bin")
    ids[:100].tofile(tmp_path / "val.bin")
    b = Batcher(tmp_path, 16)
    b._dev = {"train": torch.from_numpy(b.train.astype(np.int32)), "val": torch.from_numpy(b.val.astype(np.int32))}
    b._arange = torch.arange(16)
    x, y = b.get("train", 8)
    assert x.shape == y.shape == (8, 16) and x.dtype == torch.long
    assert torch.equal(x[:, 1:], y[:, :-1])          # y ist x um ein Token verschoben


def test_short_training_run_with_auto_batch(tmp_path, monkeypatch):
    """Ganzer Trainer-Durchlauf (CPU), inkl. Auto-Batch und gepufferten Metriken."""
    import time
    from minillm import trainer as trainer_mod
    from minillm.data import prepare
    monkeypatch.setattr(trainer_mod, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr("minillm.data.PREPARED_DIR", tmp_path / "prepared")
    docs = {"x": [CORPUS[0], CORPUS[1], CORPUS[2]] * 12}
    meta = prepare("t", docs, 300)
    cfg = GPTConfig(vocab_size=meta["vocab_size"], block_size=16, n_layer=1, n_head=2, n_embd=32)
    tr = trainer_mod.Trainer("t", tmp_path / "prepared" / "t", cfg,
                             trainer_mod.TrainConfig(batch_size=0, max_steps=30, eval_interval=10, sample_interval=15,
                                                     eval_iters=2, device="cpu"))
    tr.start()
    while tr.is_running:
        time.sleep(0.2)
    assert tr.status == "fertig", tr.error
    assert tr.cfg.batch_size == 16                    # Auto auf der CPU
    assert [m["step"] for m in tr.metrics] == list(range(1, 31))
    assert (tmp_path / "runs" / "t" / "model.pt").exists()


def test_gpu_helpers_degrade_gracefully_without_gpu():
    from minillm import device as dev
    cpu = torch.device("cpu")
    assert dev.can_compile(cpu) is False
    if not torch.cuda.is_available():
        assert dev.gpu_stats() is None or isinstance(dev.gpu_stats(), dict)


def test_suggest_model_scales_with_gpu_memory():
    from minillm import device as dev
    sizes = [GPT.estimate_params(GPTConfig(vocab_size=2048, **{k.replace("m_", "n_").replace("n_embd", "n_embd"): v
                                                                for k, v in dev.suggest_model(m).items()
                                                                if k != "m_block"}, block_size=dev.suggest_model(m)["m_block"]))
             for m in (4, 8, 12, 24, 48)]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]
    for m in (4, 8, 12, 24, 48):
        d = dev.suggest_model(m)
        assert d["m_embd"] % d["m_head"] == 0


def naive_bpe(texts, n_merges):
    """Langsame, aber offensichtlich richtige Referenz: zählt vor jedem Merge alle Paare neu."""
    from collections import Counter
    from minillm.tokenizer import _PRETOKEN
    wc = Counter(w for t in texts for w in _PRETOKEN.findall(t))
    words = {w: list(w.encode()) for w in wc}
    merges = []
    for m in range(n_merges):
        pc = Counter()
        for w, ids in words.items():
            for p in zip(ids, ids[1:]):
                pc[p] += wc[w]
        if not pc:
            break
        best = max(pc.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]
        if pc[best] < 2:
            break
        merges.append(best)
        for w, ids in words.items():
            out, i = [], 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best:
                    out.append(256 + m)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            words[w] = out
    return merges


def test_heap_bpe_matches_naive_reference():
    import random
    rng = random.Random(7)
    syll = ["ka", "to", "mi", "ra", "ne", "so", "lu", "pe", "ti", "an", "er", "st", "ch", "aa", "bb"]
    texts = [" ".join("".join(rng.choice(syll) for _ in range(rng.randint(1, 4))) for _ in range(30)) for _ in range(40)]
    texts += ["aaa aaaa aaaaa aaaaaa", "def f(x):\n    return x\n" * 5]
    tok = BPETokenizer.train(texts, 256 + 120 + 1)
    assert tok.merges == naive_bpe(texts, 120)


# ---------------------------------------------------------- Eigene Datensätze
def test_render_value_handles_chat_lists_and_nesting():
    from minillm.catalog import render_value
    chat = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hallo"}]
    assert render_value(chat) == "### Frage:\nHi\n### Antwort:\nHallo"
    sharegpt = [{"from": "human", "value": "A"}, {"from": "gpt", "value": "B"}]
    assert "### Frage:\nA" in render_value(sharegpt) and "### Antwort:\nB" in render_value(sharegpt)
    assert render_value(None) == "" and render_value(5) == "5" and render_value(["a", "b"]) == "a\nb"


def test_detect_columns_prefers_text_chat_and_qa_pairs():
    from minillm.custom import detect_columns
    assert detect_columns({"id": "1", "text": "hello world, a long text", "url": "x"}) == ["text"]
    assert detect_columns({"prompt_id": "x", "messages": [{"role": "user", "content": "a"}]}) == ["messages"]
    assert detect_columns({"instruction": "do it", "output": "done", "id": 3}) == ["instruction", "output"]
    assert detect_columns({"id": "abc", "story_body": "x" * 50, "title": "t"}) in (["story_body"], ["title"])
    assert detect_columns({"n": 1}) == []


def test_custom_dataset_roundtrip(tmp_path, monkeypatch):
    from minillm import custom
    from minillm.catalog import BY_KEY, CATALOG
    monkeypatch.setattr(custom, "CUSTOM_FILE", tmp_path / "custom.json")
    n = len(CATALOG)
    info = custom.add_custom("someone/some-data", ["question", "answer"], None, "train", headings=True, max_chars=500)
    try:
        assert info.key in BY_KEY and len(CATALOG) == n + 1 and info.category == "Eigene"
        assert info.fmt({"question": "Q?", "answer": "A!", "x": 1}) == "### question:\nQ?\n### answer:\nA!"
        # erneutes Hinzufügen ersetzt statt zu verdoppeln
        custom.add_custom("someone/some-data", ["question"], None, "train")
        assert len(CATALOG) == n + 1 and len(custom._read()) == 1
    finally:
        custom.remove_custom(info.key)
    assert info.key not in BY_KEY and len(CATALOG) == n and custom._read() == []


def test_friendly_errors():
    from minillm.custom import friendly_error
    assert "Lade-Skript" in friendly_error(RuntimeError("Dataset scripts are no longer supported, but found x.py"))
    assert "gesperrt" in friendly_error(Exception("Dataset 'a/b' is a gated dataset on the Hub. You must be authenticated"))
    assert "nicht gefunden" in friendly_error(Exception("Dataset 'a/b' doesn't exist on the Hub"))


def test_catalog_entries_are_complete():
    from minillm.catalog import CATALOG
    keys = [d.key for d in CATALOG]
    assert len(keys) == len(set(keys)), "doppelte Schlüssel im Katalog"
    assert any(d.category == "Code" for d in CATALOG) and len(CATALOG) >= 40


def test_saved_custom_datasets_load_in_both_import_orders(tmp_path):
    """Regression: gespeicherte eigene Datensätze müssen nach einem Neustart im Katalog sein, egal was zuerst importiert wird."""
    import json
    import os
    import subprocess
    import sys
    (tmp_path / "custom_datasets.json").write_text(json.dumps([{
        "key": "custom_x", "repo": "a/b", "config": None, "split": "train", "columns": ["text"],
        "headings": False, "min_chars": 1, "max_chars": None}]), encoding="utf-8")
    env = {**os.environ, "MINILLM_DATA_DIR": str(tmp_path)}
    for first in ("from minillm import custom", "import minillm.catalog"):
        code = first + "\nfrom minillm.catalog import BY_KEY\nprint('custom_x' in BY_KEY)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                             cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        assert out.stdout.strip() == "True", (first, out.stdout, out.stderr[-300:])


# ------------------------------------------- Datensätze mit wechselndem Schema (CastError)
def test_render_value_message_blocks():
    from minillm.catalog import render_value
    msg = {"role": "assistant", "content": [{"type": "text", "text": "Hallo!"},
                                            {"type": "tool_use", "name": "Bash", "input": {"cmd": "ls"}}]}
    out = render_value(msg)
    assert out.startswith("### Antwort:\nHallo!") and "[Werkzeug: Bash]" in out
    assert render_value({"role": "user", "content": "Frage?"}) == "### Frage:\nFrage?"
    assert render_value([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a\nb"


def test_detect_columns_finds_message_dict():
    from minillm.custom import detect_columns
    row = {"parentUuid": None, "cwd": "/x", "sessionId": "abc", "type": "user",
           "message": {"role": "user", "content": "hello"}}
    assert detect_columns(row) == ["message"]


def _fake_info(key="t_schema"):
    from minillm.catalog import DatasetInfo
    return DatasetInfo(key, "t", "a/b", "", "Eigene", "en", lambda r: r["text"], min_chars=1)


def test_iter_texts_falls_back_to_raw_json_on_schema_error(monkeypatch):
    from minillm import catalog

    def fake_rows(info, raw):
        if raw:
            yield from ({"text": f"raw{i}"} for i in range(5))
        else:
            raise ValueError("CastError: Couldn't cast parentUuid: string ...")
            yield   # pragma: no cover
    monkeypatch.setattr(catalog, "_rows", fake_rows)
    assert list(catalog.iter_texts(_fake_info(), 3)) == ["raw0", "raw1", "raw2"]


def test_fetch_restarts_cleanly_when_schema_changes_midway(monkeypatch, tmp_path):
    from minillm import catalog
    info = _fake_info()
    monkeypatch.setitem(catalog.BY_KEY, info.key, info)
    monkeypatch.setattr(catalog, "RAW_DIR", tmp_path)

    def fake_rows(info_, raw):
        if raw:
            yield from ({"text": f"raw{i}"} for i in range(10))
        else:
            yield {"text": "normal0"}
            yield {"text": "normal1"}
            raise ValueError("CastError: Couldn't cast column 3")
    monkeypatch.setattr(catalog, "_rows", fake_rows)
    path = catalog.fetch_dataset(info.key, 4)
    docs = catalog.read_documents(path)
    assert docs == ["raw0", "raw1", "raw2", "raw3"]      # keine Reste vom abgebrochenen Versuch


def test_other_errors_are_not_swallowed(monkeypatch):
    import pytest
    from minillm import catalog

    def boom(info, raw):
        raise ConnectionError("Netzwerk weg")
        yield   # pragma: no cover
    monkeypatch.setattr(catalog, "_rows", boom)
    with pytest.raises(ConnectionError):
        list(catalog.iter_texts(_fake_info(), 3))
