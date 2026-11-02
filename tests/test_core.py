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
