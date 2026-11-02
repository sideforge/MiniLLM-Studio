"""Training ohne App (Kommandozeile).

Beispiel:
    python train_cli.py --datasets tinystories=2000 shakespeare=200 --steps 500 --name test
"""
from __future__ import annotations

import argparse
import time

from minillm.catalog import BY_KEY
from minillm.data import load_selection, prepare, prepared_path
from minillm.generate import generate_text
from minillm.model import GPT, GPTConfig
from minillm.trainer import TrainConfig, Trainer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["tinystories=2000"],
                    help="key=Anzahl_Dokumente, z.B. tinystories=2000 (Schlüssel: " + ", ".join(BY_KEY) + ")")
    ap.add_argument("--name", default="cli")
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--embd", type=int, default=128)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    selection = {k: int(n) for k, n in (s.split("=") for s in args.datasets)}
    print("Lade Datensätze:", selection)
    docs = load_selection(selection, lambda k, i, n: print(f"  {k}: {i}/{n}", end="\r"))
    print("\nTrainiere Tokenizer und kodiere Texte ...")
    meta = prepare(args.name, docs, args.vocab)
    print(f"  {meta['tokens_train']:,} Trainings-Tokens, {meta['chars_per_token']:.2f} Zeichen pro Token")

    cfg = GPTConfig(vocab_size=meta["vocab_size"], block_size=args.block, n_layer=args.layers,
                    n_head=args.heads, n_embd=args.embd)
    tr = Trainer(args.name, prepared_path(args.name), cfg,
                 TrainConfig(max_steps=args.steps, batch_size=args.batch, lr=args.lr,
                             eval_interval=max(10, args.steps // 10), sample_interval=max(50, args.steps // 4)))
    tr.start()
    while tr.is_running:
        time.sleep(2)
        if tr.metrics:
            m = tr.metrics[-1]
            print(f"Schritt {tr.step}/{args.steps}  loss {m['loss']:.3f}  {m['tok_s']:.0f} Tok/s", end="\r")
    print()
    if tr.status == "fehler":
        raise SystemExit(tr.error)
    print("Fertig:", tr.status, "| bester Val-Loss:", round(tr.best_val, 3))
    print("Beispiel:", generate_text(tr.model, tr.tok, "Once upon a time", max_new_tokens=60))


if __name__ == "__main__":
    main()
