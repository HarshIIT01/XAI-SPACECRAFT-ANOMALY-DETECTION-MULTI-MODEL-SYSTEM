"""
Train a spacecraft anomaly detector.

Usage (from project root)::

  python scripts/train.py --model GNN --dataset SMAP --channel P-1 --epochs 50
"""

import argparse

from _bootstrap import bootstrap

bootstrap()

from spacecraft_anomaly.training import train  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Train spacecraft anomaly detector")
    parser.add_argument(
        "--model",
        default="GNN",
        choices=["LSTM_AE", "LSTM_VAE", "TRANSFORMER", "GNN", "FUSION"],
    )
    parser.add_argument("--dataset", default="SMAP", choices=["SMAP", "MSL", "OPSSAT"])
    parser.add_argument("--channel", default="P-1")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save", default="checkpoints")
    args = parser.parse_args()

    train(
        model_type=args.model,
        dataset=args.dataset,
        channel=args.channel,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        window_size=args.window,
        device_pref=args.device,
        save_dir=args.save,
    )


if __name__ == "__main__":
    main()
