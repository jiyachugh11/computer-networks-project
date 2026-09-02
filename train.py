import argparse
import os

from src.datasets.traffic_dataset import TrafficDataset
from src.training.trainer import Trainer


def main():

    parser = argparse.ArgumentParser(
        description="Train TrafficCLIP"
    )

    parser.add_argument(
        "--train_csv",
        default="data/splits/train.csv",
        help="Path to training CSV"
    )

    parser.add_argument(
        "--val_csv",
        default="data/splits/val.csv",
        help="Path to validation CSV"
    )

    parser.add_argument(
        "--test_csv",
        default="data/splits/test.csv",
        help="Path to test CSV"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.002
    )

    args = parser.parse_args()

    # Check that the CSV files exist
    for csv_file in [
        args.train_csv,
        args.val_csv,
        args.test_csv
    ]:

        if not os.path.exists(csv_file):
            raise FileNotFoundError(
                f"CSV file not found: {csv_file}"
            )

    print("==============================")
    print("TrafficCLIP Training")
    print("==============================")

    print(f"Train CSV: {args.train_csv}")
    print(f"Val CSV:   {args.val_csv}")
    print(f"Test CSV:  {args.test_csv}")
    print(f"Epochs:    {args.epochs}")
    print(f"Batch size:{args.batch_size}")

    # Load datasets
    train_dataset = TrafficDataset(
        args.train_csv
    )

    # Use the SAME class mapping for validation and test
    class_names = list(
        train_dataset.class_to_idx.keys()
    )

    val_dataset = TrafficDataset(
        args.val_csv,
        class_to_idx=train_dataset.class_to_idx
    )

    test_dataset = TrafficDataset(
        args.test_csv,
        class_to_idx=train_dataset.class_to_idx
    )

    print()
    print(f"Classes: {class_names}")

    # Create trainer
    trainer = Trainer(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        class_names=class_names,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs
    )

    # Train
    train_losses, val_losses, accuracy, macro_f1 = trainer.train()

    # Save results
    os.makedirs("outputs", exist_ok=True)

    with open(
        "outputs/results.txt",
        "a"
    ) as f:

        f.write("\n--- TrafficCLIP Training ---\n")
        f.write(f"Classes: {class_names}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Final Train Loss: {train_losses[-1]:.4f}\n")
        f.write(f"Final Val Loss: {val_losses[-1]:.4f}\n")
        f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n")
        f.write(f"Macro F1: {macro_f1 * 100:.2f}%\n")

    print()
    print("Training complete!")
    print("Results saved to outputs/results.txt")


if __name__ == "__main__":
    main()