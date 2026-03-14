import csv
import argparse
import config
import matplotlib.pyplot as plt

def parse_log(filepath):
    """Load training log CSV and return a list of dicts."""
    rows = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "val_loss": float(row["val_loss"]),
                "train_acc": float(row["train_acc"]),
                "val_acc": float(row["val_acc"])
            })
    return rows

def detect_overfit(rows, patience=5):
    """Detect epoch where validation loss starts consistently climbing."""
    best_val_loss = float("inf")
    best_epoch = 1
    counter = 0

    for row in rows:
        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            best_epoch = row["epoch"]
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                return best_epoch, best_val_loss

    return best_epoch, best_val_loss

def summarise(rows, best_epoch, best_val_loss):
    """Print a clean training summary report."""
    final = rows[-1]
    total_epochs = len(rows)

    print(f"\n============================")
    print(f"  TRAINING LOG REPORT")
    print(f"  {total_epochs} epochs")
    print(f"============================\n")

    print(f"FINAL METRICS (epoch {total_epochs})")
    print(f"  Train loss    {final['train_loss']:.4f}")
    print(f"  Val loss      {final['val_loss']:.4f}")
    print(f"  Train acc     {final['train_acc']:.1%}")
    print(f"  Val acc       {final['val_acc']:.1%}")

    print(f"\nOVERFIT DETECTION")
    if best_epoch < total_epochs:
        print(f"  ⚠ Overfitting detected from epoch {best_epoch}")
        print(f"  Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
        print(f"  Recommendation: use early stopping at epoch {best_epoch}")
    else:
        print(f"  ✓ No overfitting detected")

    print(f"\n============================")

def plot_curves(rows, best_epoch, output_path="learning_curve.png"):
    """Plot training vs validation loss and save to file."""
    epochs = [r["epoch"] for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    val_loss = [r["val_loss"] for r in rows]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="Train loss", color="steelblue")
    plt.plot(epochs, val_loss, label="Val loss", color="tomato")
    plt.axvline(x=best_epoch, color="orange", linestyle="--", label=f"Early stop (epoch {best_epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve — Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\n  Chart saved → {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Log Analyser")
    parser.add_argument("--file", type=str, default=config.DATA_PATH, help="Path to training log CSV")
    parser.add_argument("--patience", type=int, default=config.PATIENCE, help="Early stopping patience")
    args = parser.parse_args()

    rows = parse_log(args.file)
    best_epoch, best_val_loss = detect_overfit(rows, args.patience)
    summarise(rows, best_epoch, best_val_loss)

    plot_curves(rows, best_epoch)