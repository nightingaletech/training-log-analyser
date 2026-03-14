import csv
import random

random.seed(42)

def generate_training_log(filepath, epochs=50):
    """Generate synthetic training log with overfitting built in after epoch 20."""
    rows = []
    train_loss = 1.0
    val_loss = 1.0

    for epoch in range(1, epochs + 1):
        # training loss keeps improving throughout
        train_loss = train_loss * random.uniform(0.92, 0.97)
        train_acc = 1 - train_loss + random.uniform(-0.02, 0.02)

        # validation loss improves until epoch 20, then starts climbing
        if epoch <= 20:
            val_loss = val_loss * random.uniform(0.93, 0.98)
        else:
            val_loss = val_loss * random.uniform(1.01, 1.04)
        val_acc = 1 - val_loss + random.uniform(-0.02, 0.02)

        rows.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "train_acc": round(min(max(train_acc, 0), 1), 4),
            "val_acc": round(min(max(val_acc, 0), 1), 4)
        })

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {epochs} epochs → {filepath}")

if __name__ == "__main__":
    generate_training_log("training_log.csv")