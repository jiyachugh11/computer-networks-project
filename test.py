import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.datasets.traffic_dataset import TrafficDataset
from src.models.traffic_clip import TrafficCLIP


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("==============================")
    print("TrafficCLIP Testing")
    print("==============================")
    print(f"Device: {device}")

    # Create output folder
    os.makedirs("outputs/graphs", exist_ok=True)

    # Load test dataset
    test_dataset = TrafficDataset("data/splits/test.csv")

    class_names = list(test_dataset.class_to_idx.keys())

    print(f"Classes: {class_names}")
    print(f"Test samples: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    print(f"Total batches: {len(test_loader)}")

    # Load model
    print("\nLoading model...")

    model = TrafficCLIP(embed_dim=1024).to(device)

    checkpoint = torch.load(
        "outputs/checkpoints/epoch_2.pth",
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    model.eval()

    text_prompts = [
        f"a network traffic photo of {c}"
        for c in class_names
    ]

    predictions = []
    targets = []

    print("\n==============================")
    print("Starting Testing")
    print("==============================")

    progress_bar = tqdm(
        test_loader,
        desc="Testing",
        unit="batch"
    )

    with torch.no_grad():

        for batch in progress_bar:

            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits, _ = model(images, text_prompts)

            preds = torch.argmax(logits, dim=1)

            predictions.extend(
                preds.cpu().numpy()
            )

            targets.extend(
                labels.cpu().numpy()
            )

    # Calculate metrics
    accuracy = accuracy_score(
        targets,
        predictions
    )

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro"
    )

    # ==========================================
    # CONFUSION MATRIX
    # ==========================================

    cm = confusion_matrix(
        targets,
        predictions
    )

    plt.figure(figsize=(8, 6))

    plt.imshow(cm)

    plt.title("TrafficCLIP Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.xticks(
        range(len(class_names)),
        class_names,
        rotation=45
    )

    plt.yticks(
        range(len(class_names)),
        class_names
    )

    # Write numbers inside the matrix
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.colorbar()
    plt.tight_layout()

    plt.savefig(
        "outputs/graphs/confusion_matrix.png",
        dpi=300
    )

    plt.close()

    print("\nConfusion matrix saved!")

    # ==========================================
    # ACCURACY AND F1 GRAPH
    # ==========================================

    plt.figure(figsize=(7, 5))

    metrics = ["Accuracy", "Macro F1"]
    values = [
        accuracy * 100,
        macro_f1 * 100
    ]

    plt.bar(metrics, values)

    plt.ylabel("Percentage (%)")
    plt.title("TrafficCLIP Test Performance")

    plt.ylim(0, 100)

    for i, value in enumerate(values):
        plt.text(
            i,
            value + 1,
            f"{value:.2f}%",
            ha="center"
        )

    plt.tight_layout()

    plt.savefig(
        "outputs/graphs/test_metrics.png",
        dpi=300
    )

    plt.close()

    print("Test metrics graph saved!")

    # ==========================================
    # PER-CLASS METRICS
    # ==========================================

    report = classification_report(
        targets,
        predictions,
        target_names=class_names,
        output_dict=True
    )

    precision_values = [
        report[c]["precision"] * 100
        for c in class_names
    ]

    recall_values = [
        report[c]["recall"] * 100
        for c in class_names
    ]

    f1_values = [
        report[c]["f1-score"] * 100
        for c in class_names
    ]

    x = range(len(class_names))

    plt.figure(figsize=(9, 6))

    width = 0.25

    plt.bar(
        [i - width for i in x],
        precision_values,
        width,
        label="Precision"
    )

    plt.bar(
        x,
        recall_values,
        width,
        label="Recall"
    )

    plt.bar(
        [i + width for i in x],
        f1_values,
        width,
        label="F1 Score"
    )

    plt.xticks(
        list(x),
        class_names
    )

    plt.ylabel("Percentage (%)")
    plt.title("Per-Class Performance")
    plt.ylim(0, 100)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "outputs/graphs/class_metrics.png",
        dpi=300
    )

    plt.close()

    print("Class metrics graph saved!")

    # ==========================================
    # FINAL RESULTS
    # ==========================================

    print("\n==============================")
    print("FINAL TEST RESULTS")
    print("==============================")

    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro F1: {macro_f1 * 100:.2f}%")

    print("\nPer-Class Results:")

    for class_name in class_names:

        print(
            f"{class_name}: "
            f"Precision = {report[class_name]['precision'] * 100:.2f}% | "
            f"Recall = {report[class_name]['recall'] * 100:.2f}% | "
            f"F1 = {report[class_name]['f1-score'] * 100:.2f}%"
        )

    print("\n==============================")
    print("Graphs saved in:")
    print("outputs/graphs/")
    print("==============================")


if __name__ == "__main__":
    main()