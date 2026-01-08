import matplotlib.pyplot as plt

from constants import CLASS_LIST


def show_batch(
    batch, img_range=range(20),
    figsize=(25, 8), columns: int = 5
) -> None:
    """Display a batch of images with their class labels."""
    images, labels = batch
    rows = (len(img_range) + columns - 1) // columns

    fig, axes = plt.subplots(rows, columns, figsize=figsize)
    axes = axes.flatten()

    for idx, i in enumerate(img_range):
        class_label = int(labels[i])
        axes[idx].imshow(images[i])
        axes[idx].set_title(CLASS_LIST[class_label])
        axes[idx].axis('off')

    for j in range(len(img_range), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
