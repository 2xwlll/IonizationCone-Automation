import matplotlib.pyplot as plt
from pathlib import Path
import datetime

def save_plot(fig, name, directory="results/synthetic", fmt="png", show=False):
    Path(directory).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{directory}/{name}_{timestamp}.{fmt}"
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[✓] Plot saved to {filename}")
    if show:
        plt.show()
    plt.close(fig)

def plot_prediction(inputs, outputs, masks, idx=0, title=None):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(inputs[idx, 0], cmap="gray")
    axes[0].set_title("Input")
    axes[1].imshow(outputs[idx, 0], cmap="magma")
    axes[1].set_title("Prediction")
    axes[2].imshow(masks[idx, 0], cmap="viridis")
    axes[2].set_title("Ground Truth")
    if title:
        fig.suptitle(title)
    return fig

