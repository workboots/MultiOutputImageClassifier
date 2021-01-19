import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("Agg")
# Defining the loss names

lossNames = ["loss", "category_output_loss", "colour_output_loss"]
plt.style.use("ggplot")

(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))

with open('/home/workboots/Projects/MultiOutputImageClassifier/history', 'rb') as f:
    history = pickle.load(f)

for i, l in enumerate(lossNames):
    title = "Loss for {}".format(l) if l != "loss" else "Total loss"
    ax[i].set_title(title)
    ax[i].set_xlabel("Epoch #")
    ax[i].set_ylabel("Loss")
    ax[i].plot(np.arange(0, len(history[l])), history[l], label=l)
    ax[i].plot(np.arange(0, len(history[l])),
               history["val_"+l], label="val_"+l)
    ax[i].legend()

plt.tight_layout()
plt.savefig("/home/workboots/Projects/MultiOutputImageClassifier/losses.png")
plt.close()

accuracyNames = ["category_output_accuracy", "colour_output_accuracy"]
plt.style.use("ggplot")

(fig, ax) = plt.subplots(2, 1, figsize=(8, 8))

for i, l in enumerate(accuracyNames):
    title = "Accuracy for {}".format(l)
    ax[i].set_title(title)
    ax[i].set_xlabel("Epoch #")
    ax[i].set_ylabel("Accuracy")
    ax[i].plot(np.arange(0, len(history[l])), history[l], label=l)
    ax[i].plot(np.arange(0, len(history[l])),
               history["val_"+l], label="val_"+l)
    ax[i].legend()

plt.tight_layout()
plt.savefig("/home/workboots/Projects/MultiOutputImageClassifier/accs.png")
plt.close()
