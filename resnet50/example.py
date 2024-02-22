import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import warnings

import torch.nn.functional as F
from torch.utils.data import DataLoader

from doghotdogclassificationdataset import DogHotdogClassificationDataset

# Suppress warnings to make the output cleaner, especially useful when running in a notebook or a production environment
warnings.filterwarnings("ignore")

# Determine if a CUDA GPU is available and use it; otherwise, default to CPU for computation. This optimizes for performance.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for inference")


class DogHotdogClassifier(torch.nn.Module):
    """
    A classifier that uses a pretrained ResNet50 model to distinguish between images of dogs and hotdogs.
    The model's final fully connected layer is replaced to adapt to the binary classification task.
    """

    def __init__(self):
        # The super() method is used here to call the __init__ method of the parent class (torch.nn.Module),
        # allowing us to use its functionalities and ensure the class is correctly initialized.
        super().__init__()
        # Load a pretrained ResNet50 model from NVIDIA's torchhub repository. This model is chosen for its
        # effectiveness in image classification tasks. The pretrained model comes with weights trained on a
        # large dataset, providing a strong feature extractor as a starting point for our classification task.
        self.resnet50 = torch.hub.load(
            "NVIDIA/DeepLearningExamples:torchhub", "nvidia_resnet50", pretrained=True
        )
        # Replace the final fully connected layer of ResNet50 to adapt it from 1000 classes to a single output
        # neuron, since we're dealing with a binary classification problem (dog vs hotdog). This is a common
        # practice when fine-tuning pretrained models for new tasks.
        self.resnet50.fc = torch.nn.Linear(2048, 1)

    def forward(self, X):
        # Apply a sigmoid activation function to the output of the modified ResNet50 model to obtain a probability
        # score between 0 and 1, indicating the likelihood of the input image being a hotdog.
        return torch.sigmoid(self.resnet50(X))


def train(model, dataloader, epochs=20):
    """
    Trains the model on the dataset wrapped by the DataLoader for a specified number of epochs.

    Args:
        model: The neural network model to be trained.
        dataloader: DataLoader that provides batches of data from the training dataset.
        epochs: The number of times the entire dataset will be passed through the model.
    """
    # Use Adam optimizer with a common initial learning rate for binary classification tasks
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        for batch in dataloader:
            features, labels = batch
            predictions = model(features)
            # Ensure labels are the correct shape and type for binary cross-entropy loss
            labels = labels.unsqueeze(1).float()
            loss = F.binary_cross_entropy(predictions, labels)
            loss.backward()  # Compute gradients
            optimiser.step()  # Update model parameters
            optimiser.zero_grad()  # Reset gradients for the next iteration
            print(loss.item())


def accuracy(model, dataset, dataloader, show_image=True):
    """
    Calculates the accuracy of the model on the provided dataset and optionally displays an image from the validation set along with its label and prediction.

    Args:
        model: The neural network model whose accuracy is to be evaluated.
        dataset: The dataset used for evaluating the model.
        dataloader: DataLoader that provides batches of data from the dataset.
        show_image: Flag to indicate whether an image should be displayed for validation along with its label and prediction.
    """
    # Calculate total number of items in dataset
    m = len(dataset)
    n_correct = 0
    for batch in dataloader:
        features, labels = batch
        features, labels = features.to(device), labels.to(device)
        predictions = model(features)
        # Count number of correct predictions
        correct = torch.round(predictions.squeeze()) == labels
        n_correct += correct.sum().item()

    # Calculate and print accuracy
    accuracy = 100 * n_correct / m
    print(f"Accuracy: {accuracy:.2f} %")

    if show_image:
        # Display an image from the last batch
        img, label = features[-1].cpu(), labels[-1].cpu()
        predicted_label = torch.round(predictions[-1]).int().item()
        img = img.numpy().transpose(
            (1, 2, 0)
        )  # Convert image to NumPy array and change order from CxHxW to HxWxC

        # Display the image
        plt.imshow(img)
        plt.title(f"Label: {label.item()}, Predicted: {predicted_label}")
        plt.show()

        # Print label and prediction
        label_dict = {
            0: "Not Hotdog",
            1: "Hotdog",
        }  # Update this dictionary based on your dataset labels
        print(f"Actual Label: {label_dict[label.item()]}")
        print(f"Predicted Label: {label_dict[predicted_label]}")


# Ensure the rest of your script remains unchanged


if __name__ == "__main__":
    # Instantiate the classifier and dataset, then calculate initial accuracy, train, and recalculate accuracy
    classifier = DogHotdogClassifier().to(device)
    dataset = DogHotdogClassificationDataset()
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    # Move the model to the selected device for optimized computation
    classifier.to(device)
    # Evaluate accuracy before training for baseline comparison
    accuracy(classifier, dataset, train_loader)
    # Train the model on the dataset
    train(classifier, train_loader)
    # Evaluate and print the improved accuracy after training
    accuracy(classifier, dataset, train_loader)

# The following lines were originally commented out and illustrate additional capabilities such as processing
# inputs from URIs and displaying results, which might be part of an extended workflow for using the model in
# practical scenarios.

# utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')

# resnet50.eval().to(device)

# uris = [
#     'http://images.cocodataset.org/test-stuff2017/000000024309.jpg',
#     'http://images.cocodataset.org/test-stuff2017/000000028117.jpg',
#     'http://images.cocodataset.org/test-stuff2017/000000006149.jpg',
#     'http://images.cocodataset.org/test-stuff2017/000000004954.jpg',
# ]

# batch = torch.cat(
#     [utils.prepare_input_from_uri(uri) for uri in uris]
# ).to(device)

# with torch.no_grad():
#     output = torch.nn.functional.softmax(resnet50(batch), dim=1)

# results = utils.pick_n_best(predictions=output, n=5)

# for uri, result in zip(uris, results):
#     img = Image.open(requests.get(uri, stream=True).raw)
#     img.thumbnail((256,256), Image.Resampling.LANCZOS) # correction due to AttributeError: module 'PIL.Image' has no attribute 'ANTIALIAS'
#     plt.imshow(img)
#     plt.show()
#     print(result)
