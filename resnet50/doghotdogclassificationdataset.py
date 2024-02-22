from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms


class DogHotdogClassificationDataset(Dataset):
    """
    A custom dataset class for dog vs hotdog image classification, inheriting from PyTorch's Dataset class.
    This class handles loading images from a directory, applying transformations, and providing
    PyTorch tensors for training or inference.
    """

    def __init__(self):
        # Initialize the parent Dataset class to leverage built-in functionalities
        super().__init__()
        # Load examples from disk
        self.examples = self._load_examples()

        # Define transformations: Convert PIL images to PyTorch tensors
        self.pil_to_tensor = transforms.ToTensor()
        # Apply random cropping to images to make them a fixed size (64x64). This is useful for
        # standardizing input sizes and potentially augmenting the dataset with cropped variations.
        self.resize = transforms.RandomCrop((64, 64))

    def _load_examples(self):
        """
        Loads image file paths and their corresponding labels from the filesystem.
        """
        # List directories in 'classes' folder, each representing a class
        classes = os.listdir("classes")
        # Create a mapping of class names to numerical labels
        class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
        # Optional: create a reverse mapping if needed (not used in this class)
        idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

        examples = []  # This will store tuples of (image file path, label)
        for class_name in classes:
            # List all images in the class directory
            example_img_fps = os.listdir(os.path.join("classes", class_name))
            # Prepend the directory path to image file names
            example_img_fps = [
                os.path.join("classes", class_name, img_name)
                for img_name in example_img_fps
            ]
            # Pair each image file path with its label
            example_tuples = [
                (img_fp, class_to_idx[class_name]) for img_fp in example_img_fps
            ]
            examples.extend(example_tuples)

        return examples

    def __getitem__(self, idx):
        """
        Returns the image tensor and its label for a given index.
        """
        # Retrieve the image file path and label for the specified index
        img_fp, label = self.examples[idx]
        # Load the image from disk
        img = Image.open(img_fp)
        # Convert the PIL image to a PyTorch tensor
        features = self.pil_to_tensor(img)
        # Resize (crop) the image tensor to the desired size
        features = self.resize(features)
        return (features, label)

    def __len__(self):
        """
        Returns the total number of examples in the dataset.
        """
        # The length of the dataset is the number of image-label pairs
        return len(self.examples)
