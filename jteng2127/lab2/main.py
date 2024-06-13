from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
import os
from PIL import Image


def chdir_to_scrips_dir():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))


class ButterflyMothDataset(Dataset):
    def __init__(
        self,
        data_folder="data",
        csv_file="butterflies and moths.csv",
        dataset="train",
    ):
        self.data_folder = data_folder
        self.df = pd.read_csv(os.path.join(self.data_folder, csv_file))
        self.df = self.df[self.df["data set"] == dataset]
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __getitem__(self, index):
        item = self.df.iloc[index]
        image = Image.open(os.path.join(self.data_folder, item["filepaths"])).convert(
            "RGB"
        )
        image = self.transform(image)
        class_id = item["class id"]
        label = item["labels"]
        return image, label, class_id

    def __len__(self):
        return len(self.df)


class ButterflyMothLoader(DataLoader):
    def __init__(self, dataset="train", **kwargs):
        self.dataset = ButterflyMothDataset(dataset=dataset)
        super().__init__(dataset=self.dataset, **kwargs)


if __name__ == "__main__":
    chdir_to_scrips_dir()

    training_loader = ButterflyMothLoader("train")
    testing_loader = ButterflyMothLoader("test")
    validation_loader = ButterflyMothLoader("valid")

    for image, label, class_id in training_loader:
        print(image.shape)
        print(label)
        print(class_id)
        break
