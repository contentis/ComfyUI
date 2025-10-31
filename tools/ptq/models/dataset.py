import torch
import numpy as np
from typing import Dict, Any
from datasets import load_dataset


class HFPromptDataloader(torch.utils.data.Dataset):
    def __init__(self, split="test", max_samples=None):
        self.dataset = load_dataset("Gustavosta/Stable-Diffusion-Prompts", split=split)

        if max_samples is not None and max_samples < len(self.dataset):
            self.dataset = self.dataset.select(range(max_samples))

    def __iter__(self):
        for sample in self.dataset:
            yield {"prompt": sample["Prompt"]}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return {"prompt": sample["Prompt"]}


class KontextBenchDataLoader:
    def __init__(
            self,
            dataset_name: str = "black-forest-labs/kontext-bench",
            split: str = "test",
    ):
        self.dataset_name = dataset_name
        self.split = split
        self._dataset = None

    def load_dataset(self):
        if self._dataset is None:
            self._dataset = load_dataset(
                self.dataset_name,
                split=self.split,
            )

        return self._dataset

    def __iter__(self):
        dataset = self.load_dataset()
        for sample in dataset:
            yield {
                'image': self.preprocess_img(sample['image']),
                'prompt': sample['instruction']
            }

    def __len__(self) -> int:
        dataset = self.load_dataset()
        return len(dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        dataset = self.load_dataset()
        sample = dataset[idx]
        return {
            'image': self.preprocess_img(sample['image']),
            'prompt': sample['instruction']
        }

    def preprocess_img(self, img):
        img = img.convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img)
        return img.unsqueeze(0)

    def get_full_sample(self, idx: int) -> Dict[str, Any]:
        dataset = self.load_dataset()
        return dataset[idx]

    def filter_by_category(self, category: str):
        dataset = self.load_dataset()
        return dataset.filter(lambda x: x['category'] == category)

    @staticmethod
    def get_categories():
        return [
            "Character Reference",
            "Instruction Editing - Global",
            "Instruction Editing - Local",
            "Style Reference",
            "Text Editing"
        ]

