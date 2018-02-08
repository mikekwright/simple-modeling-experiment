import json
import os

from random import Random
from PIL import Image


class ICDARImageReader:
    """
    """
    def __init__(self, dataset_dir, catalog_file='char.json', labels=None, seed=42):
        self.dataset_dir = dataset_dir
        self.catalog_file = catalog_file
        self.labels = labels
        self.random = Random(seed)

    def get_dataset_list(self):
        with open(os.path.join(self.dataset_dir, self.catalog_file), 'r', encoding='utf-8') as raw_catalog:
            catalog = json.load(raw_catalog)

        return catalog

    def load_all_images(self, catalog):
        loadable_labels = [(l, img) for l, images in catalog.items()
                                for img in catalog[l]
                                if self.labels is None or l in self.labels]

        self.random.shuffle(loadable_labels)
        for label, img in loadable_labels:
            yield (label, self.read_image_pixels(img))

    def read_image_pixels(self, img_path):
        img = Image.open(os.path.join(self.dataset_dir, img_path))
        resized = img.resize((28, 28))
        return list(resized.getdata())

    def __call__(self):
        catalog = self.get_dataset_list()
        yield from self.load_all_images(catalog)

