import os

import pandas as pd

from base.base_dataset import TextImageDataset

class Ucla(TextImageDataset):
    def _load_metadata(self):
        csv_fp = os.path.join(self.metadata_dir, 'labels_split.csv')  # Load labels_split.csv
        df = pd.read_csv(csv_fp, sep='\t')

        train_df = df[df['split'] == 'train']  # Get rows where 'split' is 'train'
        test_df = df[df['split'] == 'test']  # Get rows where 'split' is 'test'

        if self.split == 'train':
            df = train_df
        else:
            df = test_df

        self.split_sizes = {'train': len(train_df), 'test': len(test_df)}

        self.metadata = df.groupby(['fname'])['caption'].apply(list)
        if self.subsample < 1:
            self.metadata = self.metadata.sample(frac=self.subsample)

        self.metadata = pd.DataFrame({'captions': self.metadata})

    def _get_video_path(self, sample):
        return os.path.join(self.data_dir, sample.name), sample.name

    def _get_caption(self, sample):
        return sample['captions'][0]
