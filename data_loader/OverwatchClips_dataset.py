import os
import random

import pandas as pd

from base.base_dataset import TextVideoDataset

class OverwatchClips(TextVideoDataset):
    def _load_metadata(self):
        csv_fp = os.path.join(self.metadata_dir, 'labels.csv')
        df = pd.read_csv(csv_fp)

        train_df = df.sample(frac=0.8, random_state=1)  # Use 80% of the data for training
        test_df = df.drop(train_df.index)  # Use the rest for testing

        if self.split == 'train':
            df = train_df
        else:
            df = test_df

        self.split_sizes = {'train': len(train_df), 'test': len(test_df)}

        self.metadata = df.groupby(['filename'])['text'].apply(list)
        if self.subsample < 1:
            self.metadata = self.metadata.sample(frac=self.subsample)

        self.metadata = pd.DataFrame({'captions': self.metadata})

    def _get_video_path(self, sample):
        return os.path.join(self.data_dir, 'overwatch_clips', 'final', sample.name + '.mp4'), sample.name + '.mp4'

    def _get_caption(self, sample):
        caption_sample = self.text_params.get('caption_sample', "rand")
        if self.split == 'train' and caption_sample == "rand":
            caption = random.choice(sample['captions'])
        else:
            caption = sample['captions'][0]
        return caption