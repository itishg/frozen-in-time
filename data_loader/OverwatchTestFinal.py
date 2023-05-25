import os
import pandas as pd
from base.base_dataset import TextVideoDataset

class OverwatchTestFinal(TextVideoDataset):
    def _load_metadata(self):
        csv_fp = os.path.join(self.metadata_dir, 'labels.csv') 
        df = pd.read_csv(csv_fp)

        self.metadata = df.groupby(['filename'])['text'].apply(list)
        self.metadata = pd.DataFrame({'captions': self.metadata})

    def _get_video_path(self, sample):
        return os.path.join(self.data_dir, sample.name), sample.name

    def _get_caption(self, sample):
        return sample['captions'][0]

if __name__ == "__main__":
    from data_loader import transforms
    tsfms = transforms.init_transform_dict()
    ds = OverwatchTest(
        'OverwatchTest',
        {"input": "text"},
        {"input_res": 224, "num_frames": 4},
        data_dir='/content/drive/MyDrive/overwatch_clips/data_test/clips/final/',
        metadata_dir='/content/drive/MyDrive/overwatch_clips/data_test/clips/final/',
        split='test',
        tsfms=tsfms['test']
    )

    for x in range(100):
        print(ds.__getitem__(x))
