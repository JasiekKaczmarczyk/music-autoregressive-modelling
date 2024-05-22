import random
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

class MusicDataset(Dataset):
    def __init__(self, filepaths: str, length: int = None):
        super().__init__()

        self.filepaths = filepaths
        self.sequence_length = length

        self.target_freqs = [91, 182, 375, 750, 1_500, 3_000, 6_000, 12_000, 24_000, 48_000]

        self.transforms = [
            torchaudio.transforms.Resample(orig_freq=48_000, new_freq=freq)
            for freq in self.target_freqs
        ]

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # record = torch.load(self.dataset[index])
        metadata = torchaudio.info(self.filepaths[index])
        offset = random.randint(0, metadata.num_frames-self.sequence_length)
        audio_data, _ = torchaudio.load(self.filepaths[index], num_frames=self.sequence_length, frame_offset=offset)

        # mono
        audio_data = torch.mean(audio_data, dim=0, keepdim=True)

        signals = {}

        for transform, freq in zip(self.transforms, self.target_freqs):
            resampled_signal = transform(audio_data)
            
            signals[freq] = resampled_signal

        # for now
        label = torch.zeros((128, ))

        return signals, label
    
if __name__ == "__main__":
    import glob
    from torch.utils.data import DataLoader
    ds = glob.glob("music/**.mp3")

    dataset = MusicDataset(ds, length=1_048_576)
    # sample = dataset[0]
    loader = DataLoader(dataset, batch_size=8)

    signals, y = next(iter(loader))

    print([x.shape for x in signals.values()])


