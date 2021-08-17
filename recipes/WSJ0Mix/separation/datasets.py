import musdb
import numpy as np
import torch
from torch.utils.data import Dataset

class MusdbDataset(Dataset):
    def __init__(self, hparams):
      self._musdb = musdb.DB(hparams["musdb_path"], subsets=["test"])

    def __len__(self):
      return len(self._musdb)

    def __getitem__(self, idx):
      track = self._musdb.tracks[idx]
      track.chunk_duration = 20
      track.chunk_start = np.random.uniform(0, track.duration - track.chunk_duration)
      x = torch.from_numpy(track.audio.T).float()
      y = torch.from_numpy(track.stems[1:, :, :]).float()
      return x, y