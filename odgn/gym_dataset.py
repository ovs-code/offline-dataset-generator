from datetime import datetime
from typing import List
from dataclasses import dataclass
from pathlib import Path
import pickle

from odgn.trajectory import Trajectory, TorchTrajectory

@dataclass
class GymDataset:
    gymnasium_env_id: str
    collection_date: datetime
    trajectories: List[Trajectory | TorchTrajectory]

    def store(self, path: Path | str):
        path = Path(path)
        if path.is_dir():
            raise ValueError(f'Path needs to end in a file name: {path}')
        if not path.exists():
            path.parent.absolute().mkdir(parents=True, exist_ok=True)
            path.touch()
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Trajectory | TorchTrajectory:
        return self.trajectories[idx]

    @staticmethod
    def load(path: Path | str) -> 'GymDataset':
        path = Path(path)
        with open(path, 'rb') as f:
            dset = pickle.load(f)
        return dset


