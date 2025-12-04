from typing import Dict, Any
import copy

import torch
import numpy as np

from odgn.constants import Numeric, Scalar


class Trajectory:
    _stacking_routine = np.array

    def __init__(self):
        self.o = []
        self.a = []
        self.a_logits = []
        self.r = []
        self.term = []
        self.trunc = []
        self.valid = []
        self.done = False
        self.packed = False

    def to_dict(self) -> Dict[str, Any]:
        return {'observations': self.o,
                'actions': self.a,
                'rewards': self.r,
                'terminals': self.term,
                'truncateds': self.trunc}

    def __repr__(self):
        return f'Trajectory ({self.length})'

    def __deepcopy__(
            self,
            memo: Dict[Any, Any]
    ):
        if id(self) in memo:
            return memo[id(self)]
        cpy = type(self)()
        memo[id(self)] = cpy

        cpy.o = copy.deepcopy(self.o, memo)
        cpy.a = copy.deepcopy(self.a, memo)
        cpy.a_logits = copy.deepcopy(self.a_logits, memo)
        cpy.r = copy.deepcopy(self.r, memo)
        cpy.term = copy.deepcopy(self.term, memo)
        cpy.trunc = copy.deepcopy(self.trunc, memo)
        cpy.valid = copy.deepcopy(self.valid, memo)

        return cpy

    def __getitem__(self, item):
        return (self.o[item], self.a[item], self.a_logits[item], self.r[item], self.term[item],
                self.trunc[item], self.valid[item])

    def __next__(self):
        try:
            for i in range(self.length):
                yield self[i]
        except IndexError:
            raise StopIteration

    @property
    def length(self):
        return len(self.a)

    def __len__(self):
        return len(self.a)

    def add_step(
            self,
            o: Numeric,
            a: Numeric,
            a_logits: Numeric,
            r: Scalar,
            term: Scalar,
            trunc: Scalar,
    ):
        self.o.append(o)
        self.a.append(a)
        self.a_logits.append(a_logits)
        self.r.append(r)
        self.term.append(term)
        self.trunc.append(trunc)
        self.valid.append(True)

    def add_final_obs(
            self,
            o_final: Numeric,
    ):
        self.o.append(o_final)
        self.done = True

    def pack(
            self,
            desired_length: int = None
    ):
        self.o = self._stacking_routine(self.o)
        self.a = self._stacking_routine(self.a)
        self.a_logits = self._stacking_routine(self.a_logits)
        self.r = self._stacking_routine(self.r)
        self.term = self._stacking_routine(self.term)
        self.trunc = self._stacking_routine(self.trunc)
        self.valid = self._stacking_routine(self.valid)

        if desired_length is None:
            desired_length = self.length

        to_pad = desired_length - self.length
        to_pad_o = desired_length + 1 - len(self.o)

        self.o = self._pad_time(self.o, to_pad_o)
        self.a = self._pad_time(self.a, to_pad)
        self.a_logits = self._pad_time(self.a_logits, to_pad)
        self.r = self._pad_time(self.r, to_pad)
        self.term = self._pad_time(self.term, to_pad)
        self.trunc = self._pad_time(self.trunc, to_pad)
        self.valid = self._pad_time(self.valid, to_pad)

        self.packed = True

        return self

    @staticmethod
    def _pad_time(
            x: Numeric,
            size: int
    ):
        if size <= 0:
            return x
        padding = np.zeros((size, *x.shape[1:]), dtype=x.dtype)
        return np.concatenate([x, padding], axis=0)


class TorchTrajectory(Trajectory):
    _stacking_routine = torch.stack

    @classmethod
    def from_trajectory(cls, traj: Trajectory) -> "TorchTrajectory":
        new_traj = cls()
        if traj.packed:
            def _convert(xs):
                return torch.from_numpy(xs)
        else:
            def _convert(xs):
                return list(map(lambda x: torch.from_numpy(x), xs))

        new_traj.o = _convert(traj.o)
        new_traj.a = _convert(traj.a)
        new_traj.a_logits = _convert(traj.a_logits)
        new_traj.r = _convert(traj.r)
        new_traj.term = _convert(traj.term)
        new_traj.trunc = _convert(traj.trunc)
        new_traj.valid = _convert(traj.valid)
        new_traj.done = traj.done
        new_traj.packed = traj.packed

        return new_traj

    @staticmethod
    def _pad_time(
            x: Numeric,
            size: int
    ):
        if size <= 0:
            return x
        padding = torch.zeros((size, *x.shape[1:]), dtype=x.dtype, device=x.device)
        return torch.cat([x, padding], dim=0)
