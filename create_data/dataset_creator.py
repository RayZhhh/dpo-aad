import copy
import math
import random
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np
from math import isinf

from tqdm import tqdm

sys.path.append("../")

import pickle
from algodisco.base.algo import AlgoProto

from typing import List, Dict, TypedDict, Optional, Any


class DataDict(TypedDict, total=False):
    score: float
    func: str


class PromptChosenRejectedPairs:
    def __init__(self, prompt: Any, chosen: Any, rejected: Any):
        self.prompt = prompt
        self.chosen = chosen
        self.rejected = rejected

    def __repr__(self):
        return (
            f"[Prompt: {self.prompt}, Chosen: {self.chosen}, Rejected: {self.rejected}]"
        )

    def to_dict(self) -> Dict[str, Any]:
        # DPO training format: both chosen and rejected are full conversations
        # (system prompt + assistant response), differing only in the algorithm body.
        if isinstance(self.prompt, str):
            prompt = [{"role": "user", "content": self.prompt}]
        else:
            prompt = self.prompt

        chosen = prompt + [{"role": "assistant", "content": str(self.chosen)}]
        rejected = prompt + [{"role": "assistant", "content": str(self.rejected)}]
        return {"chosen": chosen, "rejected": rejected}


class DatasetCreator:
    """Constructs DPO preference pairs from algorithm search results via
    Diversity-Aware Rank-based (DAR) Sampling (paper Section 2.2).

    The algorithm pool collected by iterative LLM-driven search (e.g. FunSearch,
    EoH) is partitioned into M equal-sized fitness subsets. Preference pairs are
    then sampled so that chosen (y+) comes from a high-quality subset and
    rejected (y-) comes from a clearly inferior subset, enforcing a minimum
    quality gap of one tier between them.
    """

    def __init__(
        self,
        data: List[DataDict],
        prompt: str,
        number_of_subsets: int = 10,  # M in the paper
        remove_duplicate: bool = True,
        replace: bool = False,
        p: int = 3,  # temperature τ controlling exploitation vs. exploration
    ):
        self.data = data
        self.prompt = prompt
        self.number_of_subsets = number_of_subsets
        self.replace = replace
        self.p = p

        self.all_funcs: List[AlgoProto] = []
        for i in tqdm(range(len(self.data)), desc='Parsing to "AlgoProto" instances'):
            d = self.data[i]
            score = d["score"]
            func = d["func"]
            if score is None or np.isinf(score):
                continue
            if not func:
                continue
            try:
                algo = AlgoProto(
                    program=func,
                    language="python",
                    score=abs(score),
                )
            except:
                continue
            self.all_funcs.append(algo)

        # Deduplicate by program string to avoid trivial preference pairs
        # where chosen and rejected share identical code.
        if remove_duplicate:
            seen = {}
            for f in self.all_funcs:
                if f.program not in seen:
                    seen[f.program] = f
            self.all_funcs = list(seen.values())

        # Sort ascending so that index 0 = lowest fitness, index -1 = highest fitness.
        self.sorted_funcs = sorted(self.all_funcs, key=lambda x: x.score)
        self.splitted_subsets: List[List[AlgoProto]] = self._split_data_into_subsets(
            self.sorted_funcs
        )

    def __len__(self):
        return len(self.all_funcs)

    def _split_data_into_subsets(self, data):
        # Partition the sorted pool into M equally-sized subsets S_1 ... S_M
        # (paper Section 2.2). Subset 0 has the lowest fitness values, subset M-1
        # has the highest.
        def split_array(arr, n):
            k, m = divmod(len(arr), n)
            return [
                arr[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)
            ]

        splitted_subsets = split_array(data, n=self.number_of_subsets)
        for i in range(len(splitted_subsets)):
            print(
                f"Part {i}: "
                f"Length: {len(splitted_subsets[i])}, "
                f"Best Score={splitted_subsets[i][0].score}, "
                f"Worst Score={splitted_subsets[i][-1].score}",
            )
        return splitted_subsets

    def _create_a_pair(self, chosen_part_id: int) -> Dict:
        # chosen_part_id is drawn from [0, M-3]; the two highest-fitness subsets
        # (M-2 and M-1) are reserved exclusively as rejected sources so that
        # every chosen algorithm has at least one clearly worse tier available.
        assert 0 <= chosen_part_id <= self.number_of_subsets - 3

        # Skip the immediately adjacent subset (S_{i+1}) to enforce a minimum
        # quality gap between y+ and y-, yielding cleaner supervision signals.
        rejected_part_id = random.randint(
            chosen_part_id + 2, self.number_of_subsets - 1
        )

        chosen_algo = random.choice(self.splitted_subsets[chosen_part_id])
        if self.replace:
            self.splitted_subsets[chosen_part_id].remove(chosen_algo)
        rejected_algo = random.choice(self.splitted_subsets[rejected_part_id])
        if self.replace:
            self.splitted_subsets[rejected_part_id].remove(rejected_algo)
        return PromptChosenRejectedPairs(
            prompt=self.prompt, chosen=chosen_algo.program, rejected=rejected_algo.program
        ).to_dict()

    def create_dataset(self, npairs=10000, *, dataset_size=None):
        """Create npairs preference pairs using DAR sampling.

        Subset selection is biased toward higher-quality subsets via a
        softmax distribution with temperature τ (self.p): smaller τ sharpens
        the distribution toward the best subset; larger τ approaches uniform
        sampling over the first M-2 subsets (more diversity).

        If dataset_size is not None, duplicate these pairs until getting dataset_size.
        """
        if dataset_size is not None:
            assert npairs < dataset_size

        # Pr(i) ∝ exp((M-2-i) / τ): higher-indexed subsets (better fitness)
        # receive higher sampling probability after the [::-1] reversal.
        p = [i for i in range(self.number_of_subsets - 3)]
        p = [math.exp(i / self.p) for i in p]
        p = [i / sum(p) for i in p][::-1]
        print(f"Probabilities sampling from each subset: {[round(i, 2) for i in p]}")
        chosen_subset_ids = list(range(self.number_of_subsets - 3))
        data = []
        while len(data) < npairs:
            chosen_subset_id = np.random.choice(chosen_subset_ids, p=p)
            pair = self._create_a_pair(chosen_subset_id)
            data.append(pair)
        if dataset_size is not None:
            data_ = copy.deepcopy(data)
            while dataset_size - len(data_) >= len(data):
                data_.extend(data)
            while len(data_) < dataset_size:
                data_.append(random.choice(data))
            print(len(data_))
            return data_
        return data


if __name__ == "__main__":
    template = '''
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    return -(bins - item)
    '''.strip()
    with open("../process_dataset/process_nips_data/obp_nips_expt.pkl", "rb") as f:
        data = pickle.load(f)
    data_creator = DatasetCreator(
        data, template, prompt="", remove_duplicate=True
    )
