import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class FeatureCandidate:
    island_id: int
    score: float
    function_code: str
    sample_order: int
    source_file: str

    def __hash__(self) -> int:
        return hash(self.function_code)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FeatureCandidate):
            return False
        return self.function_code == other.function_code


def load_candidates(samples_dir: str) -> List[FeatureCandidate]:
    """
    Load feature candidates from JSON files
    """

    if not os.pathexists(samples_dir):
        raise ValueError(f"Sample directory {samples_dir} does not exist")
    files = glob.glob(os.path.join(samples_dir, "*.json"))
    if not files:
        raise ValueError(f"No JSON files found in {samples_dir}")
    candidates = []
    
    for fp in files:
        try:
            with open(fp, 'r') as f:
                data = json.load(f)
                
                if all(k in data for k in ('island_id', 'score', 'function_code', 'sample_order')):
                    candidate = FeatureCandidate(
                        island_id=data['island_id'],
                        score=data['score'],
                        function_code=data['function_code'],
                        sample_order=data['sample_order'],
                        source_file=os.path.basename(fp)
                    )
                    candidates.append(candidate)
        except Exception as e:
            print(f"Error loading {fp}: {e}")
            continue
    if not candidates: 
        raise ValueError(f"No valid candidates found in {samples_dir}")
    print(f"Loaded {len(candidates)} candidates from {samples_dir}")
    return candidates
    
