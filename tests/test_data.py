"""Dataset tests. The pickling test guards a real Windows-only failure: with
num_workers>0 the DataLoader spawns workers by pickling the dataset, so the
dataset must not hold unpicklable references (e.g. a module object)."""
import pickle

from src.data import ASVspoofDataset, Utterance, parse_protocol


def test_dataset_is_picklable():
    # Simulates what a Windows DataLoader worker does at spawn time.
    items = [Utterance(path="x.flac", label=1, attack="-")]
    ds = ASVspoofDataset(items, train=True)
    restored = pickle.loads(pickle.dumps(ds))
    assert len(restored) == 1
    assert restored.train is True


def test_dataset_len_matches_items():
    items = [Utterance(path=f"{i}.flac", label=i % 2, attack="-") for i in range(7)]
    assert len(ASVspoofDataset(items)) == 7
