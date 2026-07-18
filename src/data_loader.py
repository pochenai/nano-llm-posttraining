from datasets import load_dataset, Dataset
from rich.console import Console
from rich.table import Table

from . import dprint


def load_sftdata(
    split="train", source="smol-contraints", dataset_name="HuggingFaceTB/smol-smoltalk"
):
    """Download the full dataset (cached) and keep only rows of a given source.

    smol-smoltalk mixes several sub-datasets in one config, distinguished by the
    `source` column. IFEval-style instruction-following data is "smol-contraints"
    (note the upstream typo). Pass source=None to keep everything.
    """
    ds = load_dataset(dataset_name, split=split)
    if source is not None:
        ds = ds.filter(lambda x: x["source"] == source)
    dprint(f"Loaded {len(ds)} examples from {dataset_name} (source={source})")
    return ds


def display_dataset(dataset, n=3):
    # Render a few examples as a rich table (long text wraps inside each cell).
    table = Table(show_lines=True, title=f"Dataset preview (first {n})")
    table.add_column("#", style="cyan", no_wrap=True)
    table.add_column("User Prompt", style="green", overflow="fold")
    table.add_column("Assistant Response", overflow="fold")

    for i in range(min(n, len(dataset))):
        example = dataset[i]
        user_msg = next(
            m["content"] for m in example["messages"] if m["role"] == "user"
        )
        assistant_msg = next(
            m["content"] for m in example["messages"] if m["role"] == "assistant"
        )
        table.add_row(str(i), user_msg, assistant_msg)

    Console().print(table)


if __name__ == "__main__":
    # Load only the IFEval (smol-contraints) subset for train and test.
    train = load_sftdata(split="train")
    test = load_sftdata(split="test")
    print("train:", train)
    print("test:", test)
    display_dataset(train)
