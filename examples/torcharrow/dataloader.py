import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torcharrow as ta
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torcharrow import functional
from torchrec.datasets.criteo import (
    DEFAULT_INT_NAMES,
    DEFAULT_CAT_NAMES,
    DEFAULT_LABEL_NAME,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class TorchArrowDataset(Dataset):
    def __init__(self, torcharrow_df):
        self.torcharrow_df = torcharrow_df

    def __len__(self):
        return len(self.torcharrow_df)

    def __getitem__(self, idx):
        return self.torcharrow_df[idx]


def get_dataloader(
    parquet_files, world_size, rank, num_embeddings=4096, salt=0, batch_size=16
):
    pq_tables = [pq.read_table(file) for file in parquet_files]
    pq_df = pa.concat_tables(pq_tables)
    df = ta.from_arrow(pq_df)

    def preproc(df, max_idx=num_embeddings, salt=salt):
        for int_name in DEFAULT_INT_NAMES:
            df[int_name] = (df[int_name] + 3).log()
        # construct a sprase index from a dense one
        df["bucketize_int_0"] = functional.array_constructor(
            functional.bucketize(df["int_0"], [0.5, 1.0, 1.5])
        )
        for cat_name in DEFAULT_CAT_NAMES:
            # hash our embedding index into our embedding tables
            df[cat_name] = functional.sigrid_hash(df[cat_name], salt, num_embeddings)
            df[cat_name] = functional.array_constructor(df[cat_name])
            df[cat_name] = functional.firstx(df[cat_name], 1)
        return df

    df = preproc(df)
    dataset = TorchArrowDataset(df)

    def criteo_collate(samples):
        batched_dense_values = []
        kjt_keys = DEFAULT_CAT_NAMES + ["bucketize_int_0"]
        kjt_values = []
        kjt_lengths = []
        labels = []
        for sample in samples:
            dense_values = []
            for column_name, value in zip(df.columns, sample):
                if column_name in DEFAULT_INT_NAMES:
                    dense_values.append(value)
                if column_name == DEFAULT_LABEL_NAME:
                    labels.append([value])
                if column_name in kjt_keys:
                    kjt_values.extend(value)
                    kjt_lengths.append(len(value))
            batched_dense_values.append(dense_values)
        batched_dense_values = torch.tensor(batched_dense_values)
        labels = torch.tensor(labels)
        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=kjt_keys,
            values=torch.tensor(kjt_values),
            lengths=torch.tensor(kjt_lengths),
        )

        return batched_dense_values, labels, kjt

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=criteo_collate,
        drop_last=False,
        pin_memory=True,
    )
