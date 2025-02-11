import os
import pandas as pd
from bio_image_datasets.pannuke_dataset import PanNukeDataset

if __name__ == "__main__":
    dataset = PanNukeDataset(local_path='/fast/AG_Kainmueller/data/pannuke_cp')
    samples_names = dataset.get_sample_names()

    # Create a list of (filename, split) tuples
    split_entries = []

    for sample_name in samples_names:
        if 'fold3' in sample_name:
            split_entries.append((sample_name, 'train'))
        elif 'fold2' in sample_name:
            split_entries.append((sample_name, 'test'))
        else:
            split_entries.append((sample_name, 'valid'))

    print('n samples:', len(split_entries))

    # Create DataFrame and save to CSV
    df = pd.DataFrame(split_entries, columns=['sample_name', 'train_test_val_split'])
    df = df.sort_values('sample_name')  # Optional: sort by filename

    # Save to CSV
    df.to_csv('/fast/AG_Kainmueller/data/patho_foundation_model_bench_data/pannuke/train_test_val_split.csv', index=False)