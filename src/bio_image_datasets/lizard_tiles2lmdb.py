import os
import lmdb
import pickle

from tqdm import tqdm
from bio_image_datasets.lizard_dataset import LizardDataset
from bio_image_datasets.lizard2tiles import transform_to_tiles

def create_lmdb_for_split(dataset, split_number, lmdb_path, tile_size=224):
    """
    Create an LMDB database for a specific split.

    Args:
        dataset: The LizardDataset instance.
        split_number: The split number (1, 2, or 3).
        lmdb_path: Path to the LMDB file to create.
        tile_size: The size of the tiles to create.
    """
    import numpy as np

    # Get indices for the current split
    indices = [idx for idx in range(len(dataset)) if dataset.get_sample_split(idx) == split_number]
    print(f"Number of samples in split {split_number}: {len(indices)}")

    # Calculate the total number of tiles to estimate the map size for LMDB
    # total_tiles = 0
    # for idx in tqdm(indices):
    #     tiles = transform_to_tiles(dataset, idx, tile_size=tile_size)
    #     total_tiles += len(tiles)

    # print(f"Total number of tiles in split {split_number}: {total_tiles}")

    # Estimate LMDB map size (adjust as needed)
    map_size = 10000 * 224 * 224 * 10  # Rough estimate: 10 bytes per pixel

    # Create the LMDB environment
    env = lmdb.open(lmdb_path, map_size=map_size)

    tile_idx = 0  # Global tile index for keys

    with env.begin(write=True) as txn:
        for idx in tqdm(indices):
            tiles = transform_to_tiles(dataset, idx, tile_size=tile_size)
            for tile in tiles:
                # Serialize the tile dictionary
                tile_data = pickle.dumps(tile)

                # Use the tile_idx as the key, converted to bytes
                key = f"{tile_idx:08}".encode('ascii')

                # Put the data into LMDB
                txn.put(key, tile_data)

                tile_idx += 1

                if tile_idx % 1000 == 0:
                    print(f"Processed {tile_idx} tiles...")

    env.close()
    print(f"LMDB file created at {lmdb_path} with {tile_idx} tiles.")

if __name__ == '__main__':
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Create LMDB files for LizardDataset tiles.')
    parser.add_argument('--local_path', type=str, default='~/projects/lab_hackathon_2024/Bio-Image-Datasets/downloads',
                        help='Local path to the dataset.')
    parser.add_argument('--output_dir', type=str, default='~/projects/lab_hackathon_2024/Bio-Image-Datasets/downloads/lizard_lmdb/',
                        help='Output directory for LMDB files.')
    parser.add_argument('--tile_size', type=int, default=224,
                        help='Tile size (default: 224).')

    args = parser.parse_args()

    # Expand user in paths
    args.local_path = os.path.expanduser(args.local_path)
    args.output_dir = os.path.expanduser(args.output_dir)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Instantiate the dataset
    dataset = LizardDataset(local_path=args.local_path)

    # Process each split
    for split_number in [1, 2, 3]:
        lmdb_filename = os.path.join(args.output_dir, f'lizard_split_{split_number}.lmdb')
        print(f"\nProcessing split {split_number}...")

        create_lmdb_for_split(dataset, split_number, lmdb_filename, tile_size=args.tile_size)
