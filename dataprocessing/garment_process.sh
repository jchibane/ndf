#!/usr/bin/env bash

wget https://datasets.d2.mpi-inf.mpg.de/MultiGarmentNetwork/Multi-Garmentdataset.zip
wget https://datasets.d2.mpi-inf.mpg.de/MultiGarmentNetwork/Multi-Garmentdataset_02.zip

unzip Multi-Garmentdataset.zip
unzip Multi-Garmentdataset_02.zip

rm Multi-Garment_dataset.zip
rm Multi-Garment_dataset_02.zip

mkdir "../datasets/garments_data"
conda activate pymesh

python garment_normalize.py --input_folder "./Multi-Garment_dataset" --output_folder "../datasets/garments_data"
python garment_normalize.py --input_folder "./Multi-Garment_dataset_02" --output_folder "../datasets/garments_data"

python garment_preprocess.py --input_path "../datasets/garments_data" --res 256 --inp_points 3000 --sample_points 100000 --sigmas 0.01 0.02 0.08
python garment_preprocess.py --input_path "../datasets/garments_data" --res 256 --inp_points 300 --sample_points 100000 --sigmas 0.01 0.02 0.08

