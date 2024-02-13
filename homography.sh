cd ./DeepLSD
python -m deeplsd.scripts.homography_adaptation_df \
"./data/engisense-lines/train.txt" \
"./data/engisense-homographies" \
"./data/engisense-lines/labels" \
--num_H 100  --n_jobs 4
