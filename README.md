# Predicting Writing Quality

- Competition link: [Writing Process to Writing Quality](https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality/overview)

Follow these steps to get started:

1. Get the competition data.

``` shell
# Enter the container
docker exec -it kaggle-notebooks-gpu /bin/bash

# Download raw data
export COMP=linking-writing-processes-to-writing-quality
kaggle competitions download -c $COMP -p data
bash -c "cd data && unzip -o $COMP.zip && rm $COMP.zip"

# Generate features
uv run scripts/silver_bullet_feats_v1.py
```
