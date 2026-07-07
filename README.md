# Predicting Writing Quality

- Competition link: [Writing Process to Writing Quality](https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality/overview)

Follow these steps to get started:

``` shell
# Enter the container
docker compose --profile gpu up
docker exec -it kaggle-notebooks-gpu /bin/bash

# Download raw data
export COMP=linking-writing-processes-to-writing-quality
kaggle competitions download -c $COMP -p data
bash -c "cd data && unzip -o $COMP.zip && rm $COMP.zip"

# Generate features
uv run scripts/silver_bullet_feats_v1.py

# Train the GBM model
uv run scripts/train_ensemble_gluon.py

# Submit the results
make sync linking-writing-processes-to-writing-quality rfvasile/<notebook-dir>
kaggle kernels push -p <notebook-dir>
```
