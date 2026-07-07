# Predicting Writing Quality

- Competition link: [Writing Process to Writing Quality](https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality/overview)

Follow these steps to get started:

1. Get the competition data.
``` shell
export COMP=linking-writing-processes-to-writing-quality
docker exec -it kaggle-notebooks-gpu kaggle competitions download -c $COMP -p data
docker exec -it kaggle-notebooks-gpu bash -c "cd data && unzip -o $COMP.zip && rm $COMP.zip"
```
