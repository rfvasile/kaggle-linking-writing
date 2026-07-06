- Competition link: [Writing Process to Writing Quality](https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality/overview)

Follow these steps to get started:

1. Get the competition data.
``` shell
docker exec -it kaggle-notebooks-gpu kaggle competitions download -c linking-writing-processes-to-writing-quality -p data
docker exec -it kaggle-notebooks-gpu bash -c "cd data && unzip -o linking-writing-processes-to-writing-quality.zip && rm linking-writing-processes-to-writing-quality.zip"
```
