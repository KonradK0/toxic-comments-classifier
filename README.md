# Installation

Run:
```
conda env create -f env.yml -p /home/user/anaconda3/envs/env_name
source activate env_name
pip install -r requirements.txt
```

# Datasets

The dataset used to train the classifier has been obtained from following sources:

* [Twitter comments](https://github.com/t-davidson/hate-speech-and-offensive-language/blob/master/data/labeled_data.csv)
* [Quora questions](https://www.kaggle.com/c/quora-insincere-questions-classification)
* [Wikipedia comments](https://figshare.com/articles/Wikipedia_Talk_Labels_Toxicity/4563973)
* [More Wikipedia comments](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

Datasets were preprocessed and standardized.