from datasets import DatasetDict
import re
import contractions
from nltk.corpus import stopwords
import string
import nltk

def custom_dataset_size(dataset, size, split_ratio=(0.7, 0.15, 0.15)):
    assert isinstance(dataset, DatasetDict), "Input dataset must be a DatasetDict"
    assert isinstance(size, int), "Size must be an integer"
    assert size > 0, "Size must be greater than 0"
    assert len(split_ratio) == 3, "Split ratio must have 3 elements"
    assert abs(sum(split_ratio) - 1.0) < 1e-6, "Split ratio must sum to 1"

    train_size = round(size*split_ratio[0])
    val_size = round(size*split_ratio[1])
    test_size = round(size*split_ratio[2])

    shuffled_train = dataset["train"].shuffle(seed=42)
    shuffled_test = dataset["test"].shuffle(seed=42)

    subset = DatasetDict(
        train = shuffled_train.select(range(train_size)),
        val = shuffled_train.select(range(train_size, train_size + val_size)),
        test = shuffled_test.select(range(test_size))
    )

    return subset


# patterns of metadata that need to be removed
pattern_1 = r"^[^(]*\([^\)]*\)\s*--\s*"     #LONDON (Reuters) --
pattern_2 = r"^.*UPDATED:\s+\.\s+\d{2}:\d{2}\s+\w+,\s+\d+\s+\w+\s+\d{4}\s+\.\s+"        #UPDATED: . 14:35 Tuesday, 28 May 2024 .
pattern_3 = r"^By\s+\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+\.\s+(?:and\s+Associated\s+Press\s+Reporter\s*\.\s*)?"     #By . John Smith . and Associated Press Reporter .
pattern_4 = r"^(\([^\)]*\))"        #(NEW YORK)
pattern_5 = r"^By\s+\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+\.\s+PUBLISHED:\s+\.\s+\d{2}:\d{2}\s+EST,\s+\d{1,2}\s+[A-Za-z]+\s+\d{4}\s+\.\s+\|\s+\.\s+UPDATED:\s+\.\s+\d{2}:\d{2}\s+EST,\s+\d{1,2}\s+[A-Za-z]+\s+\d{4}\s+\.\s+"     #By . John Smith . PUBLISHED: . 12:00 EST, 23 May 2024 . | . UPDATED: . 13:45 EST, 23 May 2024 .


metadata_patterns = [pattern_1, pattern_2, pattern_3, pattern_4, pattern_5]

def minimal_preprocessing(article):
    # removing metadata
    for pattern in metadata_patterns:
        article = re.sub(pattern, '', article)
    # handling contractions
    article = contractions.fix(article)
    # removing extraneous white space
    article = article.strip()

    # calling functions for additional preprocessing
    #article = lowercase(article)
    #article = remove_stopwords(article)
    #article = remove_punctuation(article)

    return article


# other functions for preprocessing that could be called
def lowercase(article):
   article = article.lower()
   return article

def remove_stopwords(article):
    # outcomment if necessary to download stopwords from nltk
    #nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))
    article = " ".join([word for word in article.split() if word not in stop_words])
    return article

def remove_punctuation(article):
    article = article.translate(str.maketrans("", "", string.punctuation))
    return article