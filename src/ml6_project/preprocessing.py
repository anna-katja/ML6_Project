from datasets import DatasetDict
from datasets import load_dataset
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

    train_size = round(size * split_ratio[0])
    val_size = round(size*(split_ratio[0]+split_ratio[1]))
    test_size = round(size*split_ratio[2])

    subset = DatasetDict(
        train = dataset["train"].shuffle(seed=42).select(range(train_size)),
        val = dataset["train"].shuffle(seed=42).select(range(train_size, val_size)),
        test = dataset["test"].shuffle(seed=42).select(range(test_size))
    )

    return subset

# patterns of metadata that needs to be removed

pattern_1 = r"^[^(]*\([^\)]*\)\s*--\s*"
pattern_2 = r"^.*UPDATED:\s+\.\s+\d{2}:\d{2}\s+\w+,\s+\d+\s+\w+\s+\d{4}\s+\.\s+"
pattern_3 = r"^By\s+\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+\.\s+(?:and\s+Associated\s+Press\s+Reporter\s*\.\s*)?"
pattern_4 = r"^(\([^\)]*\))"
pattern_5 = r"^By\s+\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+\.\s+PUBLISHED:\s+\.\s+\d{2}:\d{2}\s+EST,\s+\d{1,2}\s+[A-Za-z]+\s+\d{4}\s+\.\s+\|\s+\.\s+UPDATED:\s+\.\s+\d{2}:\d{2}\s+EST,\s+\d{1,2}\s+[A-Za-z]+\s+\d{4}\s+\.\s+"

metadata_patterns = [pattern_1, pattern_2, pattern_3, pattern_4, pattern_5]

def minimal_preprocessing(article):
    # removing metadata
    """for pattern in metadata_patterns:
        data["article"] = re.sub(pattern, '', data["article"])
    # handling contractions
    data["article"] = contractions.fix(data["article"])
    #data["highlights"] = contractions.fix(data["highlights"])
    # removing extraneous white space
    data["article"] = data["article"].strip()
    #data["highlights"] = data["highlights"].strip()

    return data"""

    for pattern in metadata_patterns:
        article = re.sub(pattern, '', article)
    article = contractions.fix(article)
    #article = lowercase(article)
    article = remove_stopwords(article)
    article = remove_punctuation(article)
    article = article.strip()
    return article


# other functions for preprocessing that could be called
#def lowercase(article):
   # article = article.lower()
    #data["highlights"] = data["highlights"].lower()
  #  return article

# outcomment if necessary
def remove_stopwords(article):
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))
    article = " ".join([word for word in article.split() if word not in stop_words])
    #data["highlights"] = " ".join([word for word in data["highlights"].split() if word not in stop_words])
    return article

def remove_punctuation(article):
    article = article.translate(str.maketrans("", "", string.punctuation))
    #data["highlights"].translate(str.maketrans("", "", string.punctuation))
    return article