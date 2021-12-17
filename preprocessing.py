import re
import json
from pyvi import ViTokenizer
from utils import preprocess_khoan, preprocess_question, preprocess_article_title


def tokenize_text(text):
    return ViTokenizer.tokenize(text)


punc = """!"#$%&'()*+,-./:;<=>?@[\]^`{|}~"""  # noqa: W605
table = str.maketrans("", "", punc)


def clean_text(text):
    words = text.lower().split()
    result = [w.translate(table) for w in words]
    stripped = " ".join(result)
    result = " ".join(stripped.split())
    return result


def clean_train_data(data):
    # clean data
    for entry in data:
        entry["question"] = preprocess_question(
            entry["question"], remove_end_phrase=False
        )
        for article in entry["relevant_articles"]:
            article["title"] = preprocess_article_title(article["title"])
            cac_khoan = article["text"].split("\n")
            khoan_clean = []
            for khoan in cac_khoan:
                khoan = preprocess_khoan(khoan)
                khoan_clean.append(khoan.strip())
            article["text"] = " ".join(khoan_clean)
        for article in entry["non_relevant_articles"]:
            cac_khoan = article["text"].split("\n")
            khoan_clean = []
            for khoan in cac_khoan:
                khoan = preprocess_khoan(khoan)
                khoan_clean.append(khoan.strip())
            article["text"] = " ".join(khoan_clean)
    # tokenize data for phobert
    for entry in data:
        entry["question"] = clean_text(tokenize_text(entry["question"]))
        for article in entry["relevant_articles"]:
            article["title"] = clean_text(tokenize_text(article["title"]))
            article["text"] = clean_text(tokenize_text(article["text"]))
        for article in entry["non_relevant_articles"]:
            article["title"] = clean_text(tokenize_text(article["title"]))
            article["text"] = clean_text(tokenize_text(article["text"]))
    return data


def clean_test_data(data):
    # clean data
    for entry in data:
        entry["question"] = preprocess_question(
            entry["question"], remove_end_phrase=False
        )
        for article in entry["articles"]:
            article["title"] = preprocess_article_title(article["title"])
            cac_khoan = article["text"].split("\n")
            khoan_clean = []
            for khoan in cac_khoan:
                khoan = preprocess_khoan(khoan)
                khoan_clean.append(khoan.strip())
            article["text"] = " ".join(khoan_clean)
    # tokenize data for phobert
    for entry in data:
        entry["question"] = clean_text(tokenize_text(entry["question"]))
        for article in entry["articles"]:
            print(article["title"])
            print(tokenize_text(article["title"]))
            article["title"] = clean_text(tokenize_text(article["title"]))
            article["text"] = clean_text(tokenize_text(article["text"]))
    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--step", type=str, required=True)
    args = parser.parse_args()
    step = args.step
    if step == "train":
        with open(
            "./data/train_data_elasticsearch.json", "r", encoding="utf-8"
        ) as f_train:
            train_data = json.load(f_train)
        train_data = clean_train_data(train_data)
        with open("./data/train_data_model.json", "w", encoding="utf-8") as f_train:
            json.dump(train_data, f_train, ensure_ascii=False, indent=4)
    elif step == "test":
        with open(
            "./data/test_data_elasticsearch.json", "r", encoding="utf-8"
        ) as f_test:
            test_data = json.load(f_test)
        test_data = clean_test_data(test_data)
        with open("./data/test_data_model.json", "w", encoding="utf-8") as f_test:
            json.dump(test_data, f_test, ensure_ascii=False, indent=4)
    elif step == "test_private":
        with open(
            "./data/test_private_data_elasticsearch.json", "r", encoding="utf-8"
        ) as f_test:
            test_data = json.load(f_test)
        test_data = clean_test_data(test_data)
        with open(
            "./data/test_private_data_model.json", "w", encoding="utf-8"
        ) as f_test:
            json.dump(test_data, f_test, ensure_ascii=False, indent=4)
