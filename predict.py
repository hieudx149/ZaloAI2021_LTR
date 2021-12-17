import argparse
import logging
import os
import json
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from utils import MODEL_CLASSES, init_logger, load_tokenizer
import numpy as np
from data_loader import InputExample

logger = logging.getLogger(__name__)


def get_device(config):
    return "cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu"


def get_args(config):
    return torch.load(os.path.join(config.model_dir, "training_args.bin"))


def load_model(config, args, device):
    # Check whether model exists
    if not os.path.exists(config.model_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        model = MODEL_CLASSES[args.model_type][1].from_pretrained(args.model_dir, args=args)
        model.to(device)
        model.eval()
        logger.info("***** Model Loaded *****")
    except Exception:
        raise Exception("Some model files might be missing...")

    return model


def create_batch_test(question_articles):
    examples = []
    question_id = question_articles["question_id"]
    question = question_articles["question"]
    for article in question_articles["articles"]:
        law_id = article["law_id"]
        article_id = article["article_id"]
        title = article["title"]
        text = article["text"]
        examples.append(InputExample(question_id=question_id, question_text=question,
                                     law_id=law_id, article_id=article_id, title_text=title,
                                     article_text=text, is_relevant=None))
    return examples


def convert_batch_test_to_features(examples, tokenizer, args):
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    for example_index, example in tqdm(enumerate(examples)):
        query_tokens = tokenizer.tokenize(example.question_text)
        title_tokens = tokenizer.tokenize(example.title_text)
        context_tokens = tokenizer.tokenize(example.article_text)

        if len(query_tokens) > args.max_question_len:
            query_tokens = query_tokens[0:args.max_question_len]

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_context = args.max_seq_len - len(query_tokens) - len(title_tokens) - 3

        if len(context_tokens) > max_tokens_for_context:
            context_tokens = context_tokens[0:max_tokens_for_context]

        tokens = [tokenizer.cls_token] + query_tokens + [tokenizer.sep_token] + title_tokens + context_tokens + \
                 [tokenizer.sep_token]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(context_tokens) + len(title_tokens) + 1)

        # Zero-pad up to the sequence length.
        padding = [tokenizer.pad_token_id] * (args.max_seq_len - len(input_ids))

        input_mask += [0] * (args.max_seq_len - len(input_ids))
        segment_ids += [0] * (args.max_seq_len - len(input_ids))
        input_ids += padding

        assert len(input_ids) == args.max_seq_len
        assert len(input_mask) == args.max_seq_len
        assert len(segment_ids) == args.max_seq_len

        if example_index < 5:
            logger.info("*** Example ***")
            logger.info("question_id: %s" % example.question_id)
            logger.info("question: {}".format(' '.join(query_tokens)))
            logger.info("tokens: {}".format(' '.join(tokens)))
            logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
            logger.info("input_mask: {}".format(' '.join(map(str, input_mask))))
            logger.info("segment_ids: {}".format(' '.join(map(str, segment_ids))))

        all_input_ids.append(input_ids)
        all_input_mask.append(input_mask)
        all_segment_ids.append(segment_ids)

    # convert to tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    return dataset


def predict(config):
    # load model and args
    args = get_args(pred_config)
    device = get_device(pred_config)
    tokenizer = load_tokenizer(args)
    model = load_model(pred_config, args, device)
    logger.info(args)
    output = []
    with open(config.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    for entry in data:
        batch_test_data = create_batch_test(entry)
        batch_test_features = convert_batch_test_to_features(batch_test_data, tokenizer, args)
        single_output = {"question_id": batch_test_data[0].question_id, "relevant_articles": []}
        # Predict
        print(len(batch_test_features))
        data_loader = DataLoader(batch_test_features, batch_size=len(batch_test_features))
        for batch in tqdm(data_loader, desc="Predicting"):
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1]}
                if args.model_type != "xlm_roberta":
                    inputs["token_type_ids"] = batch[2]
                outputs = model(**inputs)
                _, relevant_logits = outputs
                relevant_logits = relevant_logits.detach().cpu().numpy()
                relevant_preds = np.argmax(relevant_logits, axis=1)
                count = 0
                for i in range(0, len(relevant_preds)):
                    if relevant_logits[i] == 1:
                        count = 1
                        single_output["relevant_articles"].append({"law_id": batch_test_data[i].law_id,
                                                                   "article_id": batch_test_data[i].article_id})
                if count == 0:
                    single_output["relevant_articles"].append({"law_id": batch_test_data[0].law_id,
                                                               "article_id": batch_test_data[0].article_id})
                    single_output["relevant_articles"].append({"law_id": batch_test_data[1].law_id,
                                                               "article_id": batch_test_data[1].article_id})
                output.append(single_output)

    with open(config.output_file, "w", encoding='utf-8') as f_w:
        json.dump(output, f_w, ensure_ascii=False, indent=4)

    return output


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default="data/test_submit_tokenize.json", type=str,
                        help="Input file for prediction")
    parser.add_argument("--output_file", default="data/sample_pred_out.json", type=str,
                        help="Output file for prediction")
    parser.add_argument("--model_dir", default="checkpoint", type=str, help="Path to save, load model")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    pred_config = parser.parse_args()
    predict(pred_config)
