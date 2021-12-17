import json
from tqdm import tqdm
import logging
import torch
import os
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, law_id, article_id, question_id,  question_text,
                 title_text, article_text, is_relevant):
        self.law_id = law_id
        self.article_id = article_id
        self.question_id = question_id
        self.question_text = question_text
        self.title_text = title_text
        self.article_text = article_text
        self.is_relevant = is_relevant


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


def create_examples(input_file):
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)

    examples = []
    for entry in input_data:
        question_text = entry['question']
        question_id = entry['question_id']
        len_relevant = len(entry['relevant_articles'])
        for article in entry['relevant_articles']:
            law_id = article['law_id']
            article_id = article['article_id']
            title_text = article['title']
            article_text = article['text']
            is_relevant = True
            num_over_sampler = 10 // len_relevant
            for _ in range(0, num_over_sampler):
                examples.append(InputExample(law_id, article_id, question_id,
                                             question_text, title_text, article_text,
                                             is_relevant))
        for article in entry['non_relevant_articles']:
            law_id = article['law_id']
            article_id = article['article_id']
            title_text = article['title']
            article_text = article['text']
            is_relevant = False
            examples.append(InputExample(law_id, article_id, question_id,
                                         question_text, title_text, article_text,
                                         is_relevant))
    return examples


def convert_examples_to_features(examples, tokenizer, args):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
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
        label_ids = int(example.is_relevant)

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
            logger.info("is relevant: %d", label_ids)

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_ids=label_ids)
        )
    return features


def load_and_cache_examples(args, tokenizer, evaluate_mode=False):
    # Load data features from cache or dataset file
    input_file = args.evaluate_file if evaluate_mode else args.train_file
    cached_features_file = os.path.join(os.path.dirname(input_file), 'cached_cls_{}_{}_{}'.format(
        'dev' if evaluate_mode else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_len)))

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", input_file)
        examples = create_examples(input_file)
        features = convert_examples_to_features(examples, tokenizer, args)

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    return dataset

