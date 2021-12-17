import argparse
from data_loader import load_and_cache_examples
from trainer import Trainer
from utils import MODEL_CLASSES, MODEL_PATH_MAP, init_logger, load_tokenizer, set_seed


def main(args):
    init_logger()
    set_seed(args)
    tokenizer = load_tokenizer(args)

    train_dataset = load_and_cache_examples(args, tokenizer)
    trainer = Trainer(args, train_dataset)

    if args.do_train:
        trainer.train()
        trainer.save_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # require parameter
    parser.add_argument("--model_type", default="xlm_roberta", required=True, type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, required=True, type=str,
                        help="name or path of pretrained language model")
    parser.add_argument("--model_dir", default=None, required=True, type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument("--train_file", default="sub_test_data.json", type=str, help="dir of train data")
    parser.add_argument("--evaluate_file", default="sub_test_data.json", type=str, help="dir of evaluate data")
    parser.add_argument("--token_level", type=str, default="syllable-level",
                        help="Tokens are at syllable level or word level (Vietnamese) [word-level, syllable-level]")
    # store true parameter
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--do_eval_dev", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--do_lower_case", action="store_true", help="do lower case data or not.")
    # tuning parameter
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=12, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=12, type=int, help="Batch size for evaluation.")
    parser.add_argument("--num_labels", default=1, type=int,
                        help="number labels should be 1 for sigmoid 2 for softmax.")
    parser.add_argument("--max_question_len", default=50, type=int,
                        help="The maximum total input question length after tokenization.")
    parser.add_argument("--max_seq_len", default=512, type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=2, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",)
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X updates steps.")
    parser.add_argument("--gpu_id", type=int, default=0, help="Select gpu id")

    args_parse = parser.parse_args()

    args_parse.model_name_or_path = MODEL_PATH_MAP[args_parse.model_type]
    main(args_parse)
