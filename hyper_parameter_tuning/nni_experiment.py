import optparse
import sys
import os
import logging
import nni
sys.path.insert(1, os.path.join("..", "src_code"))
from activator import ENCODER_DECODER, ATTENTION, MTActivator
logger = logging.getLogger("NNI_logger")


def run_trial(params, model_type):
    params = {
        "encoder": {
            "num_layers": 1,
            "hidden_dim": params["encoder_hidden"],
            "embed_dim": params["embed_dim"],
            "dropout": params["dropout"]
        },
        "decoder": {
            "num_layers": 1,
            "hidden_dim": params["decoder_hidden"],
            "embed_dim": params["embed_dim"],
            "dropout": params["dropout"]},
        "optimizer": {
            "type": "Adam",
            "kwargs": {
                "lr": params["learning_rate"],
                "weight_decay": params["regularization"]
            }
        },
        "train": {
            "epochs": 10,
            "batch_size": 1
        },
        "data": {
            "train_source": "../data/train.src", "train_target": "../data/train.trg",
            "dev_source": "../data/dev.src", "dev_target": "../data/dev.trg",
            "test_source": "../data/test.src", "test_target": "../data/test.src"
        }
    }
    activator = MTActivator(model_type, params)
    activator.fit()


def main(model_type):
    try:
        # get parameters form tuner
        params = nni.get_next_parameter()
        logger.debug(params)
        run_trial(params, model_type)
    except Exception as exception:
        logger.error(exception)
        raise


def parse_input():
    optparser = optparse.OptionParser()
    optparser.add_option("--attention", dest="attention",
                         action='store_true', help="Use an attention based model")
    optparser.add_option("--encoder_decoder", dest="encoder_decoder",
                         action='store_true', help="Use a basic Encoder-Decoder based model")
    (opts, args) = optparser.parse_args()

    if opts.attention and opts.encoder_decoder:
        print("Error: Only a single model-type can pe picked, "
              "drop one of the args:\n\t1. --attention\n\t2. --encoder_decoder")
        exit(1)
    model_type = ENCODER_DECODER if opts.encoder_decoder else ATTENTION
    return model_type


if __name__ == "__main__":
    model_type = parse_input()
    main(model_type)

