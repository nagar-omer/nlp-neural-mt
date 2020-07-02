import sys
sys.path.insert(0, "src_code")
import json
import optparse
import os
from time import strptime
import logging
import torch
from activator import MTActivator
logging.getLogger().setLevel(logging.INFO)


def validate(model_type, params, checkpoint):
    activator = MTActivator(model_type, params, checkpoint=checkpoint)
    activator.validate_test()


def get_last_created_model():
    checkpoints = []
    attention_checkpoints = os.path.join("models_state_dict", "Attention_model")
    encoder_decoder_checkpoints = os.path.join("models_state_dict", "EncoderDecoder_model")
    if os.path.exists(encoder_decoder_checkpoints):
        checkpoints += [(os.path.join(encoder_decoder_checkpoints, f), f.split(" ", 1)[1].rsplit(".", 1)[0]) for f in
                        os.listdir(encoder_decoder_checkpoints)]
    if os.path.exists(attention_checkpoints):
        checkpoints += [(os.path.join(attention_checkpoints, f), f.split(" ", 1)[1].rsplit(".", 1)[0]) for f in
                        os.listdir(attention_checkpoints)]
    if not checkpoints:
        return None
    return sorted(checkpoints, key=lambda x: strptime(x[1], "%Y-%B-%d %H-%M"))[-1][0]


def parse_input():
    optparser = optparse.OptionParser()
    optparser.add_option("--checkpoint", dest="checkpoint", help="Path to model")
    optparser.add_option("-p", "--params", dest="params", default=None, help="path to parameters file - optional")
    (opts, args) = optparser.parse_args()

    if opts.checkpoint is None:
        checkpoint = get_last_created_model()
        if checkpoint is None:
            print("Error: cant find trained models, specify --checkpoint")
            exit(1)
        logging.info("loading pre-trained model: " + checkpoint)
    else:
        if not os.path.exists(opts.checkpoint):
            print("Error: file " + opts.checkpoint + " does not exist")
            exit(1)
        checkpoint = opts.checkpoint
    checkpoint = torch.load(checkpoint)
    params = checkpoint["params"]
    model_type = checkpoint["model_type"]

    if opts.params is not None:
        if not os.path.exists(opts.params):
            print("Error: file " + opts.params + " does not exist")
            exit(1)
        user_params = json.load(open(params, "rt"))
        user_params["encoder"] = params["encoder"]
        user_params["decoder"] = params["decoder"]
        params = user_params
    return model_type, params, checkpoint


def check():
    checkpoint = torch.load("models_state_dict/EncoderDecoder_model/EncoderDecoder 2020-June-29 20-36-54.pt")
    params = checkpoint["params"]
    params["data"]["train_source"] = "data/train.src"
    params["data"]["train_target"] = "data/train.trg"
    params["data"]["dev_source"] = "data/dev.src"
    params["data"]["dev_target"] = "data/dev.trg"
    params["data"]["test_source"] = "data/test.src"
    params["data"]["test_target"] = "data/test.trg"
    model_type = checkpoint["model_type"]
    return model_type, params, checkpoint


if __name__ == '__main__':
    validate(*check())
    # validate(*parse_input())
