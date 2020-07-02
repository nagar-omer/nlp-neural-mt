import sys
sys.path.insert(0, "src_code")
import optparse
import os
from activator import ATTENTION, ENCODER_DECODER, MTActivator


def fit_and_dump(model_type, params):
    activator = MTActivator(model_type, params)
    activator.fit()


def parse_input():
    optparser = optparse.OptionParser()
    optparser.add_option("--attention", dest="attention",
                         action='store_true', help="Use an attention based model")
    optparser.add_option("--encoder_decoder", dest="encoder_decoder",
                         action='store_true', help="Use a basic Encoder-Decoder based model")
    optparser.add_option("-p", "--params", dest="params", default="mt_params.json", help="path to parameters file")
    (opts, args) = optparser.parse_args()

    if opts.attention and opts.encoder_decoder:
        print("Error: Only a single model-type can pe picked, "
              "drop one of the args:\n\t1. --attention\n\t2. --encoder_decoder")
        exit(1)
    elif not os.path.exists(opts.params):
        print("Error: file " + opts.params + " does not exist")
        exit(1)
    model_type = ENCODER_DECODER if opts.encoder_decoder else ATTENTION
    return model_type, opts.params


if __name__ == '__main__':
    fit_and_dump(*parse_input())
