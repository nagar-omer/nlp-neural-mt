To run the Encoder decoder/Attention file follow the following steps:
    1. Params File (optional)
       edit/create params file.
       the default params file is called "mt_params.json".
       the file contains:

        - model hyper-parameters
        - train hyper-parameters
        - path to data files

    2. Fit
       command:
       python mt_train.py

       arguments:
            --encoder_decoder   Use basic Encoder-Decoder architecture
            --attention         Use Attention architecture
            --params:           path tp params file, if not specified then "mt_params.json" will be loaded.

       output:
            - model_state_dict: folder containing the best model parameters (with respect to BLEU score)
            - figs: folder containing loss & BLEU scores as function of epochs

    3. Evaluate
       command:
       python mt_train.py

       arguments:
            --checkpoint:       path to model parameters file. if not specified the last trained model will be loaded.
            --params:           a path to a new params file.
                                (encoder-decoder section will be ignored since its already defined by the uploaded model)