import datetime
import os
import random
from copy import deepcopy
from sys import stdout
import json
import nni
import torch
from bokeh.io import output_file, save, show
from bokeh.plotting import figure
from torch.nn import LogSoftmax
from torch.nn.modules.loss import CrossEntropyLoss, NLLLoss
from torch.utils.data import DataLoader
from attention import MTAttention
from dataset import MTDataset, PAD, EOS, SOS
from encoder_decoder import MTEncoderDecoder
import sacrebleu
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
logging.getLogger().setLevel(logging.INFO)

TEST_JOB = "TEST"
TRAIN_JOB = "TRAIN"
DEV_JOB = "DEV"
VALIDATE_JOB = "VALIDATE"
LOSS_PLOT = "loss"
BLEU_PLOT = "BLEU"
ATTENTION = "Attention"
ENCODER_DECODER = "EncoderDecoder"


class CostumeCrossEntropy:
    def __init__(self, ignore_index, label_counts):
        self._ignore = ignore_index
        self._weights = torch.Tensor([1/count for idx, count in sorted(label_counts.items(), key=lambda x: x[0])])
        self._criterion = NLLLoss(ignore_index=ignore_index, reduction="sum") #, weight=self._weights)
        # self._log = LogSoftmax() #, weight=self._weights)

    def __call__(self, output, target):
        return self._criterion(output[:, :target.shape[1], :].contiguous().view(-1, output[0].shape[-1]),
                               target.view(-1)) #+ loss_attend


class MTActivator:
    def __init__(self, model_type, params, checkpoint=None, with_nni=False):
        self._checkpoint = checkpoint
        self._model_type = model_type
        self._params, self._train_params, self._data_params, self._opt_params = self._load_params(params)
        self._load_data()
        self._load_model(model_type, params, checkpoint)
        self._init_loss_and_acc_vec()
        self._init_print_att()
        self._set_optimizer()
        self._best_model, self._best_bleu, self._best_loss, self._best_epoch = None, 0, 100, 0
        self._with_nni = with_nni
        self._get_nni_id(with_nni)
        self._creation_time = datetime.datetime.now().strftime("%Y-%B-%d %H-%M-%S")
        self._init_recorded_example()
        logging.info("Activator Initiated")

    def _init_recorded_example(self):
        # only for Attention model
        if self._model_type == ENCODER_DECODER:
            return
        self._heatmap_pairs = []
        for instance in open(os.path.join(__file__.replace("\\", "/").rsplit("/")[0], "heatmap.inst"), "rt", encoding="utf-8"):
            s, t = instance.split("|")
            self._heatmap_pairs.append(([SOS] + s.split() + [EOS], t.split() + [EOS]))
        self._heatmap_dir = os.path.join("figs", "Attention_heatmap", self._creation_time)
        os.makedirs(self._heatmap_dir)

    def _record_examples(self):
        if self._model_type == ENCODER_DECODER:
            return
        for s, t in self._heatmap_pairs:
            self._record_example(s, t)

    def _record_example(self, source, target):
        # only for Attention model
        if self._model_type == ENCODER_DECODER:
            return
        out, attention, loss = self._model(torch.Tensor([[self._source_to_idx[token] for token in source]]).long())
        out, attention = out[0], attention[0]
        out_seq = []
        for token in out:
            token = self._idx_to_target[token.argmax().item()]
            if token == EOS:
                break
            out_seq.append(token)
        attention = attention[: len(out_seq) + 1]
        ax = sns.heatmap(attention.detach().numpy(), cmap="Greens")
        ax.set_xticks([x + 0.5 for x in np.arange(len(source))])
        ax.set_yticks([x + 0.5 for x in np.arange(len(out_seq))])
        ax.set_xticklabels(["|".join([s, t]) for s, t in zip(source, [""] + target[:-1] + [""])])
        ax.set_yticklabels(out_seq)
        plt.yticks(rotation=0)
        ax.set_title("Attention Weights, Epoch=" + str(len(self._bleu_vec_dev)))
        plt.xlabel("Source")
        plt.ylabel("Target")
        dir_name = "".join(e for e in target[:-1] if e.isalnum())
        os.makedirs(os.path.join(self._heatmap_dir, dir_name), exist_ok=True)
        plt.savefig(os.path.join(self._heatmap_dir, dir_name,  "epoch_" + str(len(self._bleu_vec_dev)) + ".jpg"))
        # plt.show()
        plt.clf()

    def _get_nni_id(self, with_nni):
        if not with_nni:
            self._nni_id = ""
            return
        exp = nni.get_experiment_id()
        trail = nni.get_trial_id()
        self._nni_id = exp + "-" + trail + " "

    def _set_optimizer(self):
        self._criterion = CostumeCrossEntropy(self._train_dataset.mapping[0][PAD], self._train_dataset._label_counts)

    def _load_params(self, params):
        """
        load parameters from json
        """
        params = params if type(params) is dict else json.load(open(params, "rt"))
        train_params = params["train"]
        data_params = params["data"]
        opt_params = params["optimizer"]
        return params, train_params, data_params, opt_params

    def _get_data_loader(self, dataset):
        """
        build a data loader for a given dataset
        """
        dl = DataLoader(dataset,                                                     # build data loader
                        num_workers=4,
                        shuffle=True,
                        batch_size=self._train_params["batch_size"],
                        collate_fn=dataset.collate_fn)
        return dl

    def _load_data(self):
        """
        build train and dev data-loaders and obtain essential parameters
        """
        # set train loader
        train_ds = MTDataset(self._data_params["train_source"], self._data_params["train_target"])
        dev_ds = MTDataset(self._data_params["dev_source"], self._data_params["dev_target"],
                           fixed_mapping=train_ds.mapping)

        self._model_params = (train_ds.target_start_index, train_ds.source_pad_index,
                              train_ds.len_source_vocab, train_ds.len_target_vocab)
        self._train_dataset = train_ds
        self._target_to_idx, self._source_to_idx = self._train_dataset.mapping[0], self._train_dataset.mapping[1]
        self._idx_to_target = [l for l, idx in sorted(self._train_dataset.mapping[0].items(), key=lambda x: x[1])]
        self._idx_to_source = [l for l, idx in sorted(self._train_dataset.mapping[1].items(), key=lambda x: x[1])]
        self._train_loader = self._get_data_loader(train_ds)
        self._dev_loader = self._get_data_loader(dev_ds)
        self._mapping = train_ds.mapping

        logging.info("train-set source: " + self._data_params["train_source"] +
                     "\ttrain-set target: " + self._data_params["train_target"])
        logging.info("dev-set source: " + self._data_params["dev_source"] +
                     "\tdev-set target: " + self._data_params["dev_target"])

    def _load_model(self, model_type, params, checkpoint):
        if model_type == ENCODER_DECODER:
            logging.info("Initiating Encoder-Decoder Model")
            self._model = MTEncoderDecoder(params, *self._model_params)
        elif model_type == ATTENTION:
            logging.info("Initiating Attention Model")
            self._model = MTAttention(params, *self._model_params)
        else:
            print("Error: there is mo model named", model_type,
                  "\navailable_model_types:\n\t1.EncoderDecoder\n\t2.Attention")
            exit(1)

        # load parameters
        if checkpoint is not None:
            logging.info("loading pre-trained params")
            self._model.load_state_dict(checkpoint['model_state_dict'])

        self._optimizer = self._model.set_optimizer(self._opt_params)

    def _dump_best_model(self):

        pkl_dir = os.path.join("nni_dump" if self._with_nni else "models_state_dict", self._model_type + "_model")
        os.makedirs(pkl_dir, exist_ok=True)
        model_name = self._model_type + " " + self._nni_id + self._creation_time
        torch.save({
            'checkpoint': self._checkpoint,
            'model_type': self._model_type,
            'epoch': self._best_epoch,
            'model_state_dict': self._best_model,
            'BLEU': self._best_bleu,
            'loss': self._best_loss,
            'params': self._params,
        }, os.path.join(pkl_dir, model_name + ".pt"))
        json.dump(self._params, open(os.path.join(pkl_dir, model_name + " config.json"), "wt"))

    def _init_loss_and_acc_vec(self):
        """
        init loss and BLEU vectors (as function of epochs)
        """
        self._loss_vec_train, self._bleu_vec_train = [], []
        self._loss_vec_dev, self._bleu_vec_dev = [], []
        self._loss_vec_test, self._bleu_vec_test = [], []

    def _init_print_att(self):
        """
        init variables that holds the last update for loss and BLEU
        """
        self._print_train_loss, self._print_train_bleu = [], []
        self._print_dev_loss, self._print_dev_bleu = [], []
        self._print_test_loss, self._print_test_bleu = [], []

    def _update_best_model(self, epoch):
        if (self._bleu_vec_dev and self._bleu_vec_dev[-1] > self._best_bleu) or self._best_model is None:
            logging.info("Best model update, BLEU: " + str(round(self._bleu_vec_dev[-1], 4)) + "  epoch: " + str(epoch))
            self._best_model = deepcopy(self._model.state_dict())
            self._best_bleu = self._bleu_vec_dev[-1]
            self._best_loss = self._loss_vec_dev[-1]
            self._best_epoch = epoch
            self._dump_best_model()

    def _update_loss(self, loss, job=TRAIN_JOB):
        """
        update loss after validation
        """
        if job == TRAIN_JOB:
            self._loss_vec_train.append(loss)
            self._print_train_loss = loss
        elif job == DEV_JOB:
            self._loss_vec_dev.append(loss)
            self._print_dev_loss = loss
        elif job == TEST_JOB:
            self._loss_vec_test.append(loss)
            self._print_test_loss = loss

    def _extract_sentences(self, sent_batches):
        """
        break batch of indices to list of sentences
        """
        sents = []
        for batch in sent_batches:                  # loop batch
            for sent_idx in batch:                  # loop sentences
                sent = []
                for letter in sent_idx:             # loop words
                    letter2 = self._idx_to_target[letter]
                    if letter2 == PAD:
                        continue
                    if letter2 == EOS:
                        # sent.append(letter2)
                        break
                    sent.append(letter2)
                sents.append(" ".join(sent))
        return sents

    # update accuracy after validating
    def _update_bleu(self, out_sentences, target_sentences, job=TRAIN_JOB):
        """
        update loss after validation
        """
        refs = [self._extract_sentences(target_sentences)]
        out = self._extract_sentences(out_sentences)
        for i in range(5):
            sample = random.randint(0, len(refs[0]) - 1)
            print("sample", '{:{width}d}'.format(sample, width=6),
                  ":\ttarget:", '{:<40}'.format(refs[0][sample]),
                  "\ttranslation", out[sample])
        bleu = sacrebleu.corpus_bleu(out, refs, force=True).score
        if job == TRAIN_JOB:
            self._bleu_vec_train.append(bleu)
            self._print_train_bleu = bleu
        elif job == DEV_JOB:
            self._bleu_vec_dev.append(bleu)
            self._print_dev_bleu = bleu
        elif job == TEST_JOB:
            self._bleu_vec_test.append(bleu)
            self._print_test_bleu = bleu

    def _print_progress(self, batch_index, len_data, job=""):
        """
        print progress of a single epoch as a percentage
        :param job: train or validation
        """
        prog = int(100 * (batch_index + 1) / len_data)
        stdout.write("\r\r\r\r\r\r\r\r" + job + " %d" % prog + "%")
        print("", end="\n" if prog == 100 else "")
        stdout.flush()

    # print last loss and accuracy
    def _print_info(self, jobs=()):
        """
        print validation info
        """
        if TRAIN_JOB in jobs:
            print("Loss_Train: " + '{:{width}.{prec}f}'.format(self._print_train_loss, width=6, prec=8) + (" \\/"
                  if len(self._loss_vec_train) == 1 or self._print_train_loss < self._loss_vec_train[-2] else " /\\") +
                  " || BLEU_Train: " + '{:{width}.{prec}f}'.format(self._print_train_bleu, width=6, prec=4),
                  end=" || ")
        if DEV_JOB in jobs:
            print("Loss_Dev: " + '{:{width}.{prec}f}'.format(self._print_dev_loss, width=6, prec=8) +
                  " || BLEU_Dev: " + '{:{width}.{prec}f}'.format(self._print_dev_bleu, width=6, prec=4),
                  end=" || ")
        if TEST_JOB in jobs:
            print("Loss_Test: " + '{:{width}.{prec}f}'.format(self._print_test_loss, width=6, prec=8) +
                  " || BLEU_Test: " + '{:{width}.{prec}f}'.format(self._print_test_bleu, width=6, prec=4),
                  end=" || ")
        print("")

    def plot_line(self, job=LOSS_PLOT):
        """
        plot loss / bleu as function of epochs
        """
        p = figure(plot_width=600, plot_height=250, title="MT-" + self._model_type + ": " + job,
                   x_axis_label="epochs", y_axis_label=job)
        color1, color2 = ("orange", "red") if job == LOSS_PLOT else ("green", "blue")

        y_axis_train = self._loss_vec_train if job == LOSS_PLOT else self._bleu_vec_train
        y_axis_dev = self._loss_vec_dev if job == LOSS_PLOT else self._bleu_vec_dev

        x_axis = list(range(len(y_axis_dev)))
        p.line(x_axis, y_axis_train, line_color=color1, legend_label="train")
        p.line(x_axis, y_axis_dev, line_color=color2, legend_label="dev")

        pkl_dir = os.path.join("nni_figs" if self._with_nni else"figs", self._model_type + "_model")
        os.makedirs(pkl_dir, exist_ok=True)
        save_path = os.path.join(pkl_dir, self._model_type + " " + job + " " + self._nni_id + self._creation_time + ".html")
        output_file(save_path)
        save(p, filename=save_path)
        # plt.show(p)

    def _plot_acc_dev(self):
        self.plot_line(LOSS_PLOT)
        self.plot_line(BLEU_PLOT)

    @property
    def model(self):
        return self._model

    @property
    def loss_train_vec(self):
        return self._loss_vec_train

    @property
    def bleu_train_vec(self):
        return self._bleu_vec_train

    @property
    def loss_dev_vec(self):
        return self._loss_vec_dev

    @property
    def bleu_dev_vec(self):
        return self._bleu_vec_dev

    def fit(self, show_plot=True):
        self._init_loss_and_acc_vec()
        # calc number of iteration in current epoch
        len_data = len(self._train_loader)
        for epoch_num in range(self._train_params["epochs"]):
            print("Epoch:", epoch_num + 1, "/", self._train_params["epochs"])
            self._model.train()
            # calc number of iteration in current epoch
            for batch_index, (source, target) in enumerate(self._train_loader):
                self._optimizer.zero_grad()                        # zero gradients
                output = self._model(source, target, self._criterion)             # calc output of current model on the current batch
                output, attention, loss = output if self._model_type == ATTENTION else (output[0], None, output[1])
                # loss = self._criterion(output, target)       # calculate loss
                loss.backward()                                    # back propagation
                self._optimizer.step()                             # update weights

                self._print_progress(batch_index, len_data, job=TRAIN_JOB)  # print progress

            # validate and print progress
            self._validate(self._train_loader, job=TRAIN_JOB)
            self._validate(self._dev_loader, job=DEV_JOB)
            self._update_best_model(epoch_num)
            self._record_examples()
            self._print_info(jobs=[TRAIN_JOB, DEV_JOB])
        # /----------------------  FOR NNI  -------------------------
        final = max(self._bleu_vec_dev)
        nni.report_final_result(final)
        # -----------------------  FOR NNI  -------------------------/

        if show_plot:
            self._plot_acc_dev()
        self._dump_best_model()

    # validation function only the model and the data are important for input, the others are just for print
    def _validate(self, data_loader, job=""):
        # for calculating total loss and accuracy
        loss_count = 0
        out_sent, target_sent = [], []

        self._model.eval()
        # calc number of iteration in current epoch
        len_data = len(data_loader)
        for batch_index, (source, target) in enumerate(data_loader):
            self._print_progress(batch_index, len_data, job=VALIDATE_JOB)             # print progress

            output = self._model(source, target, self._criterion)  # calc output of current model on the current batch
            output, attention, loss = output if self._model_type == ATTENTION else (output[0], None, output[1])

            # calculate total loss
            loss_count += self._criterion(output, target)                             # calculate loss
            out_sent.append(output.argmax(dim=2))
            target_sent.append(target)

        # update loss accuracy
        loss = float(loss_count / len(data_loader))
        self._update_loss(loss, job=job)
        self._update_bleu(out_sent, target_sent, job=job)
        return loss

    def validate_test(self):
        test_loader = self._get_data_loader(MTDataset(self._data_params["test_source"],
                                                      self._data_params["test_target"], fixed_mapping=self._mapping))
        # test_loader = self._get_data_loader(MTDataset(self._data_params["dev_source"],
        #                                               self._data_params["dev_target"], fixed_mapping=self._mapping))
        self._validate(test_loader, job=TEST_JOB)
        self._print_info(jobs=[TEST_JOB])


def sanity_test():
    activator = MTActivator(ENCODER_DECODER, "sample_params.json")
    # activator = MTActivator(ATTENTION, "sample_params.json")
    activator.fit()


if __name__ == '__main__':
    sanity_test()
