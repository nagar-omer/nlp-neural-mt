import random
from sys import stdout
import json

import torch
from bokeh.io import output_file, save
from bokeh.plotting import figure, show
from torch.nn.modules.loss import CrossEntropyLoss, NLLLoss
from torch.utils.data import DataLoader
from dataset import MTDataset, PAD, EOS
from encoder_decoder import MTEncoderDecoder
import sacrebleu

TRAIN_JOB = "TRAIN"
DEV_JOB = "DEV"
VALIDATE_JOB = "VALIDATE"
LOSS_PLOT = "loss"
BLEU_PLOT = "accuracy"


class CostumeCrossEntropy:
    def __init__(self, ignore_index, label_counts):
        self._ignore = ignore_index
        # self._weights = torch.Tensor([1/count for idx, count in sorted(label_counts.items(), key=lambda x: x[0])])
        self._criterion = CrossEntropyLoss(ignore_index=ignore_index) # , weight=self._weights)

    def __call__(self, output, target):
        return self._criterion(output.view(-1, output[0].shape[-1]), target.view(-1))
        # l = 0
        # for i, out in enumerate(output):
        #     l += self._criterion(out, target[:, i].view(-1))
        # return l


class MTActivator:
    def __init__(self, model_type, params):
        self._train_params, self._data_params, self._opt_params = self._load_params(params)
        self._load_data(model_type)
        self._load_model(model_type, params)
        self._init_loss_and_acc_vec()
        self._init_print_att()
        self._set_optimizer()

    def _set_optimizer(self):
        self._criterion = CostumeCrossEntropy(self._train_dataset.mapping[0][PAD], self._train_dataset._label_counts)

    def _load_params(self, params_path):
        """
        load parameters from json
        """
        params = json.load(open(params_path, "rt"))
        train_params = params["train"]
        data_params = params["data"]
        opt_params = params["optimizer"]
        return train_params, data_params, opt_params

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

    def _load_data(self, model_type):
        """
        build train and dev data-loaders and obtain essential parameters
        """
        # set train loader
        train_ds = MTDataset("data/train.src", "data/train.trg")
        dev_ds = MTDataset("data/dev.src", "data/dev.trg", fixed_mapping=train_ds.mapping)

        # encoder = MTEncoder("encoder_decoder_params.json", train.len_source_vocab, train.source_pad_index)
        # decoder = MTDecoder("encoder_decoder_params.json", train.len_target_vocab, train.start_index, encoder.out_dim)
        if model_type == "EncoderDecoder":
            self._model_params = (train_ds.target_start_index, train_ds.source_pad_index,
                                  train_ds.len_source_vocab, train_ds.len_target_vocab)

        self._train_dataset = train_ds
        self._idx_to_target = [l for l, idx in sorted(self._train_dataset.mapping[0].items(), key=lambda x: x[1])]
        self._idx_to_source = [l for l, idx in sorted(self._train_dataset.mapping[1].items(), key=lambda x: x[1])]
        self._train_loader = self._get_data_loader(train_ds)
        self._dev_loader = self._get_data_loader(dev_ds)

    def _load_model(self, model_type, params):
        if model_type == "EncoderDecoder":
            self._model = MTEncoderDecoder(params, *self._model_params)
            self._optimizer = self._model.set_optimizer(self._opt_params)
        else:
            print("Error: there is mo model named", model_type,
                  "\navailable_model_types:\n\t1.EncoderDecoder\n\t2.Attention")
            exit(1)

    def _init_loss_and_acc_vec(self):
        """
        init loss and BLEU vectors (as function of epochs)
        """
        self._loss_vec_train, self._bleu_vec_train = [], []
        self._loss_vec_dev, self._bleu_vec_dev = [], []

    def _init_print_att(self):
        """
        init variables that holds the last update for loss and BLEU
        """
        self._print_train_loss, self._print_train_bleu = [], []
        self._print_dev_loss, self._print_dev_bleu = [], []

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
        print("")

    def plot_line(self, job=LOSS_PLOT):
        """
        plot loss / bleu as function of epochs
        """
        p = figure(plot_width=600, plot_height=250, title="Rand_FST - Dataset " + job,
                   x_axis_label="epochs", y_axis_label=job)
        color1, color2 = ("orange", "red") if job == LOSS_PLOT else ("green", "blue")

        y_axis_train = self._loss_vec_train if job == LOSS_PLOT else self._bleu_vec_train
        y_axis_dev = self._loss_vec_dev if job == LOSS_PLOT else self._bleu_vec_dev

        x_axis = list(range(len(y_axis_dev)))
        p.line(x_axis, y_axis_train, line_color=color1, legend="train")
        p.line(x_axis, y_axis_dev, line_color=color2, legend="dev")
        output_file(job + "_fig.html")
        save(p)
        show(p)

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
            print("Epoch:", epoch_num)
            self._model.train()
            # calc number of iteration in current epoch
            for batch_index, (source, target) in enumerate(self._train_loader):
                self._optimizer.zero_grad()                        # zero gradients
                output = self._model(source, target)               # calc output of current model on the current batch
                loss = self._criterion(output, target)             # calculate loss
                loss.backward()                                    # back propagation
                self._optimizer.step()                             # update weights

                self._print_progress(batch_index, len_data, job=TRAIN_JOB)  # print progress

            # validate and print progress
            self._validate(self._train_loader, job=TRAIN_JOB)
            self._validate(self._dev_loader, job=DEV_JOB)
            self._print_info(jobs=[TRAIN_JOB, DEV_JOB])

        if show_plot:
            self._plot_acc_dev()

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

            output = self._model(source)
            # calculate total loss
            loss_count += self._criterion(output, target)  # calculate loss
            out_sent.append(output.argmax(dim=2))
            target_sent.append(target)

        # update loss accuracy
        loss = float(loss_count / len(data_loader))
        self._update_loss(loss, job=job)
        self._update_bleu(out_sent, target_sent, job=job)
        return loss


def sanity_test():
    activator = MTActivator("EncoderDecoder", "encoder_decoder_params.json")
    activator.fit()
    import pickle
    pickle.dump(activator.model, open("model.pkl", "wb"))


if __name__ == '__main__':
    sanity_test()
