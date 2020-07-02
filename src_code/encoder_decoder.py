import torch
from torch.nn import LSTM, Embedding, Linear, Module, Dropout, LeakyReLU, GRU, LogSoftmax
import json
from torch.optim import SGD, Adam
OPT = {"Adam": Adam, "SGD": SGD}


class MTEncoder(Module):
    """
    this class implements Sentence Encoder
    It's full operation consists of:
        1. get sentences (words are indices)
        2. obtain Embeddings
        3. pass through on directional LSTM
        4. return last layer
    """
    def __init__(self, params, vocab_dim, pad_index):
        super().__init__()
        self._params = params["encoder"] if type(params) is dict else json.load(open(params, "rt"))["encoder"]
        self._embeddings = Embedding(vocab_dim, self._params["embed_dim"], padding_idx=pad_index)   # word embed layer
        self._gru = GRU(self._params["embed_dim"], self._params["hidden_dim"],               # LSTM
                        self._params["num_layers"], dropout=self._params["dropout"], batch_first=True,
                        bidirectional=True)
        self._dropout = Dropout(p=self._params["dropout"])

    def forward(self, words_embed):
        """
        :param words_embed: word indices batch
        :return: sentence representation vector
        """
        x = self._dropout(self._embeddings(words_embed))
        output_seq, hidden_seq = self._gru(x)
        return output_seq


class MTDecoder(Module):
    """
    this class implements Sentence Decoder - it generates a new sentence given a context vector
    It's full operation consists of:
        1. get context vector
        2. init hidden vector
        3. loop
            3.1 predict next word using the hidden and context vectors
            3.2 update hidden vector
            3.3 save prediction and continue
    """
    def __init__(self, params, vocab_dim, start_index, context_dim):
        super().__init__()
        self._params = params["decoder"] if type(params) is dict else json.load(open(params, "rt"))["decoder"]

        self._embeddings = Embedding(vocab_dim, self._params["embed_dim"])                          # word embed layer
        self._lstm = LSTM(self._params["embed_dim"] + context_dim,                                  # LSTM layers
                          self._params["hidden_dim"], self._params["num_layers"],
                          dropout=self._params["dropout"], batch_first=True)
        self._out_dim = vocab_dim
        self._linear = Linear(self._params["hidden_dim"], vocab_dim)                           # Linear layer
        self._start = start_index
        self._dropout = Dropout(p=self._params["dropout"])
        self._softmax = LogSoftmax(dim=2)

    def forward(self, context, max_len, target=None, criterion=None):
        """
        :param context: context vector batch
        :param max_len: the function will create sequences of length max_len
        :return: generated sequences
        """
        context = self._dropout(context)
        x = torch.ones((context.shape[0], 1)).long() * self._start                     # init start vector
        hidden = None                                                                  # init hidden vector, and out
        out = torch.zeros((context.shape[0], max_len, self._out_dim))
        loss = 0
        for i in range(max_len):
            ex = self._embeddings(x)                                                   # obtain embeddings of last word
            # concat last word and context vectors and pass through LSTM
            lstm_input = torch.cat((ex, context.unsqueeze(dim=1)), dim=2)
            output_seq, hidden = self._lstm(lstm_input, hidden)
            word_distribution = self._softmax(self._linear(output_seq))         # predict next word distribution

            loss += criterion(word_distribution, target[:, i].unsqueeze(dim=0)) if criterion is not None else 0

            out[:, i, :] = word_distribution.squeeze(dim=1)
            x = word_distribution.argmax(dim=2).detach() if target is None else target[:, i].unsqueeze(dim=1)
        return out, loss


class MTEncoderDecoder(Module):
    def __init__(self, params, start_index, sourcee_pad, source_vocab_dim, target_vocab_dim):
        """
        this class implements a Full Machine Translation Model based on Encoder-Decoder architecture
        It's full operation consists of:
            1. Encode - obtain context vector for source sentence
            2. Decode = generate new sentence by the context vector
        """
        # super(MTEncoderDecoder, self).__init__()
        super().__init__()
        self._params = params if type(params) is dict else json.load(open(params, "rt"))
        self._encoder = MTEncoder(params, source_vocab_dim, sourcee_pad)
        self._decoder = MTDecoder(params, target_vocab_dim, start_index, 4 * self._params["encoder"]["hidden_dim"])
        # self.set_optimizer(self._params["optimizer"])

    def set_optimizer(self, opt_params):
        return OPT[opt_params["type"]](self.parameters(), **opt_params["kwargs"])

    def forward(self, word_embed, target=None, criterion=None):
        """
        :param word_embed: bach of source sentences
        :return: new generated sentence (word distribution)
        """

        context = self._encoder(word_embed)
        context = torch.cat([context[:, 0, :], context[:, -1, :]], dim=1)  # first and last hidden
        max_len = 2 * word_embed.shape[1] if target is None else word_embed.shape[1] - 1
        generated, loss = self._decoder(context, max_len=max_len, target=target, criterion=criterion)
        return generated, loss


def sanity_test():
    from dataset import MTDataset
    from torch.utils.data import DataLoader
    train = MTDataset("../data/train.src", "../data/train.trg")
    model = MTEncoderDecoder("sample_params.json", train.target_start_index, train.source_pad_index,
                             train.len_source_vocab, train.len_target_vocab)

    dl = DataLoader(train,
                    num_workers=4,
                    shuffle=True,
                    batch_size=64,
                    collate_fn=train.collate_fn)

    for src_, dst_ in dl:
        print(model(src_, target=dst_))


if __name__ == '__main__':
    sanity_test()
