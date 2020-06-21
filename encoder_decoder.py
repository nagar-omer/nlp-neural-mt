import torch
from torch.autograd import Variable
from torch.nn import LSTM, Embedding, Linear, Module
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
        self._params = json.load(open(params, "rt"))["encoder"]                                     # load hyper-params
        self._embeddings = Embedding(vocab_dim, self._params["embed_dim"], padding_idx=pad_index)   # word embed layer
        self._lstm = LSTM(self._params["embed_dim"], self._params["lstm_hidden_dim"],               # LSTM
                          self._params["num_lstm_layers"], dropout=self._params["dropout"], batch_first=True)

    def forward(self, words_embed):
        """
        :param words_embed: word indices batch
        :return: sentence representation vector
        """
        x = self._embeddings(words_embed)
        output_seq, hidden_seq = self._lstm(x)
        return output_seq[:, -1, :].unsqueeze(dim=1)


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
        self._params = json.load(open(params, "rt"))["decoder"]
        self._start = start_index

        self._embeddings = Embedding(vocab_dim, self._params["embed_dim"])                          # word embed layer
        self._lstm = LSTM(self._params["embed_dim"] + context_dim,                                  # LSTM layers
                          self._params["lstm_hidden_dim"], self._params["num_lstm_layers"],
                          dropout=self._params["dropout"], batch_first=True)
        self._out_dim = vocab_dim
        self._linear = Linear(self._params["lstm_hidden_dim"], vocab_dim)                           # Linear layer

    def forward(self, context, max_len, target=None):
        """
        :param context: context vector batch
        :param max_len: the function will create sequences of length max_len
        :return: generated sequences
        """
        x = torch.ones((context.shape[0], 1)).long() * self._start                     # init start vector
        hidden = None                                                                  # init hidden vector, and out
        out = torch.zeros((context.shape[0], max_len, self._out_dim))
        for i in range(max_len):
            ex = self._embeddings(x)                                                   # obtain embeddings of last word
            # concat last word and context vectors and pass through LSTM
            output_seq, hidden = self._lstm(torch.cat((ex, context), dim=2), hidden)
            word_distribution = torch.softmax(self._linear(output_seq), dim=2)         # predict next word distribution

            out[:, i, :] = word_distribution.squeeze(dim=1)
            x = word_distribution.argmax(dim=2).detach() if target is None else target[:, i].unsqueeze(dim=1)
        return out


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
        self._params = json.load(open(params, "rt"))
        self._encoder = MTEncoder(params, source_vocab_dim, sourcee_pad)
        self._decoder = MTDecoder(params, target_vocab_dim, start_index, self._params["encoder"]["lstm_hidden_dim"])
        # self.set_optimizer(self._params["optimizer"])

    def set_optimizer(self, opt_params):
        return OPT[opt_params["type"]](self.parameters(), **opt_params["kwargs"])

    def forward(self, word_embed, target=None):
        """
        :param word_embed: bach of source sentences
        :return: new generated sentence (word distribution)
        """

        context = self._encoder(word_embed)
        generated = self._decoder(context, max_len=word_embed.shape[1] - 1, target=target)
        return generated


def sanity_test():
    from dataset import MTDataset
    from torch.utils.data import DataLoader
    train = MTDataset("data/train.src", "data/train.trg")
    model = MTEncoderDecoder("encoder_decoder_params.json", train.target_start_index, train.source_pad_index,
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
