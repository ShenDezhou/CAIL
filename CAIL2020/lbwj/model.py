"""BERT and RNN model for sentence pair classification.

Author: Yixu GAO (yxgao19@fudan.edu.cn)

Used for SMP-CAIL2020-Argmine.
"""
import torch

from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from transformers.modeling_bert import BertModel
# from pytorch_pretrained_bert import BertModel

class BertForClassification(nn.Module):
    """BERT with simple linear model."""
    def __init__(self, config):
        """Initialize the model with config dict.

        Args:
            config: python dict must contains the attributes below:
                config.bert_model_path: pretrained model path or model type
                    e.g. 'bert-base-chinese'
                config.hidden_size: The same as BERT model, usually 768
                config.num_classes: int, e.g. 2
                config.dropout: float between 0 and 1
        """
        super().__init__()
        self.bert = BertModel.from_pretrained(config.bert_model_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.linear = nn.Linear(config.hidden_size, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)
        self.num_classes = config.num_classes

    def forward(self, input_ids, attention_mask, token_type_ids):
        """Forward inputs and get logits.

        Args:
            input_ids: (batch_size, max_seq_len)
            attention_mask: (batch_size, max_seq_len)
            token_type_ids: (batch_size, max_seq_len)

        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size = input_ids.shape[0]
        bert_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            # encoder_hidden_states=False
        )
        # bert_output[0]: (batch_size, sequence_length, hidden_size)
        # bert_output[1]: (batch_size, hidden_size)
        pooled_output = bert_output[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output).view(batch_size, self.num_classes)
        logits = nn.functional.softmax(logits, dim=-1)
        # logits: (batch_size, num_classes)
        return logits


class RnnForSentencePairClassification(nn.Module):
    """Unidirectional GRU model for sentences pair classification.
    2 sentences use the same encoder and concat to a linear model.
    """
    def __init__(self, config):
        """Initialize the model with config dict.

        Args:
            config: python dict must contains the attributes below:
                config.vocab_size: vocab size
                config.hidden_size: RNN hidden size and embedding dim
                config.num_classes: int, e.g. 2
                config.dropout: float between 0 and 1
        """
        super().__init__()
        self.embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)
        self.rnn = nn.GRU(
            config.hidden_size, hidden_size=config.hidden_size,
            bidirectional=False, batch_first=True)
        self.linear = nn.Linear(config.hidden_size * 2, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)
        self.num_classes = config.num_classes

    def forward(self, s1_ids, s2_ids, s1_lengths, s2_lengths):
        """Forward inputs and get logits.

        Args:
            s1_ids: (batch_size, max_seq_len)
            s2_ids: (batch_size, max_seq_len)
            s1_lengths: (batch_size)
            s2_lengths: (batch_size)

        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size = s1_ids.shape[0]
        # ids: (batch_size, max_seq_len)
        s1_embed = self.embedding(s1_ids)
        s2_embed = self.embedding(s2_ids)
        # embed: (batch_size, max_seq_len, hidden_size)
        s1_packed: PackedSequence = pack_padded_sequence(
            s1_embed, s1_lengths, batch_first=True, enforce_sorted=False)
        s2_packed: PackedSequence = pack_padded_sequence(
            s2_embed, s2_lengths, batch_first=True, enforce_sorted=False)
        # packed: (sum(lengths), hidden_size)
        self.rnn.flatten_parameters()
        _, s1_hidden = self.rnn(s1_packed)
        _, s2_hidden = self.rnn(s2_packed)
        s1_hidden = s1_hidden.view(batch_size, -1)
        s2_hidden = s2_hidden.view(batch_size, -1)
        hidden = torch.cat([s1_hidden, s2_hidden], dim=-1)
        hidden = self.linear(hidden).view(-1, self.num_classes)
        hidden = self.dropout(hidden)
        logits = nn.functional.softmax(hidden, dim=-1)
        # logits: (batch_size, num_classes)
        return logits

class bilstm_attn(torch.nn.Module):
    def __init__(self, config):#batch_size, output_size, hidden_size, vocab_size, embed_dim, bidirectional, dropout, use_cuda, attention_size, sequence_length):
        super(bilstm_attn, self).__init__()

        # self.batch_size = config
        self.output_size = config.num_classes#output_size
        self.embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)

        # self.vocab_size = vocab_size
        # self.embed_dim = embed_dim
        # self.bidirectional = bidirectional
        # self.dropout = dropout
        # self.use_cuda = use_cuda
        # self.sequence_length = sequence_length
        self.lookup_table = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.lookup_table.weight.data.uniform_(-1., 1.)
        self.use_cuda = True
        self.bidirectional = True
        self.layer_size = 1
        self.sequence_length = config.max_seq_len
        # self.lstm = nn.LSTM(config.vocab_size,
        #                     config.hidden_size,
        #                     num_layers=self.layer_size,
        #                     dropout=config.dropout,
        #                     bidirectional=self.bidirectional)
        self.rnn = nn.LSTM(
            config.hidden_size, hidden_size=config.hidden_size,
            bidirectional=False, batch_first=True)
        # self.rnn2 = nn.GRU(
        #     config.hidden_size, hidden_size=config.hidden_size,
        #     bidirectional=False, batch_first=True)

        self.hidden_size = config.hidden_size

        if self.bidirectional:
            self.layer_size = self.layer_size * 2
        else:
            self.layer_size = self.layer_size
        self.attention_size = config.attention_size

        # self.attention = nn.Sequential(
        #     # 对应于论文权重矩阵：W_s1，其中10指： d_a
        #     nn.Linear(2 * self.hidden_size, 10),
        #     nn.Tanh(True),
        #     # # 对应于论文权重矩阵：W_s2, 其中5指：r
        #     nn.Linear(10, 5)
        # )

        if self.use_cuda:
            self.w_omega = Variable(torch.zeros(config.hidden_size * self.layer_size, self.attention_size).cuda())
            self.u_omega = Variable(torch.zeros(config.attention_size).cuda())
        else:
            self.w_omega = Variable(torch.zeros(config.hidden_size * self.layer_size, self.attention_size))
            self.u_omega = Variable(torch.zeros(self.attention_size))

        self.label = nn.Linear(config.hidden_size * self.layer_size, self.output_size)

    # self.attn_fc_layer = nn.Linear()

    def attention_net(self, lstm_output):
        #print(lstm_output.size()) = (squence_length, batch_size, hidden_size*layer_size)

        output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size*self.layer_size])
        #print(output_reshape.size()) = (squence_length * batch_size, hidden_size*layer_size)

        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        #print(attn_tanh.size()) = (squence_length * batch_size, attention_size)

        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        #print(attn_hidden_layer.size()) = (squence_length * batch_size, 1)

        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, self.hidden_size])
        #print(exps.size()) = (batch_size, squence_length)

        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        #print(alphas.size()) = (batch_size, squence_length)

        alphas_reshape = torch.Tensor.reshape(alphas, [-1, self.hidden_size, 1])
        #print(alphas_reshape.size()) = (batch_size, squence_length, 1)

        state = self.rnn.permute(1, 0, 2)
        #print(state.size()) = (batch_size, squence_length, hidden_size*layer_size)

        attn_output = torch.sum(state * alphas_reshape, 1)
        #print(attn_output.size()) = (batch_size, hidden_size*layer_size)

        return attn_output

    # def forward(self, input_sentences, batch_size=None):
    #     input = self.lookup_table(input_sentences)
    #     input = input.permute(1, 0, 2)
    #
    #     if self.use_cuda:
    #         h_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size).cuda())
    #         c_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size).cuda())
    #     else:
    #         h_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size))
    #         c_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size))
    #
    #     lstm_output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))

        # attn_output = torch.nn.functional.softmax(self.attention(lstm_output), dim=2).permute(0, 2, 1)
    def forward(self, s1_ids, s2_ids, s1_lengths, s2_lengths):
        """Forward inputs and get logits.

        Args:
            s1_ids: (batch_size, max_seq_len)
            s2_ids: (batch_size, max_seq_len)
            s1_lengths: (batch_size)
            s2_lengths: (batch_size)

        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size = s1_ids.shape[0]
        # ids: (batch_size, max_seq_len)
        s1_embed = self.embedding(s1_ids)
        s2_embed = self.embedding(s2_ids)
        # embed: (batch_size, max_seq_len, hidden_size)
        s1_packed: PackedSequence = pack_padded_sequence(
            s1_embed, s1_lengths, batch_first=True, enforce_sorted=False)
        s2_packed: PackedSequence = pack_padded_sequence(
            s2_embed, s2_lengths, batch_first=True, enforce_sorted=False)
        # packed: (sum(lengths), hidden_size)
        self.rnn.flatten_parameters()
        # self.rnn2.flatten_parameters()
        s1_encoded, s1_hidden = self.rnn(s1_packed)
        s2_encoded, s2_hidden = self.rnn(s2_packed)

        s1_hidden = s1_encoded[0].view(batch_size, -1)
        s2_hidden = s2_encoded[0].view(batch_size, -1)

        s1_hidden = self.attention_net(s1_hidden)
        s2_hidden = self.attention_net(s2_hidden)
        hidden = torch.cat([s1_hidden, s2_hidden], dim=-1)

        # logits = self.label(attn_output)
        # return logits
        hidden = self.linear(hidden).view(-1, self.num_classes)
        hidden = self.dropout(hidden)
        logits = nn.functional.softmax(hidden, dim=-1)
        # logits: (batch_size, num_classes)
        return logits

