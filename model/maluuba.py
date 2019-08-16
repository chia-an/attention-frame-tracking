import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class EncoderText(nn.Module):
    def __init__(self, embed_type,
                       embed_tri,
                       embed_dim_text,
                       bert_embedding=None):
        super().__init__()

        self.embed_type = embed_type

        if self.embed_type == 'trigram':
            self.embed_dim_tri = embed_tri.embedding_dim
            self.embed_dim_text = embed_dim_text
            self.rnn_hidden_size = embed_dim_text // 2
            assert self.rnn_hidden_size * 2 == self.embed_dim_text, \
                   self.embed_dim_text

            self.embed_tri = embed_tri
            self.rnn_text = nn.GRU(
                input_size=self.embed_dim_tri,
                hidden_size=self.rnn_hidden_size,
                num_layers=1,
                bidirectional=True
            )

        elif self.embed_type == 'word':
            raise NotImplementedError

        elif self.embed_type.startswith('bert'):
            self.bert_embedding = bert_embedding

            self.embed_dim_text = embed_dim_text

            if bert_embedding is None:
                from pytorch_pretrained_bert import BertModel

                self.bert = BertModel.from_pretrained(self.embed_type)

                for p in self.bert.parameters():
                    p.requires_grad = False

                assert self.bert.config.hidden_size == 768, \
                       'bert.config.hidden_size= {}'.format(
                            self.bert.config.hidden_size)

            self.proj = nn.Linear(
                768,
                self.embed_dim_text,
                bias=False
            )

        else:
            raise Exception('Unknown embedding {}.'.format(self.embed_type))

    def forward(self, sent):
        """
        embed_type = 'trigram':
            sent: list[word], a word has size[1, *].
            output: size = [embed_dim_text].
        embed_type = 'bert':
            sent: list[word], a word has size[1, *].
            output: size = [embed_dim_text].

        NOTE: assume batch = 1.
        """

        if self.embed_type == 'trigram':
            # size = [*, embed_dim_tri]
            words = torch.stack([
                self.embed_tri(word.long()).sum(dim=1) for word in sent])

            # size = [n_dir, batch, hidden_size]
            _, hidden = self.rnn_text(words.view(-1, 1, self.embed_dim_tri))

            # size = [hidden_size * 2]
            output = hidden.view(-1)

        elif self.embed_type == 'word':
            raise NotImplementedError

        elif self.embed_type.startswith('bert'):
            if self.bert_embedding is not None:
                output = self.bert_embedding[str(sent.tolist())]
            else:
                # list[size[batch_size, seq_len, hidden_size]]
                output, _ = self.bert(sent)

                # size = [bert.embed_dim_text]
                output = output[-1].mean(dim=1).view(-1)

            # size = [embed_dim_text]
            output = self.proj(output)
        
        else:
            raise exception('Unkown embedding {}.'.format(self.embed_type))

        return output


class EncoderASV(nn.Module):
    def __init__(self, embed_act,
                       embed_slot,
                       encoder_text,
                       embed_dim,
                       rnn_hidden_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_dim_act = embed_act.embedding_dim
        self.embed_dim_slot = embed_slot.embedding_dim
        self.embed_dim_text = encoder_text.embed_dim_text
        # self.embed_dim_text = embed_dim_text
        # self.embed_dim_act = embed_dim_act
        # self.embed_dim_slot = embed_dim_slot

        self.rnn_hidden_size = rnn_hidden_size
        # self.embed_dim_asv = embed_dim_asv
        # self.rnn_hidden_size = self.embed_dim_asv // 2
        # assert self.rnn_hidden_size * 2 == self.embed_dim_asv, \
        #        self.embed_dim_asv

        # TODO: num_embeddings = size of act/slot dict
        self.encoder_text = encoder_text
        self.embed_act = embed_act
        self.embed_slot = embed_slot
        # self.embed_act = nn.Embedding(num_embeddings=?,
        #                               embedding_dim=self.embed_dim_act)
        # self.embed_slot = nn.Embedding(num_embeddings=?,
        #                                embedding_dim=self.embed_dim_slot)
        self.rnn_asv = nn.GRU(
            input_size=self.embed_dim_act,
            hidden_size=self.rnn_hidden_size,
            num_layers=1,
            bidirectional=True
        )
        self.proj = nn.Linear(
            self.rnn_hidden_size * 2 + self.embed_dim_text,
            self.embed_dim,
            bias=False
        )

    def forward(self, asvs, embed_sent):
        """
        asvs: list[(size[], size[], list[word])].
        embed_sent: size[embed_dim_text]
        # sent: list[word], a word has size[1, *].

        NOTE: assume batch = 1.
        NOTE: not clear from the paper.

        "A second bi-directional GRU r_asv computes a hidden activation
        for each of the (act, slot, value) triples in the current turn.
        We compute a value summary vector m_asv by appending each hidden
        state of r_asv with the utterance embedding and projecting to a
        256-dimensional space."
        """
        # size = [embed_dim_text]
        # embed_sent = self.encoder_text(sent)

        # TODO: one batch per asvs?
        # NOTE: take output instead of hidden
        hiddens = []
        for asv in asvs:
            # size = [embed_dim_act]
            act = self.embed_act(asv[0])

            # size = [embed_dim_slot]
            slot = self.embed_slot(asv[1])

            # size = [embed_dim_text]
            value = self.encoder_text(asv[2])

            # size = [n_dir, batch, hidden_size]
            _, hidden = self.rnn_asv(torch.stack([
                act.view(1, -1),
                slot.view(1, -1),
                value.view(1, -1),
            ]))

            # size = [hidden_size * 2 + embed_dim_text]
            hidden = torch.cat([hidden.view(-1), embed_sent])
            hiddens.append(hidden)

        # size = [n_asvs, embed_dim_asv * 2 + embed_dim_text]
        embed_asvs = torch.stack(hiddens)

        # size = [n_asvs, embed_dim]
        output = self.proj(embed_asvs)

        return output


class EncoderFrame(nn.Module):
    def __init__(self, embed_slot,
                       encoder_text,
                       embed_dim,
                       rnn_hidden_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_dim_slot = embed_slot.embedding_dim
        self.embed_dim_text = encoder_text.embed_dim_text
        # self.embed_dim_slot = embed_dim_slot
        # self.embed_dim_text = embed_dim_text

        self.rnn_hidden_size = rnn_hidden_size
        # self.embed_dim_asv = embed_dim_asv
        # self.rnn_hidden_size = self.embed_dim_asv // 2
        # assert self.rnn_hidden_size * 2 == self.embed_dim_asv, \
        #        self.embed_dim_asv

        # TODO: num_embeddings = size of act/slot dict
        self.embed_slot = embed_slot
        self.encoder_text = encoder_text
        # self.embed_act = nn.Embedding(num_embeddings=?,
        #                               embedding_dim=self.embed_dim_act)
        # self.embed_slot = nn.Embedding(num_embeddings=?,
        #                                embedding_dim=self.embed_dim_slot)
        self.rnn_frame = nn.GRU(
            input_size=self.embed_dim_slot + self.embed_dim_text,
            hidden_size=self.rnn_hidden_size,
            num_layers=1,
            bidirectional=True
        )
        self.proj = nn.Linear(
            self.embed_dim_slot + self.embed_dim_text,
            self.embed_dim,
            bias=False
        )

    def forward(self, frames):
        # TODO: One batch per frames?
        embed_frames = []
        for frame in frames:
            embed_frame = []
            for sv in frame:
                # size = [embed_dim_slot]
                slot = self.embed_slot(sv[0])

                # size = [embed_dim_text]
                value = self.encoder_text(sv[1])

                # size = [embed_dim_slot + embed_dim_text]
                embed_sv = torch.cat([slot.view(-1), value])
                embed_frame.append(embed_sv)

            # size = [n_svs, embed_dim_slot + embed_dim_text]
            embed_frame = torch.stack(embed_frame)

            # size = [n_dir, 1, rnn_hidden_size]
            _, embed_frame = self.rnn_frame(embed_frame.view([
                -1, 1, self.embed_dim_slot + self.embed_dim_text]))

            # size = [rnn_hidden_size * 2]
            # embed_frame = torch.cat([embed_frame[0, :, :],
            #                          embed_frame[1, :, :]])
            embed_frames.append(embed_frame.view(-1))

        # size = [n_frames, rnn_hidden_size * 2]
        output = torch.stack(embed_frames)

        # size = [n_frames, embed_dim]
        output = self.proj(output)

        return output


class Model(nn.Module):
    def __init__(self, n_acts,
                       n_slots,
                       n_tris,
                       embed_dim_act=256,
                       embed_dim_slot=256,
                       embed_dim_text=256,
                       embed_dim_tri=256,
                       embed_dim=256,
                       asv_rnn_hidden_size=256,
                       frame_rnn_hidden_size=256,
                       cuda=True):
        super().__init__()

        self.device = torch.device('cuda' if cuda else 'cpu')

        self.embed_act = nn.Embedding(
            num_embeddings=n_acts,
            embedding_dim=embed_dim_act
        ).to(self.device)
        self.embed_slot = nn.Embedding(
            num_embeddings=n_slots,
            embedding_dim=embed_dim_slot
        ).to(self.device)
        self.embed_tri = nn.Embedding(
            num_embeddings=n_tris,
            embedding_dim=embed_dim_tri
        ).to(self.device)

        self.encoder_text = EncoderText(
            embed_tri=self.embed_tri,
            embed_dim_text=embed_dim_text
        ).to(self.device)
        self.encoder_asv = EncoderASV(
            embed_act=self.embed_act,
            embed_slot=self.embed_slot,
            encoder_text=self.encoder_text,
            embed_dim=embed_dim,
            rnn_hidden_size=asv_rnn_hidden_size
        ).to(self.device)
        self.encoder_frame = EncoderFrame(
            embed_slot=self.embed_slot,
            encoder_text=self.encoder_text,
            embed_dim=embed_dim,
            rnn_hidden_size=frame_rnn_hidden_size
        ).to(self.device)

        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.parameters(),
                                    lr=1e-4,
                                    weight_decay=1e-4)

    def forward(self, input):
        """
        input: (sent, asvs, frames, active_frame, new_frames)
        output: size[n_asvs, n_frames]

        No batching, one input per turn.
        """
        sent, asvs, frames, active_frame, new_frames = input

        # Transform to cuda
        sent = [word.to(self.device) for word in sent]
        asvs = [(a.to(self.device),
                 s.to(self.device),
                 [w.to(self.device) for w in v]) for a, s, v in asvs]
        frames = [[(s.to(self.device),
                    [w.to(self.device) for w in v])
                   for s, v in frame]
                  for frame in frames]
        active_frame.to(self.device)
        new_frames.to(self.device)

        embed_sent = self.encoder_text(sent)

        # size = [n_asvs, embed_dim]
        embed_asvs = self.encoder_asv(asvs, embed_sent)

        # size = [n_frames, embed_dim]
        embed_frames = self.encoder_frame(frames)

        # size = [n_asvs, n_frames]
        output = torch.mm(embed_asvs, embed_frames.t())
        output = torch.nn.LogSoftmax(dim=1)(output)

        # TODO: add normalized string edit distance between values.
        # TODO: linear combination of the two similarities.

        return output

    def step(self, input, fs, train=True):
        fs = fs.to(self.device)

        self.optimizer.zero_grad()

        output = self.forward(input)
        self.loss = self.criterion(output, fs.view(-1))
        preds = None

        if train:
            self.loss.backward()
            self.optimizer.step()
        else:
            preds = output.argmax(dim=1)

        # return self.loss.item(), output, preds
        return self.loss.item(), output.unsqueeze(0)

