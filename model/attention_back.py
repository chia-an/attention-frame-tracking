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
            # self.precomputed = ['last-layer']
            # self.bert_layer = bert_layer
            # if bert_layer in self.precomputed:
            #     import pickle
            #     with open('', 'rb') as f_embed:
            #         self.bert_embed = pickle.load(f_embed)

            self.bert_embedding = bert_embedding

            self.embed_dim_text = embed_dim_text
            # self.embed_dim_text = self.bert.config.hidden_size

            if bert_embedding is None:
                from pytorch_pretrained_bert import BertModel

                # self.bert = BertModel.from_pretrained(self.embed_type).to('cuda:1')
                self.bert = BertModel.from_pretrained(self.embed_type)

                for p in self.bert.parameters():
                    p.requires_grad = False

                assert self.bert.config.hidden_size == 768, \
                       'bert.config.hidden_size = {}'.format(
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
            # sent: size[batch_size, seq_len]
            sent: list[word], a word has size[].
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

            return output

        elif self.embed_type == 'word':
            raise NotImplementedError

        elif self.embed_type.startswith('bert'):
            if self.bert_embedding is not None:
                output = self.bert_embedding[str(sent.tolist())]
            else:
                # # size = [1, *]
                # device = sent[0].device
                # sent = torch.tensor(sent, device=device).unsqueeze(0)

                # list[size[batch_size, seq_len, hidden_size]]
                output, _ = self.bert(sent)

                # size = [bert.embed_dim_text]
                output = output[-1].mean(dim=1).view(-1)

            # size = [embed_dim_text]
            output = self.proj(output)

            return output

        else:
            raise Exception('Unknown embedding {}.'.format(self.embed_type))


class EncoderASV(nn.Module):
    def __init__(self, embed_act,
                       embed_slot,
                       encoder_text,
                       embed_dim,
                       rnn_hidden_size,
                       with_utterance=False,
                       rnn_hidden=True,
                       rnn_output=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.embed_dim_act = embed_act.embedding_dim
        self.embed_dim_slot = embed_slot.embedding_dim
        self.embed_dim_text = encoder_text.embed_dim_text

        self.rnn_hidden_size = rnn_hidden_size
        self.with_utterance = with_utterance
        self.rnn_hidden = rnn_hidden
        self.rnn_output = rnn_output

        self.encoder_text = encoder_text
        self.embed_act = embed_act
        self.embed_slot = embed_slot

        self.rnn_asv = nn.GRU(
            input_size=self.embed_dim_act,
            hidden_size=self.rnn_hidden_size,
            num_layers=1,
            bidirectional=True
        )

        # NOTE: non-linear?
        proj_in_dim = 0
        if self.with_utterance:
            proj_in_dim += self.embed_dim_text
        if self.rnn_hidden:
            proj_in_dim += self.rnn_hidden_size * 2
        if self.rnn_output:
            proj_in_dim += self.rnn_hidden_size * 6
        # self.embed_dim_act + self.embed_dim_slot + self.embed_dim_text * 2, 
        # self.rnn_hidden_size * 6 + self.embed_dim_text,
        # self.rnn_hidden_size * 6,
        # self.rnn_hidden_size * 2  + self.embed_dim_text,
        # self.rnn_hidden_size * 2,

        self.proj = nn.Linear(
            proj_in_dim,
            self.embed_dim,
            bias=False
        )
        # self.activation = nn.Tanh()

    def forward(self, asvs, embed_sent):
        """
        asvs: list[(size[], size[], list[word])].
        embed_sent: size[embed_dim_text]
        # sent: list[word], a word has size[1, *].

        NOTE: assume batch = 1.
        """
        # size = [embed_dim_text]
        # embed_sent = self.encoder_text(sent)

        # TODO: one batch per asvs?
        # NOTE: take output instead of hidden
        embeddings = []
        for asv in asvs:
            # size = [embed_dim_act]
            act = self.embed_act(asv[0]).view(-1)

            # size = [embed_dim_slot]
            slot = self.embed_slot(asv[1]).view(-1)

            # size = [embed_dim_text]
            value = self.encoder_text(asv[2])

            # out size = [seq_len, batch, num_dir * hidden_size]
            # hid size = [num_layers * num_dir, batch, hidden_size]
            embed_asv, hidden = self.rnn_asv(
                torch.stack([act, slot, value]).unsqueeze(1))

            device = embed_asv.device
            embedding = torch.zeros(0).to(device)

            if self.with_utterance:
                embedding = torch.cat([embedding, embed_sent])
            if self.rnn_hidden:
                embedding = torch.cat([embedding, hidden.view(-1)])
            if self.rnn_output:
                embedding = torch.cat([embedding, embed_asv.view(-1)])

            embeddings.append(embedding)
            
            # # size = [3 * 2 * hidden_size + embed_dim_text]
            # embeddings.append(torch.cat([embed_asv.view(-1), embed_sent]))
            # # size = [3 * 2 * hidden_size]
            # embeddings.append(embed_asv.view(-1))

            # # size = [2 * hidden_size + embed_dim_text]
            # embeddings.append(torch.cat([hidden.view(-1), embed_sent]))
            # # size = [2 * hidden_size]
            # embeddings.append(hidden.view(-1))

            # # size = [embed_dim_act + embed_dim_slot + embed_dim_text * 2]
            # embeddings.append(torch.cat([act, slot, value, embed_sent]))

        # # size = [n_asvs, embed_dim_act + embed_dim_slot + embed_dim_text * 2]
        # # size = [n_asvs, 3 * 2 * hidden_size + embed_dim_text]
        # size = [n_asvs, 2 * hidden_size + embed_dim_text]
        embed_asvs = torch.stack(embeddings)

        # NOTE: what to do after concat?
        # size = [n_asvs, embed_dim]
        output = self.proj(embed_asvs)
        # output = self.activation(output)

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

        self.rnn_hidden_size = rnn_hidden_size

        self.embed_slot = embed_slot
        self.encoder_text = encoder_text

        self.rnn_frame = nn.GRU(
            input_size=self.embed_dim_slot + self.embed_dim_text,
            hidden_size=self.rnn_hidden_size,
            num_layers=1,
            bidirectional=True
        )

        self.proj = nn.Linear(
            # self.embed_dim_slot + self.embed_dim_text,
            self.rnn_hidden_size * 2,
            self.embed_dim,
            bias=False
        )
        self.activation = nn.Tanh()

    def forward(self, frames):
        """
        frames: list[list[(size[], list[word])]]

        output: list[size[n_svs, embed_dim]]
        output: list[size[embed_dim]]
        """
        # TODO: One frame per batch?
        embed_frames = []
        for frame in frames:
            embed_frame = []
            for sv in frame:
                # size = [embed_dim_slot]
                slot = self.embed_slot(sv[0]).view(-1)

                # size = [embed_dim_text]
                value = self.encoder_text(sv[1])

                # size = [embed_dim_slot + embed_dim_text]
                embed_sv = torch.cat([slot, value])
                embed_frame.append(embed_sv)

            # size = [n_svs, embed_dim_slot + embed_dim_text]
            embed_frame = torch.stack(embed_frame)

            # size = [seq_len, batch, num_dir * hidden_size]
            # hid size = [num_layers * num_dir, batch, hidden_size]
            embed_frame, hidden = self.rnn_frame(embed_frame.view([
                -1, 1, self.embed_dim_slot + self.embed_dim_text]))

            # NOTE: what to do after concat?
            # size = [n_svs, embed_dim]
            embed_frame = self.proj(embed_frame.view([
                -1, 2 * self.rnn_hidden_size]))

            # # size = [embed_dim]
            # embed_frame = self.proj(hidden.view(-1))

            embed_frames.append(embed_frame)

        return embed_frames


class AttFrame(nn.Module):
    """
    Modified from torchnlp.nn.attention.
    """
    def __init__(self, dim_asv,
                       dim_sv,
                       dim_proj_v,
                       attention_type='simple'):
        super().__init__()

        self.dim_asv = dim_asv
        self.dim_sv = dim_sv
        self.dim_proj_v = dim_proj_v

        self.attention_type = attention_type

        if self.attention_type == 'general':
            self.proj_q = nn.Linear(
                self.dim_asv,
                self.dim_sv,
                bias=False,
            )

        if self.attention_type == 'content':
            self.content_w = nn.Linear(self.dim_sv, 1)
            self.content_a = nn.Tanh()

        self.softmax = nn.Softmax(dim=-1)

        # self.att_weight = nn.Linear(32, 1, bias=False)

        # self.proj_v = nn.Linear(
        #     self.dim_sv,
        #     self.dim_proj_v,
        # )
        # self.activation = nn.Tanh()

    def forward_position(self, frame):
        """
        frame: size[n_svs, dim_sv]

        output: size[dim_sv], attention features.
        attention_weights: None.
        """
        output = self.att_weight(frame.transpose(0, 1))

        return output, None

    def forward(self, asvs, frame):
        """
        asvs: size[n_asvs, dim_asv]
        frame: size[n_svs, dim_sv]

        output: size[n_asvs, dim_proj_v], attention features.
        attention_weights: size[n_asvs, n_svs].

        NOTE: batch = 1 for now.
        NOTE: Luong, dot product
        https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html
        """

        if self.attention_type == 'simple':
            assert asvs.size()[-1] == frame.size()[-1], \
                   'asvs.size() = {}, frame.size() = {}'.format(
                       asvs.size(), frame.size())

            # NOTE: what is contiguous()?
            # size = [n_asvs, n_svs]
            attention_scores = torch.matmul(
                asvs, frame.transpose(0, 1).contiguous())

        elif self.attention_type == 'general':
            asvs = self.proj_q(asvs)

            # size = [n_asvs, n_svs]
            attention_scores = torch.matmul(
                asvs, frame.transpose(0, 1).contiguous())

        elif self.attention_type == 'content':
            # size = [n_svs]
            attention_score = self.content_w(frame).view(-1)
            attention_score = self.content_a(attention_score)

            # size = [n_asvs, n_svs]
            attention_scores = attention_score.repeat(asvs.size()[0], 1)

        else:
            raise Exception('Unknown attention {}.'.format(attention_type))

        # size = [n_asvs, n_svs]
        attention_weights = self.softmax(attention_scores)

        # NOTE: non-linear after linear combination
        # size = [n_asvs, dim_sv]
        output = torch.matmul(attention_weights, frame)

        return output, attention_weights

    def forward_general(self, asvs, frame):
        """
        asvs: size[n_asvs, dim_asv]
        frame: size[n_svs, dim_sv]

        output: size[n_asvs, dim_proj_v], attention features.
        attention_weights: size[n_asvs, n_svs].

        NOTE: batch = 1 for now.
        NOTE: Luong, general
        https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html
        """

        # n_asvs, dim_asv = asvs.size()
        # n_svs = frame.size(0)

        # NOTE: separate proj_q into proj_q and proj_k?
        # size = [n_asvs, dim_sv]
        asvs = self.proj_q(asvs)

        # NOTE: what is contiguous()?
        # size = [n_asvs, n_svs]
        attention_scores = torch.matmul(
            asvs, frame.transpose(0, 1).contiguous())

        # size = [n_asvs, n_svs]
        attention_weights = self.softmax(attention_scores)

        # NOTE: non-linear after linear combination
        # size = [n_asvs, dim_sv]
        output = torch.matmul(attention_weights, frame)

        # size = [n_asvs, dim_proj_v]
        # output = self.proj_v(output)
        # output = self.activation(output)

        return output, attention_weights


class Model(nn.Module):
    def __init__(self, n_acts,
                       n_slots,
                       n_tris,
                       embed_type='trigram',
                       embed_dim_act=256,
                       embed_dim_slot=256,
                       embed_dim_text=256,
                       embed_dim_tri=256,
                       embed_dim=256,
                       bert_embedding=None,
                       asv_with_utterance=False,
                       asv_rnn_hidden=True,
                       asv_rnn_output=False,
                       attention_type='simple',
                       asv_rnn_hidden_size=256,
                       frame_rnn_hidden_size=256,
                       device=torch.device('cpu')):
        super().__init__()

        print('Initialize attention model.')

        self.device = device

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
            embed_type=embed_type,
            embed_tri=self.embed_tri,
            embed_dim_text=embed_dim_text,
            bert_embedding=bert_embedding
        ).to(self.device)
        self.encoder_asv = EncoderASV(
            embed_act=self.embed_act,
            embed_slot=self.embed_slot,
            encoder_text=self.encoder_text,
            embed_dim=embed_dim,
            rnn_hidden_size=asv_rnn_hidden_size,
            with_utterance=asv_with_utterance,
            rnn_hidden=asv_rnn_hidden,
            rnn_output=asv_rnn_output
        ).to(self.device)
        self.encoder_frame = EncoderFrame(
            embed_slot=self.embed_slot,
            encoder_text=self.encoder_text,
            embed_dim=embed_dim,
            rnn_hidden_size=frame_rnn_hidden_size
        ).to(self.device)
        self.attention = AttFrame(
            dim_asv=embed_dim,
            dim_sv=embed_dim,
            dim_proj_v=embed_dim,
            attention_type=attention_type
        ).to(self.device)

        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=1e-4,
            weight_decay=1e-4
        )

    def forward(self, input):
        """
        input: (sent, asvs, frames, active_frame, new_frames)
        output: size[n_asvs, n_frames]

        No batching, one input per turn.
        """
        sent, asvs, frames, active_frame, new_frames = input

        # Transform to cuda
        if self.encoder_text.embed_type.startswith('bert'):
            sent = sent.to(self.device)
            asvs = [(a.to(self.device),
                     s.to(self.device),
                     v.to(self.device)) for a, s, v in asvs]
            frames = [[(s.to(self.device),
                        v.to(self.device))
                       for s, v in frame]
                      for frame in frames]
        else:
            sent = [word.to(self.device) for word in sent]
            asvs = [(a.to(self.device),
                     s.to(self.device),
                     [w.to(self.device) for w in v]) for a, s, v in asvs]
            frames = [[(s.to(self.device),
                        [w.to(self.device) for w in v])
                       for s, v in frame]
                      for frame in frames]
        # active_frame.to(self.device)
        # new_frames.to(self.device)

        # size = [embed_dim_text]
        embed_sent = self.encoder_text(sent)

        # size = [n_asvs, embed_dim]
        embed_asvs = self.encoder_asv(asvs, embed_sent)

        # size = list[size[n_svs, embed_dim]]
        # # size = list[size[embed_dim]]
        embed_frames = self.encoder_frame(frames)

        sims = []
        for embed_frame in embed_frames:
            # Attention
            # size = [n_asvs, embed_dim]
            embed_att_frame, _ = self.attention(embed_asvs, embed_frame)

            # size = [n_asvs]
            sim = (embed_att_frame * embed_asvs).sum(dim=-1)

            # # No attention
            # sim = torch.matmul(embed_asvs, embed_frame)

            # # No query attention
            # # size = [embed_dim]
            # embed_att_frame, _ = self.attention(embed_frame)
            # # size = [n_asvs]
            # sim = torch.matmul(embed_asvs, embed_att_frame).view(-1)

            sims.append(sim)

        # size = [n_asvs, n_frames]
        output = torch.stack(sims, dim=1)
        output = torch.nn.LogSoftmax(dim=1)(output)

        return output

    def step(self, input, fs, train=True):
        """
        fs: size[1, n_asvs]
        """
        fs = fs.to(self.device)

        self.optimizer.zero_grad()

        output = self.forward(input)
        self.loss = self.criterion(output, fs.view(-1))
        # preds = output.argmax(dim=1)

        if train:
            self.loss.backward()
            self.optimizer.step()

        return self.loss.item(), output.unsqueeze(0)

