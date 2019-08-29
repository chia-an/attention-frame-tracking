import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import math


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

        proj_in_dim = 0
        if self.with_utterance:
            proj_in_dim += self.embed_dim_text
        if self.rnn_hidden:
            proj_in_dim += self.rnn_hidden_size * 2
        if self.rnn_output:
            proj_in_dim += self.rnn_hidden_size * 6

        self.proj = nn.Linear(
            proj_in_dim,
            self.embed_dim,
            bias=False
        )

    def forward(self, asvs, embed_sent):
        """
        asvs: list[(size[], size[], list[word])].
        embed_sent: size[embed_dim_text]
        # sent: list[word], a word has size[1, *].

        NOTE: assume batch = 1.
        """
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

        embed_asvs = torch.stack(embeddings)

        # size = [n_asvs, embed_dim]
        output = self.proj(embed_asvs)

        return output


class EncoderFrame(nn.Module):
    def __init__(self, embed_slot,
                       encoder_text,
                       embed_dim,
                       rnn_hidden_size,
                       with_attention=True):
        super().__init__()

        self.with_attention = with_attention

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
            self.rnn_hidden_size * 2,
            self.embed_dim,
            bias=False
        )
        self.activation = nn.Tanh()

    def forward(self, frames):
        """
        frames: list[list[(size[], list[word])]]

        output:
            with_attention = True, list[size[n_svs, embed_dim]]
            with_attention = False, list[size[embed_dim]]
        """
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

            if self.with_attention:
                # NOTE: what to do after concat?
                # size = [n_svs, embed_dim]
                embed_frame = self.proj(embed_frame.view([
                    -1, 2 * self.rnn_hidden_size]))
            else:
                # size = [embed_dim]
                embed_frame = self.proj(hidden.view(-1))

            embed_frames.append(embed_frame)

        return embed_frames


class AttFrame(nn.Module):
    def __init__(self, dim_asv,
                       dim_sv,
                       dim_proj_v,
                       dim_att,
                       attention_type='simple'):
        super().__init__()

        self.dim_asv = dim_asv
        self.dim_sv = dim_sv
        self.dim_proj_v = dim_proj_v
        self.dim_att = dim_att

        self.attention_type = attention_type

        if self.attention_type == 'simple':
            pass

        elif self.attention_type == 'scaled-simple':
            pass

        elif self.attention_type == 'general':
            self.proj_q = nn.Linear(
                self.dim_asv,
                self.dim_sv,
                bias=False,
            )

        elif self.attention_type == 'cosine':
            self.cos = nn.CosineSimilarity()

        elif self.attention_type == 'query-key':
            self.proj_q = nn.Linear(self.dim_asv, self.dim_att, bias=False)
            self.proj_k = nn.Linear(self.dim_sv, self.dim_att, bias=False)

        elif self.attention_type == 'scaled-query-key':
            self.proj_q = nn.Linear(self.dim_asv, self.dim_att, bias=False)
            self.proj_k = nn.Linear(self.dim_sv, self.dim_att, bias=False)

        elif self.attention_type == 'content':
            self.content_w = nn.Linear(self.dim_sv, 1)
            self.content_a = nn.Tanh()

        else:
            raise Exception('Unknown atention type {}.'. \
                            format(attention_type))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, asvs, frame):
        """
        asvs: size[n_asvs, dim_asv]
        frame: size[n_svs, dim_sv]

        output: size[n_asvs, dim_proj_v], attention features.
        attention_weights: size[n_asvs, n_svs].

        NOTE: batch = 1.
        NOTE: Luong, dot product
        https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html
        """

        if self.attention_type == 'simple':
            assert asvs.size()[-1] == frame.size()[-1], \
                   'asvs.size() = {}, frame.size() = {}'.format(
                       asvs.size(), frame.size())

            # size = [n_asvs, n_svs]
            attention_scores = torch.matmul(
                asvs, frame.transpose(0, 1).contiguous())

        elif self.attention_type == 'scaled-simple':
            # size = [n_asvs, n_svs]
            attention_scores = torch.matmul(
                asvs, frame.transpose(0, 1).contiguous())
            attention_scores *= 1 / math.sqrt(frame.size(1))

        elif self.attention_type == 'general':
            asvs = self.proj_q(asvs)
            attention_scores = torch.matmul(
                asvs, frame.transpose(0, 1).contiguous())

        elif self.attention_type == 'cosine':
            # size = [n_asvs, n_svs]
            attention_scores = self.cos(
                asvs.unsqueeze(2).repeat(1, 1, frame.size(0)),
                frame.transpose(0, 1).repeat(asvs.size(0), 1, 1)
            )

        elif self.attention_type == 'query-key':
            # size = [n_asvs, dim_att]
            q = self.proj_q(asvs)

            # size = [n_svs, dim_att]
            k = self.proj_k(frame)

            # size = [n_asvs, n_svs]
            attention_scores = torch.matmul(q, k.transpose(0, 1))

        elif self.attention_type == 'scaled-query-key':
            # size = [n_asvs, dim_att]
            q = self.proj_q(asvs)

            # size = [n_svs, dim_att]
            k = self.proj_k(frame)

            # size = [n_asvs, n_svs]
            attention_scores = torch.matmul(q, k.transpose(0, 1))
            attention_scores *= 1 / math.sqrt(k.size(1))

        elif self.attention_type == 'content':
            # size = [n_svs]
            attention_score = self.content_w(frame).view(-1)
            attention_score = self.content_a(attention_score)

            # size = [n_asvs, n_svs]
            attention_scores = attention_score.repeat(asvs.size(0), 1)

        else:
            raise Exception('Unknown attention {}.'.format(attention_type))

        # size = [n_asvs, n_svs]
        attention_weights = self.softmax(attention_scores)

        # size = [n_asvs, dim_sv]
        output = torch.matmul(attention_weights, frame)

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
                       att_dim=64,
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
        self.with_attention = attention_type != 'no'

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
            rnn_hidden_size=frame_rnn_hidden_size,
            with_attention=self.with_attention
        ).to(self.device)
        if self.with_attention:
            self.attention = AttFrame(
                dim_asv=embed_dim,
                dim_sv=embed_dim,
                dim_proj_v=embed_dim,
                dim_att=att_dim,
                attention_type=attention_type
            ).to(self.device)

        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=1e-4,
            weight_decay=1e-4
        )

    def set_device(self, device):
        self.to(device)
        self.device = device
        self.optimizer.load_state_dict(self.optimizer.state_dict())
        if getattr(self.encoder_text, 'bert_embedding', None) is not None:
            for k, v in self.encoder_text.bert_embedding.items():
                self.encoder_text.bert_embedding[k] = v.to(device)

    def forward(self, input):
        """
        input: (sent, asvs, frames, active_frame, new_frames)
        output: size[n_asvs, n_frames]

        No batching, one turn per batch.
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

        # size = [embed_dim_text]
        embed_sent = self.encoder_text(sent)

        # size = [n_asvs, embed_dim]
        embed_asvs = self.encoder_asv(asvs, embed_sent)

        # size = list[size[n_svs, embed_dim]], with_attention = True
        # size = list[size[embed_dim]], with_attention = False
        embed_frames = self.encoder_frame(frames)

        sims = []
        for embed_frame in embed_frames:
            if self.with_attention == True:
                # Attention
                # size = [n_asvs, embed_dim]
                embed_att_frame, _ = self.attention(embed_asvs, embed_frame)

                # size = [n_asvs]
                sim = (embed_att_frame * embed_asvs).sum(dim=-1)
            else:
                # No attention
                sim = torch.matmul(embed_asvs, embed_frame)

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

        if train:
            self.loss.backward()
            self.optimizer.step()

        return self.loss.item(), output.unsqueeze(0)

