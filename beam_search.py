import torch

from make_vocab import END_TOKEN

# Reference:
# https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/translate/beam_search.py


class BeamSearch(object):
    """ Implement beam search decoding

    Args:
        beam_size (int): Number of beams to use.
        batch_size (int): Current batch size.

    Attributes:
        _batch_offset (LongTensor): Shape ``(B,)``.
        _beam_offset (LongTensor): Shape ``(batch_size x beam_size,)``.
        alive_seq (LongTensor): See base.
        topk_log_probs (FloatTensor): Shape ``(B , beam_size,)``. These
            are the scores used for the topk operation.
        select_indices (LongTensor or NoneType): Shape
            ``(B x beam_size,)``. This is just a flat view of the
            ``_batch_index``.
        topk_scores (FloatTensor): Shape
            ``(B, beam_size)``. These are the
            scores a sequence will receive if it finishes.
        topk_ids (LongTensor): Shape ``(B, beam_size)``. These are the
            word indices of the topk predictions.
        _batch_index (LongTensor): Shape ``(B, beam_size)``.
        hypotheses (list[list[Tuple[Tensor]]]): Contains a tuple
            of score (float), sequence (long).
    """

    def __init__(self, beam_size, batch_size):
        self.beam_size = beam_size
        self.batch_size = batch_size

        # result cacheing
        self.hypotheses = [[] for _ in range(batch_size)]

        # beam state
        self.top_beam_finished = torch.zeros([batch_size], dtype=torch.uint8)
        self._batch_offset = torch.arange(batch_size, dtype=torch.long)
        self._beam_offset = torch.arange(
            0, batch_size * beam_size, step=beam_size, dtype=torch.long)
        self.topk_log_probs = torch.tensor(
            [0.0] + [float("-inf")] * (beam_size - 1)
        ).repeat(batch_size)
        self.select_indices = None

        # buffers for the topk scores and 'backpointer'
        self.topk_scores = torch.empty((batch_size, beam_size),
                                       dtype=torch.float)
        self.topk_ids = torch.empty((batch_size, beam_size), dtype=torch.long)
        self._batch_index = torch.empty(
            [batch_size, beam_size], dtype=torch.long)
        self.done = False

    @property
    def current_predictions(self):
        return self.alive_seq[:, -1]

    @property
    def current_origin(self):
        return self.select_indices

    def __len__(self):
        return self.alive_seq.size(1)

    def advance(self, log_probs):
        """Args:
            log_probs: [_B * self.beam_size, vocab_size]
        """
        vocab_size = log_probs.size(-1)

        # using integer division to get an integer _B without casting
        _B = log_probs.size(0) // self.beam_size

        # step = len(self)
        # Multiply probs by the beam probability
        log_probs += self.topk_log_probs.view(_B * self.beam_size, 1)
        log_probs = log_probs.view(_B, self.beam_size * vocab_size)
        torch.topk(log_probs, self.beam_size, dim=-1,
                   out=(self.topk_scores, self.topk_ids))

        # Resolve beam origin and map to batch index flat representation
        # _batch_index = [_B, beam_size]
        # _beam_offset = (_B,)
        # topk_ids [_B, beam_size]
        torch.div(self.topk_ids, vocab_size, out=self._batch_index)
        self._batch_index += self._beam_offset[:_B].unsqueeze(1)
        self.select_indices = self._batch_index.view(_B * self.beam_size)

        # resolve true word ids
        self.topk_ids.fmod_(vocab_size)

        # update topk_log_probs
        self.topk_log_probs = self.topk_scores

        # append last prediction
        self.alive_seq = torch.cat(
            [self.alive_seq.index_select(0, self.select_indices),
             self.topk_ids.view(_B * self.beam_size, 1)], -1,)

        self.is_finished = self.topk_ids.eq(END_TOKEN)

    def update_finished(self):

        _B_old = self.topk_log_probs.size(0)
        step = self.alive_seq.size(-1)
        self.topk_log_probs.masked_fill_(self.is_finished, -1e10)
        self.top_beam_finished |= self.is_finished[:, 0].eq(1)
        predictions = self.alive_seq.view(_B_old, self.beam_size, step)

        non_finished_batch = []
        for i in range(self.is_finished.size(0)):
            b = self._batch_offset[i]
            finished_hyp = self.is_finished[i].nonzero().view(-1)
            # Store finished hypotheses for this batch
            for j in finished_hyp:
                self.hypotheses[b].append((
                    self.topk_scores[i, j],
                    predictions[i, j, 1:]  # Ignore start token
                ))

            finish_flag = self.top_beam_finished[i] != 0
            # if len(self.hypotheses[b]) >= self.beam_size:
            if not finish_flag:
                non_finished_batch.append(i)
        non_finished = torch.tensor(non_finished_batch)
        if len(non_finished) == 0:  # if all finished
            self.done = True
            return

        self.batch_size = non_finished.size(0)
        # Remove finished batches for the next step
        self._batch_offset = self._batch_offset.index_select(0, non_finished)
        self.topk_log_probs = self.topk_log_probs.index_select(0, non_finished)

        self._batch_index = self._batch_index.index_select(0, non_finished)
        self.select_indices = self._batch_index.view(
            self.batch_size * self.beam_size)
        self.alive_seq = predictions.index_select(0, non_finished) \
            .view(-1, self.alive_seq.size(-1))
        self.topk_scores = self.topk_scores.index_select(0, non_finished)
        self.topk_ids = self.topk_ids.index_select(0, non_finished)
