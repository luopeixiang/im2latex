import torch

from beam_search import BeamSearch
from make_vocab import END_TOKEN, PAD_TOKEN, START_TOKEN
from utils import tile


class LatexProducer(object):
    """
    Model wrapper, implementing batch greedy decoding and
    batch beam search decoding
    """

    def __init__(self, model, vocab, max_len=150):
        """args:
            the path to model checkpoint
        """
        self._model = model
        self._sign2id = vocab.sign2id
        self._id2sign = vocab.id2sign
        self.max_len = max_len

    def __call__(self, imgs, beam_size=1):
        """args:
            imgs: images need to be decoded
            beam_size: if equal to 1, use greedy decoding
           returns:
            formulas list of batch_size length
        """
        if beam_size == 1:
            results = self._greedy_decoding(imgs)
        else:
            results = self._beam_search_decoding(imgs, beam_size)
        return results

    def _greedy_decoding(self, imgs):
        enc_outs, hiddens = self.encode(imgs)
        dec_states, O_t = self._model.init_decoder(enc_outs, hiddens)

        batch_size = imgs.size(0)
        # storing decoding results
        formulas_idx = torch.ones(batch_size, self.max_len).long() * PAD_TOKEN
        # first decoding step's input
        tgt = torch.ones(batch_size, 1).long() * START_TOKEN
        for t in range(self.max_len):
            dec_states, O_t, logit = self._model.step_decoding(
                dec_states, O_t, enc_outs, tgt)
            tgt = torch.argmax(logit, dim=1, keepdim=1)
            formulas_idx[:, t:t + 1] = tgt
        results = self._idx2formulas(formulas_idx)
        return results

    def _beam_search_decoding(self, imgs, beam_size):
        B = imgs.size(0)
        # use batch_size*beam_size as new Batch
        imgs = tile(imgs, beam_size, dim=0)
        enc_outs, hiddens = self.model.encode(imgs)
        dec_states, O_t = self.model.init_decoder(enc_outs, hiddens)

        new_B = imgs.size(0)
        # first decoding step's input
        tgt = torch.ones(new_B, 1).long() * START_TOKEN
        beam = BeamSearch(beam_size, B)
        for t in range(self.max_len):
            tgt = beam.current_predictions.unsqueeze(1)
            dec_states, O_t, probs = self.step_decoding(
                dec_states, O_t, enc_outs, tgt)
            log_probs = torch.log(probs)

            beam.advance(log_probs)
            any_beam_is_finished = beam.is_finished.any()
            if any_beam_is_finished:
                beam.update_finished()
                if beam.done:
                    break

            select_indices = beam.current_origin
            if any_beam_is_finished:
                # Reorder states
                h, c = dec_states
                h = h.index_select(0, select_indices)
                c = c.index_select(0, select_indices)
                dec_states = (h, c)
                O_t = O_t.index_select(0, select_indices)
        # get results
        formulas_idx = torch.stack([hyps[1] for hyps in beam.hypotheses],
                                   dim=0)
        results = self._idx2formulas(formulas_idx)
        return results

    def _idx2formulas(self, formulas_idx):
        """convert formula id matrix to formulas list"""
        results = []
        for id_ in formulas_idx:
            id_list = id_.tolist()
            result = []
            for sign_id in id_list:
                if sign_id != END_TOKEN:
                    result.append(self._id2sign[sign_id])
                else:
                    break
            results.append(" ".join(result))
        return results

    def bs_decoding(self, img, beam_size):
        """beam search decoding not support batch
          For Testing
        """

        # encoding
        img = img.unsqueeze(0)  # [1, C, H, W]
        enc_outs, hiddens = self.model.encode(img)

        # prepare data for decoding
        dec_states, O_t = self.model.init_decoder(enc_outs, hiddens)
        dec_states = (dec_states[0].expand(beam_size, -1),
                      dec_states[1].expand(beam_size, -1))
        O_t = O_t.expand(beam_size, -1)

        # store top k ids (k is less or equal to beam_size)
        topk_ids = torch.ones(beam_size).long() * START_TOKEN
        topk_log_probs = torch.Tensor([0.0] + [-1e10] * (beam_size - 1))
        seqs = torch.ones(beam_size, 1).long() * START_TOKEN
        # store complete sequences and corrosponing scores
        complete_seqs = []
        complete_seqs_scores = []
        k = beam_size
        for t in range(self.max_len):
            dec_states, O_t, logit = self.model.step_decoding(
                dec_states, O_t, enc_outs, topk_ids.unsqueeze(1))
            log_probs = torch.log(logit)

            log_probs += topk_log_probs.unsqueeze(1)
            topk_log_probs, topk_ids = torch.topk(log_probs.view(-1), k)

            vocab_size = len(self._sign2id)
            beam_index = topk_ids // vocab_size
            topk_ids = topk_ids % vocab_size

            seqs = torch.cat([seqs.index_select(0, beam_index), topk_ids],
                             dim=1)

            complete_inds = [
                ind for ind, next_word in enumerate(topk_ids)
                if next_word == END_TOKEN
            ]
            incomplete_inds = list(
                set(range(len(topk_ids))) - set(complete_inds))
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(topk_log_probs[complete_inds])
            k -= len(complete_inds)
            if k == 0:  # all beam finished
                break

            # prepare for next step
            seqs = seqs[incomplete_inds]
            topk_ids = topk_ids[incomplete_inds]
            topk_log_probs = topk_log_probs[incomplete_inds]

            enc_outs = enc_outs[:k]
            O_t = O_t[beam_index[incomplete_inds]]
            dec_states = (dec_states[0][beam_index[incomplete_inds]],
                          dec_states[1][beam_index[incomplete_inds]])

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        result = self._idx2formulas(seq)[0]

        return result
