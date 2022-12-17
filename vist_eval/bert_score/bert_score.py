from bert_score import score
import numpy as np

class Bert_score:
    def __init__(self):
        self._hypo_for_image = {}
        self.ref_for_image = {}

    def compute_score(self, gts, res):
        s1 = sorted(gts.keys())
        s2 = sorted(res.keys())
        for i in range(len(s1)):
            if s1[i] != s2[i]:
                print(s1[i], s2[i])
        # assert(gts.keys() == res.keys())
        assert(sorted(gts.keys()) == sorted(res.keys()))
        imgIds = gts.keys()

        all_hypo = []
        all_ref = []

        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) >= 1)

            # bert_score_scorer += (hypo[0], ref)
            hypo *= len(ref)
            all_hypo += hypo
            all_ref += ref
            assert len(all_hypo) == len(all_ref)

        all_f1 = np.zeros((1, 5))
        # imgId_score = score(hypo, ref, lang='en', verbose=False)
        batch = 100
        for i in range(len(all_hypo) // batch + 1):
            h = all_hypo[batch * i : batch * (i + 1)]
            r = all_ref[batch * i : batch * (i + 1)]
            if len(h) == 0:
                continue
            imgId_score = score(h, r, lang='en', verbose=False)
            assert len(imgId_score) == 3 # precision, recall, f1
            f1 = np.array([np.array(i) for i in imgId_score])[2]
            f1 = f1.reshape(-1, 5)
            all_f1 = np.concatenate((all_f1, f1), axis=0)
        all_f1 = all_f1[1:]
        per_album_f1 = np.mean(all_f1, axis=1)
        overall_f1 = np.mean(per_album_f1)

        return overall_f1, per_album_f1

    def method(self):
        return "Bert_score"
