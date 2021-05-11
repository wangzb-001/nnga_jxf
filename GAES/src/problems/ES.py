from src.ES_predict import search
from src.util.ga_utils import word2idx, words
from src.util.redis_manage import RedisManage
import numpy as np
from src.logers import LOGS


class ES:
    valuable_num = len(word2idx)
    iterator_num = 2
    Bound = [0, 1]
    accuracy = 0.01
    golbal_fv = 0
    word = []

    def __init__(self):
        pass

    def pop_to_words(self, pop: list):
        pop = [1 if p > 0.5 else 0 for p in pop]
        all_w = [words[i] for i, p in enumerate(pop) if p == 1]
        new_w = [words[i] for i, p in enumerate(pop) if p == 1 and words[i] not in self.word]
        not_w = [w for w in self.word if w not in words or w in all_w]
        # not_w2 = [w for w in self.word if w not in words or w not in all_w]
        pop_words = new_w + not_w
        return pop_words

    def Func(self, pops: list, subject_id: str):
        if not isinstance(pops[0], list):
            pops = [1 if p > 0.5 else 0 for p in pops]
            if sum(pops) == 0: return 10 ** 4
            pops = [pops]
        # 以es检索召回的同类分数和的倒数*100，也就是趋于0越好
        all_words = [self.pop_to_words(p) for p in pops]
        all_sentences = [','.join(w) for w in all_words]
        sentences_score_dict = {}
        for si in all_sentences:
            # if si in sentences_score_dict:
            #     print(1)
            sentences_score_dict[si] = 0

        search_sentences = []
        for sent in sentences_score_dict:
            v = RedisManage.get(sent)
            if v:
                if not isinstance(v, float):
                    v = float(v)
                sentences_score_dict[sent] = v
            else:
                if sent.strip() == '':
                    sentences_score_dict[sent] = 0
                else:
                    search_sentences.append(sent)
        scores = search(search_sentences, [subject_id] * len(search_sentences))
        for i, sent in enumerate(search_sentences):
            if i >= len(scores):
                LOGS.log.debug(f'{i},{scores},{search_sentences}')
            try:
                sentences_score_dict[sent] = scores[i]
            except Exception as e:
                print(e)
            RedisManage.set(sent, scores[i])

        pops_score = []
        for s in all_sentences:
            score = sentences_score_dict[s]
            pops_score.append(score)
        fv = [1000 / (ps + 0.1) for ps in pops_score]
        return np.array(fv)
