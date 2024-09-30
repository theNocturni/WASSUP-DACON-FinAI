'''
BERT 기반의 데이터 증강 기법을 제공하는 클래스
문장에서 무작위로 단어를 마스킹하거나 삽입하여 텍스트 데이터를 증강하는 기능
'''

import transformers
import re
import random
import numpy as np


class BERT_Augmentation():
    def __init__(self):
        self.model_name = 'monologg/koelectra-base-v3-generator' # ELECTRA는 BERT와 유사하지만, 특정한 방식으로 사전 훈련이 이루어지는 모델
        self.model = transformers.AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        self.unmasker = transformers.pipeline("fill-mask", model=self.model, tokenizer=self.tokenizer)
        random.seed(42)

    def random_masking_replacement(self, sentence, ratio=0.15):
        """문장의 임의의 어절을 마스킹하고 사전학습모델을 사용하여 복구합니다.
        문장 내의 일부 단어를 무작위로 마스킹하여 모델이 이를 예측하도록 유도합니다.
        
        Args:
            sentence (str): Source sentence
            ratio (int): Ratio of masking

        Returns:
          str: Recovered sentence
        """
        
        span = int(round(len(sentence.split()) * ratio))
        
        # 품질 유지를 위해, 문장의 어절 수가 4 이하라면 원문장을 그대로 리턴합니다.
        if len(sentence.split()) <= 4:
            return sentence

        mask = self.tokenizer.mask_token
        unmasker = self.unmasker


        # 소수점 분리안되도록 DOT으로 변경
        unmask_sentence = re.sub(r'(?<=\d)\.(?=\d)', 'DOT', sentence)

        # 처음과 끝 부분을 [MASK]로 치환 후 추론할 때의 품질이 좋지 않음.
        random_idx = random.randint(1, len(unmask_sentence.split()) - span)

        unmask_sentence = unmask_sentence.split()
        # del unmask_sentence[random_idx:random_idx+span]
        cache = []
        for _ in range(span):
            # 처음과 끝 부분을 [MASK]로 치환 후 추론할 때의 품질이 좋지 않음.
            while cache and random_idx in cache:
                random_idx = random.randint(1, len(unmask_sentence) - 2)
            cache.append(random_idx)
            unmask_sentence[random_idx] = mask
            unmask_sentence = unmasker(" ".join(unmask_sentence))[0]['sequence']
            unmask_sentence = unmask_sentence.split()
        unmask_sentence = " ".join(unmask_sentence)
        unmask_sentence = unmask_sentence.replace("  ", " ")

        # DOT을 소수점으로 변경
        unmask_sentence = unmask_sentence.replace('DOT', '.') 
        return unmask_sentence.strip()

    def random_masking_insertion(self, sentence, ratio=0.15):
        '''
        문장 내에 임의의 위치에 [MASK]를 삽입하고, 모델을 통해 이 [MASK]를 적절한 단어로 채웁니다. 

        문장에 추가적인 [MASK]를 삽입하여 새로운 문장을 생성합니다.
        '''
        
        span = int(round(len(sentence.split()) * ratio))
        mask = self.tokenizer.mask_token
        unmasker = self.unmasker
        
        # Recover
        unmask_sentence = re.sub(r'(?<=\d)\.(?=\d)', 'DOT', sentence)
        
        for _ in range(span):
            unmask_sentence = unmask_sentence.split()
            random_idx = random.randint(0, len(unmask_sentence)-1)
            unmask_sentence.insert(random_idx, mask)
            unmask_sentence = unmasker(" ".join(unmask_sentence))[0]['sequence']

        unmask_sentence = unmask_sentence.replace("  ", " ")
        # DOT을 소수점으로 변경
        unmask_sentence = unmask_sentence.replace('DOT', '.') 

        return unmask_sentence.strip()

if __name__ == "__main__":
    # BERT_Augmentation 클래스 인스턴스 생성
    augmenter = BERT_Augmentation()

    # 원문 예시
    original_sentence = "2024년 중앙정부의 예산 지출은 일반회계 356.5조원, 21개 특별회계 81.7조원으로 구성되어 있습니다."
    dot_sen = augmenter.random_masking_insertion(original_sentence, ratio=0.15)

    # 결과 출력
    print("원문 :       ", original_sentence)
    print("변형된 문장: ", dot_sen)
    


