from util import *

texts, _ = read_files('ctext - 副本 - 副本')

discoverer = WordDiscoverer(min_freq=10, min_pmi=2.5, min_entropy=1.8)
discoverer.fit(texts)
new_words_dict = discoverer.extract_words()

# 提取纯粹的词汇集合
vocab_set = set(new_words_dict.keys())
print(f"成功挖掘出 {len(vocab_set)} 个古文词汇！例如：{list(vocab_set)[:20]}")

segmenter_mm = MaxMatchSegmenter(vocab_set)
with open('hmm_train_corpus.txt', 'w', encoding='utf-8') as f:
    for content in texts.values():
        sentences = split_sentences(content)
        for sentence in sentences:
            words = segmenter_mm.cut_sentence(sentence)
            f.write(' '.join(words) + '\n')

print("机械分词完成，已生成 hmm_train_corpus.txt")

hmm = HMMSegmenter()
hmm.train('hmm_train_corpus.txt')
hmm.save('ancient_chinese_hmm.json')

test_text = "臣聞之也，君子坦蕩蕩，小人長戚戚"
print(f"HMM 测试结果: {hmm.cut(test_text)}")
