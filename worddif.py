from util import *


def compare_text_vectors(text1_name, text2_name, texts, all_chars):
    def nand(d, k):
        try:
            r = d[k]
        except:
            return 0
        return r

    if text1_name not in texts or text2_name not in texts:
        raise ValueError('文本不在texts中')

    char_freqs = {text1_name: char_freq(texts[text1_name]),
                  text2_name: char_freq(texts[text2_name])}

    all_chars_list = sorted(list(all_chars))
    vec_d = t2v(text1_name, all_chars_list, char_freqs) - t2v(text2_name, all_chars_list, char_freqs)

    word_d = {}
    char_c = {text1_name: {c: count for c, count in Counter(texts[text1_name]).items()},
              text2_name: {c: count for c, count in Counter(texts[text2_name]).items()}}
    for c, diff in zip(all_chars_list, vec_d):
        if diff != 0:
            c1 = nand(char_c[text1_name], c)
            c2 = nand(char_c[text2_name], c)
            word_d[c] = (c1-c2, diff)

    return vec_d, dict(sorted(word_d.items(), key=lambda i: i[1][1], reverse=True))


def compare_texts(text1, text2, texts, all_chars):
    word_d = compare_text_vectors(text1, text2, texts, all_chars)[1]

    for word, diff in word_d.items():
        print(f'{word}: {diff}')


if __name__ == '__main__':
    texts, all_chars = read_files('ctext - all - slice')
    while True:
        compare_texts(input('t1'), input('t2'), texts, all_chars)
