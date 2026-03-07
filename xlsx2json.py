import pandas as pd
import re


def parse_def(text):
    text = str(text).strip()
    if not text:
        return []

    nums = list(re.finditer(r'\d+', text))
    split = []
    e = 1
    found_first = False

    end = set('。！？”」》】；： \n\t\r　')

    for n in nums:
        val = int(n.group())
        s = n.start()

        if not found_first:
            if val == 1:
                if s == 0 or text[s - 1] in end:
                    split.append(n)
                    e = 2
                    found_first = True
        else:
            if e - 1 <= val <= e + 1:
                if s > 0 and text[s - 1] in end:
                    split.append(n)
                    e = val + 1

    if not split:
        return {"說明": "", "義項": [text]}

    result = {"說明": "", "義項": []}
    s_ = split[0].start()
    if s_ > 0:
        meta = text[:s_].strip()
        if meta:
            result["說明"] = meta

    for i in range(len(split)):
        start_i = split[i].end()
        if i + 1 < len(split):
            end_i = split[i + 1].start()
        else:
            end_i = len(text)

        part = text[start_i:end_i].strip()
        if part:
            result["義項"].append(part)
    return result


def parse_xiaoyun_chars(text):
    text = str(text).strip()
    if not text:
        return []

    result = []
    # (.) 捕获任意单个字符
    # (?:\{([^}]+)\})? 匹配可选的 {...}，并捕获其中的注释内容
    for m in re.finditer(r'(.)(?:\{([^}]+)\})?', text):
        char = m.group(1)
        note = m.group(2) if m.group(2) else ""
        result.append({
            "字": char,
            "注釋": note
        })

    return result


def get_note(row):
    char = str(row.get('字')).strip()
    ipa = str(row.get('音', '')).strip()
    return notes.get((char, ipa), "")


df_d = pd.read_excel('上古汉语音节表.xlsx', sheet_name='字典表', keep_default_na=False, engine='openpyxl')
df_x = pd.read_excel('上古汉语音节表.xlsx', sheet_name='小韻表', keep_default_na=False, engine='openpyxl')


notes = {}
for _, row in df_x.iterrows():
    ipa = str(row.get('IPA', '')).strip()
    parsed_c = parse_xiaoyun_chars(row.get('字', ''))
    for i in parsed_c:
        if i["注釋"]:
            notes[(i["字"], ipa)] = i["注釋"]


df_d['小韻表注釋'] = df_d.apply(get_note, axis=1)
df_d['釋義'] = df_d['釋義'].apply(parse_def)

df_d.to_json('上古汉语音节表.json', orient='records', force_ascii=False, indent=4)
