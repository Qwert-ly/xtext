import pandas as pd
import re, json, gzip, shutil, msgpack


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
        return ["", [text]]

    result_meta = ""
    result = []

    s_ = split[0].start()
    if s_ > 0:
        meta = text[:s_].strip()
        if meta:
            result_meta = meta

    for i in range(len(split)):
        start_i = split[i].end()
        if i + 1 < len(split):
            end_i = split[i + 1].start()
        else:
            end_i = len(text)

        part = text[start_i:end_i].strip()
        if part:
            result.append(part)

    return [result_meta, result]


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

for col in ['先秦字頻（歸一化）', '西周字頻（歸一化）', '小韻表注釋']:
    if col in df_d.columns:
        df_d = df_d.drop(columns=[col])

records = df_d.to_dict(orient='records')

# 剔除空值
cleaned_records = []
for r in records:
    cleaned_r = {}
    for k, v in r.items():
        if v != "" and v is not None and v != []:
            if k == '釋義' and isinstance(v, list) and len(v) == 2:
                if not v[0] and not v[1]:
                    continue # 说明和义项都空，不保留
            cleaned_r[k] = v
    cleaned_records.append(cleaned_r)

with open('上古汉语音节表.json', 'w', encoding='utf-8') as f:
    json.dump(cleaned_records, f, ensure_ascii=False, separators=(',', ':'))

key_map = {
    "字": "z", "音": "y", "拼音": "p", "見詩經韻": "s", "見其他韻": "q",
    "總出現次數": "c", "少見詞出處": "r", "見西周": "x",
    "釋義": "d", "注釋": "n", "字統·字源諸說（zi.tools）": "e"
}

short_records = []
for r in cleaned_records:
    sr = {}
    for k, v in r.items():
        short_key = key_map.get(k, k)
        if v == "√":
            v = 1
        sr[short_key] = v
    short_records.append(sr)


with gzip.open('上古汉语音节表.json.gz', 'wb', compresslevel=9) as f_out:
    f_out.write(json.dumps(short_records, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))

# df_d.to_json('上古汉语音节表.json', orient='records', force_ascii=False)
