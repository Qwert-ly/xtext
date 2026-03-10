import pandas as pd


def parse(ipa):
    声 = 'K'
    元 = 'A'
    尾 = ''
    ipa = ipa.rstrip('ʔh')
    if 'e' in ipa:
        元 = 'E'
    elif 'i'in ipa:
        元 = 'I'
    elif 'ə' in ipa:
        元 = 'Y'
    elif 'u' in ipa:
        元 = 'U'
    elif 'o' in ipa:
        元 = 'O'

    if ipa.endswith(('wk', 'w')):
        尾 = 'W'
    elif ipa.endswith(('ps', 'p')):
        尾 = 'P'
    elif ipa.endswith('ŋ'):
        尾 = 'NG'
    elif ipa.endswith('k'):
        尾 = 'K'
    elif ipa.endswith('n'):
        尾 = 'N'
    elif ipa.endswith('r'):
        尾 = 'R'
    elif ipa.endswith(('s', 't')):
        尾 = 'T'
    elif ipa.endswith('j'):
        尾 = 'J'
    elif ipa.endswith('m'):
        尾 = 'M'

    if ipa.startswith(('l', 'sl', 'l̥', 'ml')):
        声 = 'L'
    elif ipa.startswith(('j', 'j̊', 'kj', 'kʰj', 'mj', 'sj', 'hj')):
        声 = 'J'
    elif ipa.startswith(('p', 'b')):
        声 = 'P'
    elif ipa.startswith(('m', 'm̥')):
        声 = 'M'
    elif ipa.startswith(('ts', 'dz', 's')):
        声 = 'TS'
    elif ipa.startswith(('t', 'd', 'st')):
        声 = 'T'
    elif ipa.startswith(('n', 'n̥')):
        声 = 'N'
    elif ipa.startswith(('r', 'r̥', 'C.r')):
        声 = 'R'
    elif ipa.startswith(('ŋ', 'ŋ̊')):
        声 = 'NG'
    elif ipa.startswith('h'):
        声 = 'H'
    elif ipa.startswith(('w', 'ẘ', 'kw', 'kʰw', 'gw', 'ŋw', 'ʔw')):
        声 = 'W'
    elif ipa.startswith('ʔ'):
        声 = 'Q'
    return 声+元+尾


df_d = pd.read_excel('上古汉语音节表.xlsx', sheet_name='字典表', keep_default_na=False, engine='openpyxl')
for idx, row in df_d.iterrows():
    ipa = str(row.get('音', '')).strip()
    xsy = parse(ipa)
    df_d.at[idx, '諧聲域'] = xsy

df_d.to_excel('上古汉语音节表_輸出.xlsx', sheet_name='字典表', index=False, engine='openpyxl')
