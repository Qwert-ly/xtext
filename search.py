from util import *

texts_dir = 'ctext - all - slice'
df = pd.read_excel('工作簿1.xlsx')
idx = create_idx(texts_dir, 'index-all.json', load=True)

while True:
    char = input("请输入要查找的字符 (打 'q' 退出): ")
    if char.lower() == 'q':
        break

    display_results(idx, char, texts_dir, df=df)

