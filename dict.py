from util import *

char_list = load_char_list('字.txt')

dir = 'ctext - 副本 - 副本'
all_new_chars = set()
with open('new_chars.txt', 'w', encoding='utf-8') as file:
    for f in os.listdir(dir):
        if f.endswith('.txt'):
            new_c = process_text(os.path.join(dir, f), char_list)
            if len(new_c):
                all_new_chars.update(new_c)
                print(f'\n{f[:-4]}中{len(new_c)}个字字音不明')
                print(''.join(sorted(new_c)))
                file.write(f'\n{f[:-4]}中{len(new_c)}个字字音不明\n')
                file.write(''.join(sorted(new_c)) + '\n')

    print(f'\n共{len(all_new_chars)}个')
    file.write(f'\n共{len(all_new_chars)}个\n')
    all_new_chars = ''.join(sorted(all_new_chars))
    print(all_new_chars)
    file.write(all_new_chars)
