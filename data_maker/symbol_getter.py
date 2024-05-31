from pathlib import Path

input_dir = Path("data/sentence_labels") 
 
symbols = ''''''
symbols_list = []
letters = 'abcçdeəfgğhxıijkqlmnoöprsştuüvwyzABCÇDEƏFGĞHXIİJKQLMNOÖPRSŞTUÜVWYZ0123456789'
for txt_file in input_dir.glob("*.txt"):
    with txt_file.open('r', encoding='utf-8') as file:
        content = file.read().strip()
        for j,i in enumerate(content):
            if i not in letters:
                if i not in symbols:
                    print(j,i, txt_file)
                    symbols+=i
                    symbols_list.append(i)
print(symbols)
print(symbols_list)