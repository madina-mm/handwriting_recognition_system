import os


def make_letter_dirs():
    # letters = 'abcçdeəfgğhxıijkqlmnoöprsştuüvwyzABCÇDEƏFGĞHXIİJKQLMNOÖPRSŞTUÜVWYZ'

    letters_dir = 'data\\letters'
    letters = 'abcçdeəfgğhxıijkqlmnoöprsştuüvwyz'
    # uppercase_letters = 'ABCÇDEƏFGĞHXIİJKQLMNOÖPRSŞTUÜVWYZ'

    if not os.path.exists(letters_dir):
        os.makedirs(letters_dir)

    for letter in letters:
        save_dir = os.path.join(letters_dir, letter)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_dir_uppercase = os.path.join(letters_dir, f'{letter.upper()}_up')
        if not os.path.exists(save_dir_uppercase):
            os.makedirs(save_dir_uppercase)