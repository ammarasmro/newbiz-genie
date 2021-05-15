from __future__ import division, print_function, unicode_literals

import re
import string
import unicodedata

import pandas as pd
import torch
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 21


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class CharLang:
    def __init__(self, name):
        self.name = name
        all_letters = string.ascii_letters + " .,;'-?!"
        n_letters = len(all_letters)
        self.char2index = {char: index + 2 for index,
                           char in enumerate(all_letters)}
        # self.char2count = {}
        self.index2char = {index + 2: char for index,
                           char in enumerate(all_letters)}
        # self.index2char = {0: "[", 1: "]"}
        self.index2char[0] = '['
        self.index2char[1] = ']'
        self.n_chars = n_letters + 2  # Count SOS and EOS


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang, df):
    reverse_array = df[['name_short', 'description_short']].to_numpy()
    # Split every line into pairs and normalize
    pairs = reverse_array[:, (1, 0)]
    input_lang = Lang(lang)
    char_lang = CharLang('charlang')

    return input_lang, char_lang, pairs


def prepareData(lang, df):
    input_lang, char_lang, pairs = readLangs(lang, df)
    print("Read %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    return input_lang, char_lang, pairs


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def charIndexesFromSentence(lang, sentence):
    return [lang.char2index[char] for char in sentence]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def charTensorFromSentence(lang, sentence):
    indexes = charIndexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair, input_lang, char_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = charTensorFromSentence(char_lang, pair[1])
    return (input_tensor, target_tensor)


def shorten_name_and_descriptions(df):
    df.name = df.name.apply(normalizeString)
    df.description = df.description.apply(normalizeString)
    df['name_short'] = df.name.apply(lambda x: ' '.join(x.split(' ')[:3]))
    df['description_short'] = df.description.apply(
        lambda x: ' '.join(x.split(' ')[:MAX_LENGTH - 1]))


def download_raw_data(url, output_path):
    pass


def raw_data_to_dataframe(raw_data_path):
    pass


class BusinessNamesDataset(Dataset):
    def __init__(self, data_path, transform=None, target_transform=None):
        super().__init__()
        self.data = pd.read_csv(data_path)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        business_name = self.data.iloc[idx]['name']
        business_description = self.data.iloc[idx]['description']
        sample = {
            'sample_id': idx,
            'business_name': business_name,
            'business_description': business_description
        }
        return sample
