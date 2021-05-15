from __future__ import division, print_function, unicode_literals

import math
import pickle
import random
import re
import string
import time
import unicodedata
from io import open
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from name_generator.data.utils import (CharLang, Lang, normalizeString,
                                       prepareData, tensorFromSentence,
                                       tensorsFromPair)
from name_generator.models.encoder_decoder_model import (evaluate,
                                                         generate_name, train)
from name_generator.models.nn_models import AttnDecoderRNN, EncoderRNN
from name_generator.utils import timeSince
from torch import optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 21


def trainIters(
        encoder,
        decoder,
        n_iters,
        pairs,
        input_lang,
        char_lang,
        print_every=1000,
        plot_every=100,
        learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs),  input_lang, char_lang)
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            evaluateRandomly(encoder, decoder, pairs,
                             input_lang, char_lang, n=3)
            # generate_name(encoder, decoder, 'elevate human potential with machine learning')
            # generate_name(encoder1, attn_decoder1, 'elevate human potential with ai')

    showPlot(plot_losses)


plt.switch_backend('agg')


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def evaluateRandomly(encoder, decoder, pairs, input_lang, char_lang, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(
            encoder, decoder, pair[0], input_lang, char_lang)
        output_sentence = ''.join(output_words)
        print('<', output_sentence)
        print('')


# generate_name(encoder1, attn_decoder1, 'elevate human potential with ai')


def run(input_path, output_path):
    df = pd.read_csv(input_path / 'dataset.csv', index_col=0).iloc[:1000]
    df.name = df.name.apply(normalizeString)
    df.description = df.description.apply(normalizeString)
    df['name_short'] = df.name.apply(lambda x: ' '.join(x.split(' ')[:3]))
    df['description_short'] = df.description.apply(
        lambda x: ' '.join(x.split(' ')[:MAX_LENGTH - 1]))
    input_lang, char_lang, pairs = prepareData('eng', df)
    print(random.choice(pairs))
    # print(pd.DataFrame(input_lang.word2count.values()).describe())
    hidden_size = 128
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(
        hidden_size=hidden_size, output_size=char_lang.n_chars, dropout_p=0.1).to(device)

    trainIters(encoder1, attn_decoder1, 100, pairs,
               input_lang, char_lang, print_every=100)
    print('END OF TRAINING')
    evaluateRandomly(encoder1, attn_decoder1, pairs,
                     input_lang, char_lang, n=2)

    torch.save(encoder1, output_path / 'encoder.pth')
    torch.save(attn_decoder1, output_path / 'decoder.pth')

    with (output_path / 'input_lang.pkl').open(mode='wb') as f:
        pickle.dump(input_lang, f)

    with (output_path / 'char_lang.pkl').open(mode='wb') as f:
        pickle.dump(char_lang, f)


if __name__ == '__main__':
    input_path = Path('./data')
    output_path = Path('./data/output')
    output_path.mkdir(parents=True, exist_ok=True)
    run(input_path, output_path)
