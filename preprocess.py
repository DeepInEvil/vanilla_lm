import data
import model

def get_batch(source, i):
    seq_len = min(30, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

corpus = data.Corpus('./data/wikitext-2')
train_data = corpus.valid
#
# for batch, i in enumerate(range(0, train_data.size(0) - 1, 30)):
#     data, targets = get_batch(train_data, i)

print (get_batch(train_data, 0))