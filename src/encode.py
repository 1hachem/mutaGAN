# encoding sequences

def get_vocab(sequences):

    vocab = set()
    for record in sequences:
        vocab.update(str(record.seq))
    vocab.add("<pad>"),vocab.add("<sos>"), vocab.add("<eos>")
    
    to_ix = {char: i for i, char in enumerate(vocab)}

    return to_ix

to_ix = {
 'V': 0,
 'I': 1,
 'G': 2,
 '<sos>': 3,
 'D': 4,
 'C': 5,
 'S': 6,
 'B': 7,
 'Z': 8,
 'N': 9,
 'F': 10,
 'A': 11,
 'M': 12,
 'E': 13,
 'J': 14,
 'P': 15,
 'T': 16,
 'Q': 17,
 'K': 18,
 'H': 19,
 'W': 20,
 '*': 21,
 '<eos>': 22,
 '<pad>': 23,
 'X': 24,
 'R': 25,
 'Y': 26,
 'L': 27
 }

inv_to_ix = {0: 'V',
  1: 'I',
  2: 'G',
  3: '<sos>',
  4: 'D',
  5: 'C',
  6: 'S',
  7: 'B',
  8: 'Z',
  9: 'N',
  10: 'F',
  11: 'A',
  12: 'M',
  13: 'E',
  14: 'J',
  15: 'P',
  16: 'T',
  17: 'Q',
  18: 'K',
  19: 'H',
  20: 'W',
  21: '*',
  22: '<eos>',
  23: '<pad>',
  24: 'X',
  25: 'R',
  26: 'Y',
  27: 'L'}