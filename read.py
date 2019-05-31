# coding:utf-8

# @Created : Macielyoung
# @Time : 2019/5/21
# @Function : use bert for text generation

import re
import os
import unicodedata

# Turn a Unicode string to plain ASCII
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
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def addSep(s):
	return "[CLS] " + s + " [SEP] "

def get_length(line):
	return len(line.split(" "))

def trimm(pairs):
	print("Trim Pairs...")
	count = 0
	new_pairs = []
	pair_length = []
	for pair in pairs:
		ques_len, answ_len = get_length(pair[0]), get_length(pair[1])
		if ques_len + answ_len > 120:
			continue
		else:
			new_pairs.append(pair)
			pair_length.append([ques_len, answ_len])
			count += 1
	print("Trimmed {} Sentence Pair...".format(count))
	return new_pairs

def readPairs(corpus):
	print("Reading lines...")

	# combine every two lines into pairs and normalize
	with open(corpus, encoding='utf-8') as f:
		content = f.readlines()
	lines = [x.lower().strip() for x in content]
	it = iter(lines)
	# pairs = [[normalizeString(x), normalizeString(next(it))] for x in it]
	pairs = [[addSep(x), next(it)] for x in it]
	print("Read {} Sentence Pair...".format(len(pairs)))
	pairs = trimm(pairs)
	return pairs

if __name__ == "__main__":
	corpus = "dialogue.txt"
	pairs = readPairs(corpus)
	#print(pairs)
