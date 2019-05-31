import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForPreTraining, BertAdam
import random, os
from tqdm import tqdm
from read import readPairs

def get_tokenizer(modelpath):
	tokenizer = BertTokenizer.from_pretrained(modelpath)
	return tokenizer

def get_pretrain_model(modelpath):
	model = BertForMaskedLM.from_pretrained(modelpath)
	return model

def transfer(pair, modelpath):
	tokenizer = get_tokenizer(modelpath)
	input_sentence, label_sentence = pair[0], pair[1]
	input_text, label_text = tokenizer.tokenize(input_sentence), tokenizer.tokenize(label_sentence)

	for _ in label_text:
		input_text.append("[MASK]")
	for _ in range(128-len(input_text)):
		input_text.append("[MASK]")
	input_tokens = tokenizer.convert_tokens_to_ids(input_text)
	input_tensor = torch.tensor([input_tokens])

	label_ids = [-1] * len(tokenizer.tokenize(input_sentence))
	label_ids += tokenizer.convert_tokens_to_ids(label_text)
	label_ids.append(tokenizer.convert_tokens_to_ids(['[SEP]'])[0])
	for _ in range(128-len(label_ids)):
		label_ids.append(-1)
	label_tensors = torch.tensor([label_ids])
	return [input_tensor, label_tensors]

def process_pairs(pairs, modelpath):
	tensor_pairs = []
	for pair in pairs:
		tensor_pair = transfer(pair, modelpath)
		tensor_pairs.append(tensor_pair)
	return tensor_pairs

def train(tensor_pairs, modelpath, directory, batch_size):
	model = get_pretrain_model(modelpath)
	optimizer = torch.optim.Adamax(model.parameters(), lr = 5e-5)
	model.train()

	for i in tqdm(range(1, 151)):
		# tensor_batch = random.sample(tensors, batch_size)
		# input_batch = [tensor[0] for tensor in tensor_batch]
		# label_batch = [tensor[1] for tensor in tensor_batch]
		# input_token = torch.cat(input_batch, 0)
		# label_token = torch.cat(label_batch, 0)
		input_tensor, label_tensor = tensor_pairs[0][0], tensor_pairs[0][1]

		loss = model(input_tensor, masked_lm_labels=label_tensor)
		eveloss = loss.mean().item()
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print("step "+ str(i) + " : " + str(eveloss))

		if i % 50 == 0:
			torch.save(model, os.path.join(directory, '{}_{}_backup.tar'.format(i, batch_size)))

def infer(file, modelpath):
	tokenizer = get_tokenizer(modelpath)
	model = torch.load(file)
	model.eval()
	while(1):
		try:
			print("Please input :")
			question = input("> ")
			if question == 'q': break
			question = '[CLS] ' + question + ' [SEP] '
			print("question : ", question)
			tokenized_text = tokenizer.tokenize(question)
			print(tokenized_text)
			for _ in range(128-len(tokenized_text)):
				tokenized_text.append("[MASK]")
			indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
			tokens_tensor = torch.tensor([indexed_tokens])
			print(tokens_tensor)

			answer = []
			with torch.no_grad():
				predictions = model(tokens_tensor)
				start = len(tokenizer.tokenize(question))
				while start < len(predictions[0]):
					predicted_index = torch.argmax(predictions[0, start]).item()
					print(predicted_index)
					predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
					print(predicted_token)
					answer += predicted_token
					if "[SEP]" in predicted_token:
						break
					start += 1
				result = " ".join(answer)
				print(result)
		except KeyError:
			print("Error !")


if __name__ == "__main__":
	# corpus = "sentences.txt"
	# pairs = readPairs(corpus)
	# print(pairs[0])

	# input_sentence, label_sentence = pairs[0][0], pairs[0][1]
	# print(input_sentence)
	# print(label_sentence)

	# modelpath = "bert-base-uncased"
	# # tokenizer = get_tokenizer(modelpath)
	# # label_text = tokenizer.tokenize(label_sentence)
	# # print(label_text)
	# print("Transfer to Tensor...")
	# tensor_pair = process_pairs(pairs[0], modelpath)
	# print(tensor_pair)

	# batch_size = 1
	# directory = "model/"
	# print("Start Training...")
	# train(tensor_pair, modelpath, directory, batch_size)

	file = "model/150_1_backup.tar"
	modelpath = "bert-base-uncased"
	infer(file, modelpath)
