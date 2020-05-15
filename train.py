#!/usr/bin/python

import sys, getopt
import pandas as pd
import numpy as np
from tqdm import tqdm
import os 

# to visualise the performance of the model
import matplotlib.pyplot as plt 
from matplotlib.ticker import MaxNLocator

from sklearn.model_selection import train_test_split
from BERTSeqClassifier import BERTSequenceModel

# for sequence classification
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn as nn
import torch.optim as optim

# for convolution functions
import torch.nn.functional as F


def plot_log_history(nb_epochs, train_loss_history, test_loss_history):

	if not os.path.exists("training_log"):
		os.makedirs("training_log")

	ax = plt.figure().gca()
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	plt.plot(range(nb_epochs), train_loss_history, 'b')
	plt.plot(range(nb_epochs), test_loss_history, 'r')
	plt.legend(['train', 'val'])
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	ax.set_ylim([0,None])
	plt.grid()
	plt.savefig('training_log/' + str(nb_epochs)+"epochs.png")

def preprocessAndSplitData(bert):

	# append the title to the text
	news_df = pd.read_csv('data/fake-and-real-news-dataset/combined.csv')
	news_df["text"] = news_df["title"] + ": " + news_df["text"] 
	news_df["fake"] = news_df["label"].apply(lambda x: True if x == 'real' else False)

	# embed the data acording to BERTTokenizer
	nb_samples = 25
	tensor_list = bert.preprocess_text_samples(news_df[:nb_samples])
	tensor_labels = news_df.fake[:nb_samples].apply(lambda x: torch.tensor([x]).long().to(bert.device)).to_list()

	# split the data
	X_train, X_temp, y_train, y_temp = train_test_split(tensor_list, tensor_labels, test_size=0.4, 
                                                random_state=1)
	X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=1)

	# save to folder
	if not os.path.exists("data"):
	    os.makedirs("data")

	torch.save(X_train, 'data/X_train.pt')
	torch.save(y_train, 'data/y_train.pt')
	torch.save(X_val, 'data/X_val.pt')
	torch.save(y_val, 'data/y_val.pt')
	torch.save(X_test, 'data/X_test.pt')
	torch.save(y_test, 'data/y_test.pt')


if __name__ == "__main__":

	# valid input arguments
	validArgs = ["nbEpochs=", "trainOnly="]

	try:
		# parse command line options
		opts, args = getopt.getopt(sys.argv[1:], '', validArgs)
		assert len(opts) == len(validArgs)
	except getopt.GetoptError:
		print("The input should have the following structure: python train.py --nbEpochs <nbEpochs> --trainOnly <trainOnly>")
		raise
	except AssertionError:
		raise AssertionError("The number of argments is incorrect, the input should have the following structure: python train.py --nbEpochs <nbEpochs> --trainOnly <trainOnly>")

	nbEpochs = 0
	trainOnly = True

	# iterate through the input arguments
	for opt, arg in opts:
		if opt == "--nbEpochs":
			try:
				nbEpochs = int(arg)
			except ValueError:
				print("<nbEpochs> should be an integer")
		else:
			try: 
				trainOnly = bool(arg)
			except ValueError:
				print("<trainOnly> should be a boolean")

	bert = BERTSequenceModel()

	# if we want to encode the datapoints according to BERT and split the data in addition to training the data
	if not trainOnly:
		preprocessAndSplitData(bert)
	
	X_train = torch.load('data/X_train.pt')
	y_train = torch.load('data/y_train.pt')
	X_val = torch.load('data/X_val.pt')
	y_val = torch.load('data/y_val.pt')

	# train the model and save a plot of the performance of the model
	train_loss_history, test_loss_history = bert.train_model(X_train, y_train, X_val, y_val, nb_epochs = nbEpochs, log_freq = 5)
	plot_log_history(nbEpochs, train_loss_history, test_loss_history)
