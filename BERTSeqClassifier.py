#!/usr/bin/python

import pandas as pd
import numpy as np
from tqdm import tqdm
import os 

# to visualise the performance of the model
import matplotlib.pyplot as plt 
from matplotlib.ticker import MaxNLocator

# for sequence classification
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn as nn
import torch.optim as optim

# for convolution functions
import torch.nn.functional as F


class BERTSequenceModel():
    
    def __init__(self, pretrained_name = "bert-base-uncased"):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.criterion = None
        self.optimizer = None
        
        # https://huggingface.co/transformers/model_doc/bert.html#berttokenizer
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_name)
        
        # https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification
        self.model = BertForSequenceClassification.from_pretrained(pretrained_name)
        self.model.config.num_labels = 1
        
        # Freeze the pre trained parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        layers = [nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)]
        
        self.addLayers(layers)
        
    def addLayers(self, layers):
        modules = []
        
        for layer in layers:
            modules.append(layer)
        
        self.model.classifier = nn.Sequential(*modules)
        self.model = self.model.to(self.device)
        
    def preprocess_text_samples(self, samples, max_seq_length = 300):
    
        '''
        Adapted from https://www.kaggle.com/clmentbisaillon/classifying-fake-news-with-bert/notebook
        '''

        encoded_samples = []
        
        for idx, sample in tqdm(samples.iterrows(), total = samples.shape[0]):
            encoded_text = []
            words = sample.text.strip().split(' ')
            nb_seqs = int(len(words)/max_seq_length)

            for i in range(nb_seqs+1):
                words_part = ' '.join(words[i*max_seq_length : (i+1)*max_seq_length])

                try:
                    # https://huggingface.co/transformers/main_classes/tokenizer.html#pretrainedtokenizer
                    # encoding using BERT pretrained tokeinizer and converts to pytorch tensors
                    encoded_text.append(self.tokenizer.encode(words_part, return_tensors="pt", 
                                                         max_length = 500, device = self.device))
                except:
                    print("Issue at: " +str(idx))
                    raise

            encoded_samples.append(encoded_text)

        return encoded_samples
    
    def train_model(self, X_train, y_train, X_val, y_val, nb_epochs = 10, log_freq = 500):
        
        train_loss_history = []
        test_loss_history = []

        criterion = nn.MSELoss().to(self.device)
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

        print("------- Training started -------\n")
        for epoch in range(nb_epochs):

            train_loss = 0.0
            train_accuracy = 0.0
            test_accuracy = 0.0

            '''
            Iteration through training set
            '''
            self.model.train()

            # iterate through the datapoints
            for idx, text_tensor in enumerate(X_train):
                # set gradients of all optimizers to zero -> avoid accumulation
                self.model.zero_grad()

                # define a tensor for the output 
                output = torch.zeros((1, 2)).float().to(self.device)

                # iterate through each part of the text (each part is represented by a tensor)
                # and obtain the average of the outputs
                for i in range(len(text_tensor)):
                    input = text_tensor[i]
                    output += self.model(input, labels = y_train[idx])[1].float().to(self.device)

                output = F.softmax(output[0], dim=-1)

                # determine loss and accuracy
                label = torch.tensor([1.0, 0.0]).float().to(self.device) if y_train[idx] == 0 else torch.tensor([0.0, 1.0]).float().to(self.device)
                loss = criterion(output, label)
                train_loss += loss.item()

                if label.max(0)[1] == output.max(0)[1]:
                    train_accuracy += 1.0/len(X_train)

                # backpropagate
                loss.backward()
                optimizer.step()

                if log_freq and idx>0 and idx%log_freq == 0:
                    print("Trained {}/{}: avg loss = {:.2f}".format(idx, len(X_train), train_loss/log_freq))
            
            # iterate through test set
            test_loss, test_accuracy = self.test_model(X_val, y_val, criterion)
            
            print(">>>>>>> Epoch ({}/{}): train accuracy = {:.2f}%, test accuracy = {:.2f}%\n".format(epoch+1, nb_epochs, 
                                                                                                      train_accuracy*100, test_accuracy*100))
            train_loss_history.append(train_loss)
            test_loss_history.append(test_loss)

        #save weights after training
        print("------- End of training, saving weights -------\n")

        if not os.path.exists("weights"):
            os.makedirs("weights/")

        torch.save(self.model.state_dict(), "weights/model_" + str(nb_epochs) + "epochs_" 
                   + str(np.round(train_accuracy*100, 2)) + "train_" + str(np.round(test_accuracy*100, 2)) + "test" + ".pt")
        
        return train_loss_history, test_loss_history
            
    def test_model(self, X_test, y_test, criterion):
        
        '''
        Iteration through test set
        '''

        # set layers to test/evaluation mode
        self.model.eval()

        # set all grad flags to inactive
        with torch.no_grad():
            test_accuracy = 0.0
            test_loss = 0.0

            for idx, tensor in enumerate(X_test):

                # define a tensor for the output 
                output = torch.zeros((1, 2)).float().to(self.device)

                # iterate through each part of the text (each part is represented by a tensor)
                # and obtain the average of the outputs
                for text in tensor:
                    output += self.model(text)[0].float().to(self.device)

                output = F.softmax(output[0], dim=-1)

                # determine loss and accuracy
                label = torch.tensor([1.0, 0.0]).float().to(self.device) if y_test[idx] == 0 else torch.tensor([0.0, 1.0]).float().to(self.device)
                loss = criterion(output, label)
                test_loss += loss.item()

                if label.max(0)[1] == output.max(0)[1]:
                    test_accuracy += 1.0/len(X_test)
                    
        return test_loss, test_accuracy
    
    def predict():
        pass

