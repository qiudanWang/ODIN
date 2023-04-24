import torch
import torch.nn as nn
import argparse
from torch.autograd import Variable
import warnings
warnings.filterwarnings('ignore')
import numpy as np



from utils import *
from config import *
from classifier import SentimentClassifier


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('review', type=str, nargs='+',help = 'A review to classify')
    parsed_arguments = parser.parse_args()
    return parsed_arguments


def process(review, vocab):
    review_processed =  review.strip().lower().replace("<br />", " ").translate(str.maketrans('', '', string.punctuation)).split(" ")
    return torch.LongTensor(vocab.vocab.lookup_indices(review_processed))

def predict(model, review, vocab, temper=pow(10, 0), noiseMagnitude=0.0014):
    review_processed = process(review, vocab)
    review_processed = review_processed.unsqueeze(0)
    model.eval()
    # model forward
    review_processed = review_processed.cpu()
    x = model.emblayer(review_processed)
    x.retain_grad()
    h0 = torch.zeros(1, x.size(0), model.config.HIDDEN_SIZE)
    c0 = torch.zeros(1, x.size(0), model.config.HIDDEN_SIZE)
    y, _ = model.lstmlayer(x, (h0, c0))
    y = y[:,-1,:]
    y = model.linear1(y)
    y = model.relu(y)
    # Using temperature scaling
    logits = model.linear2(y)
    logits = logits / temper
    maxIndexTemp = torch.argmax(logits, dim=1).item()
    labels = Variable(torch.LongTensor([maxIndexTemp]))
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)
    loss.backward()
    # Calculating the perturbation we need to add, that is,
    gradient =  torch.ge(x.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2
    x = torch.add(x.data,  -noiseMagnitude, gradient)
    h0 = torch.zeros(1, x.size(0), model.config.HIDDEN_SIZE)
    c0 = torch.zeros(1, x.size(0), model.config.HIDDEN_SIZE)
    y, _ = model.lstmlayer(x, (h0, c0))
    y = y[:,-1,:]
    y = model.linear1(y)
    y = model.relu(y)
    logits = model.linear2(y)
    logits = logits / temper
    logits = logits.detach().numpy()
    nnOutputs = np.exp(logits)/np.sum(np.exp(logits))
    print(str(temper) + ", " + str(noiseMagnitude) + ", " + str(np.max(nnOutputs)))



if __name__ == '__main__':
    
    args = parse_command_line_arguments()
    review = ' '.join([str(elem) for elem in args.review])
    
    config = Config()
    
    vocab = load_vocab()

    model = SentimentClassifier().cpu()
    model.load_state_dict(torch.load(f"saved/IMDBmodel_best.pth", map_location=torch.device('cpu')))
    optimizer = torch.optim.Adam(model.parameters(),lr= config.L_RATE)
    
    predict(model, review, vocab)
    


