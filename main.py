import argparse
from data import *
from evaluate import *
from models import *
import numpy as np
import time


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="TrivialNN", help="model to run")
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--hidden_size', type=int, default=100, help='hidden layer size')
    parser.add_argument('--batch_size', type=int, default=100, help='training batch size; 1 by default')
    
    parser.add_argument('--device', default="cuda:0")
    args = parser.parse_args()
    return args


def evaluate(classifier, X_all, y_all):
    X_all = torch.from_numpy(np.array(X_all)).float()
    y_all = torch.from_numpy(np.array(y_all)).float().view((-1, 1))
    
    y_pred = np.zeros(len(X_all))
    
    for idx, X in enumerate(X_all):
        y_pred[idx] = classifier.predict(X)
        
    return print_evaluation(y_all, y_pred)
        
if __name__ == '__main__':
    args = _parse_args()
    print(args)
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_datasets()
    print("%i train exs, %i dev exs, %i test exs" % (len(y_train), len(y_val), len(y_test)))
    
    # Train and evaluate
    start_time = time.time()
    if args.model == "TrivialNN":
        model = train_trivialnn(args, X_train, y_train, X_val, y_val)
    
    model.to("cpu")
    print("=====Train Accuracy=====")
    train_eval = evaluate(model, X_train, y_train)
    print(train_eval)
    
    print("=====Val Accuracy=====")
    val_eval = evaluate(model, X_val, y_val)
    print(val_eval)
    
    train_eval_time = time.time() - start_time
    print("Time for training and evaluation: %.2f seconds" % train_eval_time)
    
    # TODO: gpu, batch size