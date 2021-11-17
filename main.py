import argparse
from data import *
from evaluate import *
from models import *
import numpy as np
import time


def _parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model', type=str, default="TrivialNN", help="model to run")
    # parser.add_argument('--model', type=str, default="SeveritySumNN", help="model to run")
    parser.add_argument('--model', type=str, default="SeverityIndNN", help="model to run")
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--hidden_size', type=int, default=100, help='hidden layer size')
    parser.add_argument('--batch_size', type=int, default=100, help='training batch size; 1 by default')
    
    parser.add_argument('--device', default="cuda:0")
    args = parser.parse_args()
    return args


def evaluate(args, classifier, X_all, y_all):
    X_all = torch.from_numpy(np.array(X_all)).float()
    if args.model == "TrivialNN":
        y_all = torch.from_numpy(np.array(y_all)).float().view((-1, 1))
    elif args.model == "SeveritySumNN":
        y_all = torch.from_numpy(np.array(y_all)).float()
    
    # y_pred = np.zeros_like(y_all)
    
    # for idx, X in enumerate(X_all):
    #     y_pred[idx] = classifier.predict(X)
        
    y_pred = classifier.predict(X_all)    
    return print_evaluation(y_all, y_pred)


def evaluate_rnn(args, classifier, exs):
    # X_all = torch.from_numpy(np.array(X_all)).float()
    # if args.model == "TrivialNN":
    #     y_all = torch.from_numpy(np.array(y_all)).float().view((-1, 1))
    # elif args.model == "SeveritySumNN":
    #     y_all = torch.from_numpy(np.array(y_all)).float()

    y_pred = np.zeros((len(exs), exs[0].x_temporal.shape[1]))
    y_all = np.zeros_like(y_pred)
    for idx, X in enumerate(exs):
        y_pred[idx, :] = classifier.predict(torch.from_numpy(exs[idx].x_temporal).float(), torch.from_numpy(exs[idx].x_geo).float())
        y_all[idx, :] = exs[idx].y
    return print_evaluation(y_all, y_pred)


if __name__ == '__main__':
    args = _parse_args()
    print(args)
    
    # Train and evaluate
    start_time = time.time()
    if args.model == "TrivialNN":
        X_train, y_train, X_val, y_val, X_test, y_test, preprocessor = load_datasets(is_remove_lon_lat=True)
        print("%i train exs, %i dev exs, %i test exs" % (len(y_train), len(y_val), len(y_test)))
        model = train_trivialnn(args, X_train, y_train, X_val, y_val)

        model.to("cpu")
        print("=====Train Accuracy=====")
        train_eval = evaluate(args, model, X_train, y_train)
        print(train_eval)
        
        print("=====Val Accuracy=====")
        val_eval = evaluate(args, model, X_val, y_val)
        print(val_eval)
        
        train_eval_time = time.time() - start_time
        print("Time for training and evaluation: %.2f seconds" % train_eval_time)
    
    elif args.model == "SeveritySumNN":
        X_train, y_train, X_val, y_val, X_test, y_test, preprocessor, y_col_names = load_datasets_severities_sum()
        print("X_train.shape", X_train.shape, "y_train.shape", y_train.shape,
              "\nX_val.shape", X_val.shape, "y_val.shape", y_val.shape,
              "\nX_test.shape", X_test.shape, "y_test.shape", y_test.shape)
        model = train_severity_sum_nn(args, X_train, y_train, X_val, y_val)

        model.to("cpu")
        print("=====Train Accuracy=====")
        train_eval = evaluate(args, model, X_train, y_train)
        print(train_eval)
        
        print("=====Val Accuracy=====")
        val_eval = evaluate(args, model, X_val, y_val)
        print(val_eval)
        
        train_eval_time = time.time() - start_time
        print("Time for training and evaluation: %.2f seconds" % train_eval_time)
    
    elif args.model == "SeverityIndNN":
        train_exs, test_exs = load_datasets_severities_ind()
        print("train_exs.shape", len(train_exs), "; test_exs.shape", len(test_exs))
        print("example", train_exs[0])
        model = train_severity_ind_nn(args, train_exs, test_exs)
        
        model.to("cpu")
        print("=====Train Accuracy=====")
        train_eval = evaluate_rnn(args, model, train_exs)
        print(train_eval)
        
        print("=====Val Accuracy=====")
        val_eval = evaluate_rnn(args, model, test_exs)
        print(val_eval)
        
        train_eval_time = time.time() - start_time
        print("Time for training and evaluation: %.2f seconds" % train_eval_time)        

    
    # t1 = model.predict(torch.from_numpy(np.array(X_val)).float())
    # # y_pred_new = self.classifier.predict(X_new)
    
    # # self.sensitivity_list[col_idx, :] = np.mean((y_pred_new - self.y_pred) / self.y_pred, axis=0)
    # np.mean(t1, axis=0)
    # self.sensitivity_list.append(np.mean((t1.numpy() - t1.numpy()) / t1.numpy(), axis=0))
    