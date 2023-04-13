#Python script for training model

import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

import utils
import models.LeNet as ln
import models.data_loader as data_loader
from evaluate import evaluate

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', default='./data/', help="Directory that holds dataset")
parser.add_argument('--model_dir', default='experiments/LeNet/base_model', help="Directory that holds model parameters file")
parser.add_argument('--model', default='LeNet', help="Model used for training (LeNet, VGG16, ResNet18)")
parser.add_argument('--dataset', default='MNIST', help="Dataset used for training model (MNIST, CIFAR10)")
parser.add_argument('--restore_file', default=None, help="Optional, name of the file in --model_dir containing weights to reload before training")



def train(model, optimizer, loss_fn, dataloader, metrics, params):
    #Train the model on 'num_steps' batches
    #Args:
        #model: (torch.nn.Module) the neural network
        #optimizer: (torch.optim) optimizer for parameters of model
        #loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        #dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        #metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        #params: (Params) hyperparameters
        #num_steps: (int) number of batches to train on, each of size params.batch_size
    
    #set model to training mode
    model.train()

    #summary for current training loop and a running average object for loss
    summary = []
    loss_avg = utils.RunningAverage()

    #show progress bar with tqdm
    with tqdm(total=len(dataloader)) as t:

        for i, (train_batch, labels_batch) in enumerate(dataloader):

            #If available, move to GPU
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(non_blocking=True), labels_batch.cuda(non_blocking=True)

            #convert to torch variables
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            #compute model output and loss
            output_batch = model(train_batch, params)
            loss = loss_fn(output_batch, labels_batch)

            #clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            #perform updates using calculated gradients
            optimizer.step()

            #evaluate summaries
            if i % params.save_summary_steps == 0:

                #extract data from torch variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                #compute all metrics on this batch
                summary_batch = {metric: metrics[metric](output_batch, labels_batch) for metric in metrics}
                summary_batch['loss'] = loss.item()
                summary.append(summary_batch)

            #update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    #compute average of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]for x in summary]) for metric in summary[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k,v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir, restore_file=None):

    #Train the model and evaluate every epoch
    #Args:
        #model: (torch.nn.Module) the neural network
        #train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        #val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        #optimizer: (torch.optim) optimizer for parameters of model
        #loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        #metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        #params: (Params) hyperparameters
        #model_dir: (string) directory containing config, weights and log
        #restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)

    #If specified, reload weights from restore_file 
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    for epoch in range(params.num_epochs):
        
        #run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        #compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, metrics, params)

        #evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)

        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc

        #save weights
        utils.save_checkpoint({'epoch': epoch + 1, 
                              'state_dict': model.state_dict(),
                              'optim_dict': optimizer.state_dict()},
                              is_best = is_best,
                              checkpoint = model_dir)
        
        #if best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            #save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        #save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

if __name__ == '__main__':

    #load parameters from params.json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    #If available, use GPU
    params.cuda = torch.cuda.is_available()

    #Set the random seed for reproducible experiments
    torch.manual_seed(1514)
    if params.cuda:
        torch.cuda.manual_seed(1514)

    #Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    #Create the input data pipeline
    logging.info("Loading dataset...")

    #get dataloader
    dataloaders = data_loader.fetch_dataloader(['train', 'val'], args.data_dir, args.dataset, params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']

    logging.info("- done.")

    #define the model and optimizer
    model = ln.LeNet(params).cuda() if params.cuda else ln.LeNet(params)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    #get loss function and metrics
    loss_fn = ln.loss_fn
    metrics = ln.metrics

    #train the model 
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, metrics, params, args.model_dir, args.restore_file)


