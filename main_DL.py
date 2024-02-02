import shutil, time, random, json
from tqdm import tqdm
from os.path import join, exists
import numpy as np

from config import get_config
from model.net import Net
from data_loader import get_data_loader
from utils import get_trial_num, AverageMeter, prepare_dirs, save_config, get_fold_ids, set_optimizer
from utils import load_config, RecursiveNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn



def main(config):
    if config.classification_way=='AD_SZ_HC':
        config.num_classes = 3
    else: 
        config.num_classes = 2
        
    # ensure directories are setup
    prepare_dirs(config)
    
    # set a standard random seed for reproducible results
    random.seed(41)
    np.random.seed(41)
    
    if config.learning_stage == 'train' and not config.resume:
        try:
            save_config(config)
        except ValueError:
            print(
                "[!] file already exist. Either change the trial number,",
                "or delete the json file and rerun.",
                sep=' ',
            )
            
    trainer = Trainer(config)

    if config.learning_stage == 'train': # train + validate
        trainer.train()
    elif config.learning_stage == 'test':
        trainer.test()
    else:
        raise NotImplementedError(f'{config.learning_stage} is not supported')


    
class Trainer(object):
    def __init__(self, config):
        self.config = config

        # set device on GPU if available, else CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # model & path params
        self.trial_num = get_trial_num(self.config)
        self.logs_dir = join(self.config.logs_dir, self.config.model, self.config.classification_way,
                             config.folder_name, self.trial_num)

    def setup_model(self):
        # set model and loss function
        model = Net(self.config)
        criterion = torch.nn.CrossEntropyLoss()
        
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print('# of gpus available:' , torch.cuda.device_count())
            model = nn.DataParallel(model)
        model = model.to(self.device)
        criterion = criterion.to(self.device)
        cudnn.benchmark = (self.device=='cuda')
            
        return model, criterion
    
    def train(self):         
        metric_logs = {}
        logs_path = join(self.logs_dir, f'metric_logs_fold{self.config.fold_num}_seed{self.config.random_seed}.json')
        
        test_data, val_data, train_data = get_fold_ids(fold_num=self.config.fold_num, num_folds=self.config.num_folds,
                                                       seed=self.config.random_seed, cway=self.config.classification_way)
        
        # create data loaders
        valid_loader, num_valid = get_data_loader(config=self.config, data=val_data, shuffle=False, pin_memory=True)
        train_loader, num_train = get_data_loader(config=self.config, data=train_data,
                                                  shuffle=True, pin_memory=True, drop_last=True)

        print(f'{test_data.shape[0]} test subjects, {val_data.shape[0]} validation subjects, {train_data.shape[0]} train subjects')
        
        print("\n==================================")
        print('\tFold number {}/{}'.format(self.config.fold_num, self.config.num_folds))
        print("==================================")

        self.start_epoch = 1 
        # early stopping params
        self.best_perf_metric = None
        self.counter = 0
        # build model and criterion
        self.model, self.criterion = self.setup_model()
        # build optimizer
        self.optimizer = set_optimizer(self.config, self.model)
        
        if self.config.resume:
            self.load_checkpoint(best=False, fold=self.config.fold_num)
            if exists(logs_path):
                with open(logs_path, 'r+') as f:
                    metric_logs = json.load(f)

        for epoch in range(self.start_epoch, self.config.epochs):

            train_loss, train_acc, train_time = self.train_one_epoch(epoch, train_loader, num_train)
            valid_loss, valid_acc = self.validate(epoch, valid_loader)

            # check for improvement
            if self.best_perf_metric is None:
                self.best_perf_metric = valid_loss if self.config.early_stop_on_loss else valid_acc
                is_best = True
            else:
                is_best = valid_loss < self.best_perf_metric if self.config.early_stop_on_loss else valid_acc > self.best_perf_metric 
                
            msg = "Epoch:{}, {:.1f}s - train loss: {:.3f}, train acc: {:.1f} - val loss: {:.3f}, valid acc: {:.1f}"
            if is_best:
                msg += " [*]"
                self.counter = 0
            print(msg.format(epoch, train_time, train_loss, train_acc, valid_loss, valid_acc))

            # checkpoint the model
            if not is_best:
                self.counter += 1
            if self.counter > self.config.train_patience and self.config.early_stop:
                print("[!] No improvement in a while, stopping training.")
                break
            self.best_perf_metric = min(valid_loss, self.best_perf_metric) if self.config.early_stop_on_loss else max(valid_acc, self.best_perf_metric)
            self.save_checkpoint(
            {
                'epoch': epoch,
                'model_state': self.model.state_dict(),
                'optim_state': self.optimizer.state_dict(),
                'best_perf_metric': self.best_perf_metric,
                'counter': self.counter,
            }, is_best, fold=self.config.fold_num)
            
            # log valid loss
            metric_logs.update({f'best_perf_metric':round(self.best_perf_metric,5)})
            metric_logs.update({f'epoch_{epoch}_tr_val_loss':[round(train_loss,5), round(valid_loss,5)]})
            metric_logs.update({f'epoch_{epoch}_tr_val_acc':[round(train_acc,2), round(valid_acc,2)]})
            with open(logs_path, 'w') as fp:
                json.dump(metric_logs, fp, indent=4, sort_keys=False)           

        print("\ndone!")

    def train_one_epoch(self, epoch, train_loader, num_train):
        batch_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()
        batch_time.reset
        losses.reset
        accuracies.reset
        tic = time.time()
        
        # switch to train mode
        self.model.train()
        
        with tqdm(total=num_train) as pbar:            
            for batch_index, (inputs, targets) in enumerate(train_loader):  
                batch_size = inputs.shape[0]

                inputs, targets = inputs.to(self.device), targets.to(self.device, dtype=torch.int64)

                # compute loss
                outputs = self.model(inputs) 
                loss = self.criterion(outputs, targets)   
                
                # update metric
                losses.update(loss.item(), batch_size)
                acc = (torch.argmax(outputs, 1) == targets).float().mean()*100
                accuracies.update(acc.item(), batch_size) 

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc-tic)
                tic = time.time()

                pbar.set_description(("{:.1f}s - loss: {:.3f}".format(batch_time.val, losses.val)))
                pbar.update(batch_size)

        return losses.avg, accuracies.avg, batch_time.sum
    
    def validate(self, epoch, valid_loader):
        losses = AverageMeter()
        accuracies = AverageMeter()
        losses.reset
        accuracies.reset
        
        # switch to evaluate mode
        self.model.eval()

        with torch.no_grad():
            for batch_index, (inputs, targets) in enumerate(valid_loader):
                batch_size = inputs.shape[0]
                inputs, targets = inputs.to(self.device), targets.to(self.device, dtype=torch.int64)

                # compute loss
                outputs = self.model(inputs) 
                loss = self.criterion(outputs, targets)                     

                # update metric
                losses.update(loss.item(), batch_size)
                acc = (torch.argmax(outputs, 1) == targets).float().mean()*100
                accuracies.update(acc.item(), batch_size) 

        return losses.avg, accuracies.avg
        
    def test(self):
        losses = AverageMeter()
        accuracies = AverageMeter()
        losses.reset
        accuracies.reset
                
        metric_logs = {}
        seed = self.config.random_seed
        trained_model_dir = join(self.config.logs_dir, self.config.model, self.config.classification_way, 
                                 self.config.folder_name, self.trial_num)
        params = load_config(trained_model_dir, self.config.fold_num)
        self.config = RecursiveNamespace(**params)
        self.config.random_seed = seed
        
        test_data, val_data, train_data = get_fold_ids(fold_num=self.config.fold_num, num_folds=self.config.num_folds,
                                                       seed=self.config.random_seed, cway=self.config.classification_way)
        test_loader, num_test = get_data_loader(config=self.config, data=test_data, shuffle=True, pin_memory=True)
        
        # build model and criterion
        self.model, self.criterion = self.setup_model()
        # build optimizer
        self.optimizer = set_optimizer(self.config, self.model)   
        # load best model
        self.load_checkpoint(best=self.config.best, fold=self.config.fold_num)
        # switch to evaluate mode
        self.model.eval()

        t_TP, t_TN, t_FP, t_FN = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch_index, (inputs, targets) in enumerate(test_loader):
                batch_size = inputs.shape[0]
                inputs, targets = inputs.to(self.device), targets.to(self.device, dtype=torch.int64)

                # compute loss
                outputs = self.model(inputs) 
                loss = self.criterion(outputs, targets)  
                
                if self.config.num_classes == 2:
                    # create a confusion matrix
                    conf_matrix = torch.zeros(self.config.num_classes, self.config.num_classes)
                    for t, o in zip(targets, torch.argmax(outputs, 1)):
                        conf_matrix[t, o] += 1
                    
                    TP = conf_matrix.diag()[0]
                    idx = torch.ones(self.config.num_classes).byte()
                    idx[0] = 0
                    TN = conf_matrix[idx.nonzero()[:, None], idx.nonzero()].sum() 
                    FP = conf_matrix[idx, 0].sum()
                    FN = conf_matrix[0, idx].sum()
                    t_TP += TP
                    t_TN += TN
                    t_FP += FP
                    t_FN += FN
                        
                # update metric
                losses.update(loss.item(), batch_size)
                acc = (torch.argmax(outputs, 1) == targets).float().mean()*100
                accuracies.update(acc.item(), batch_size) 
         
        print("[*] Avg Test Loss: ({:.2f})".format(losses.avg))
        print("[*] Avg Test Accuracy: ({:.2f})".format(accuracies.avg))
        
        logs_path = f'./logs/DL/{self.config.classification_way}__{self.config.folder_name}__testlogs.json'
        if exists(logs_path):
            with open(logs_path, 'r+') as f:
                metric_logs = json.load(f)
        
        if self.config.num_classes == 2:
            sens = t_TP / (t_TP + t_FN) * 100
            spec = t_TN / (t_TN + t_FP) * 100
            metric_logs.update({
                f'test_accuracy_fold{self.config.fold_num}_seed{self.config.random_seed}': accuracies.avg, 
                f'test_sensitivity_fold{self.config.fold_num}_seed{self.config.random_seed}': sens.item(), 
                f'test_specificity_fold{self.config.fold_num}_seed{self.config.random_seed}': spec.item(),
            })
        else:
            metric_logs.update({
                f'test_accuracy_fold{self.config.fold_num}_seed{self.config.random_seed}': accuracies.avg, 
                f'test_sensitivity_fold{self.config.fold_num}_seed{self.config.random_seed}': 0.0, 
                f'test_specificity_fold{self.config.fold_num}_seed{self.config.random_seed}': 0.0,
            })
        with open(logs_path, 'w') as fp:
            json.dump(metric_logs, fp, indent=4, sort_keys=False)
                
        return losses.avg, accuracies.avg
                
    def save_checkpoint(self, state, is_best, fold=0):
        filename = f'model_ckpt_fold{fold}_seed{self.config.random_seed}.tar'
        ckpt_path = join(self.logs_dir, filename)
        torch.save(state, ckpt_path)

        if is_best:
            filename = f'best_model_ckpt_fold{fold}_seed{self.config.random_seed}.tar'
            shutil.copyfile(
                ckpt_path, join(self.logs_dir, filename)
            )
            
    def load_checkpoint(self, best, fold):
        model_dir = self.logs_dir
        print("[*] Loading model from {} - fold: {}".format(model_dir, fold))

        filename = f'model_ckpt_fold{fold}_seed{self.config.random_seed}.tar'
        if best:
            filename = f'best_model_ckpt_fold{fold}_seed{self.config.random_seed}.tar'
        ckpt_path = join(model_dir, filename)
        ckpt = torch.load(ckpt_path, map_location=torch.device(self.device))
        
        # load variables from checkpoint
        self.start_epoch = ckpt['epoch'] + 1
        self.best_perf_metric = ckpt['best_perf_metric']
        self.counter = ckpt['counter']
        self.model.load_state_dict(ckpt['model_state'])
        model_dict = {k.replace("module.",""): v for k, v in ckpt['model_state'].items()}
        self.model.load_state_dict(model_dict)
        self.optimizer.load_state_dict(ckpt['optim_state'])
        
        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} with best valid loss of {:.4f}".format(
                    filename, ckpt['epoch'], ckpt['best_perf_metric'])
            )
        else:
            print(
                "[*] Loaded {} checkpoint @ epoch {} with best valid loss of {:.4f}".format(
                    filename, ckpt['epoch'], ckpt['best_perf_metric'])
            )

            
            
if __name__ == '__main__':    
    config, unparsed = get_config()
    main(config)
            