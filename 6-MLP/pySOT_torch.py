#!/usr/bin/env python
import csv
import numpy as np
from threading import Thread, current_thread
from subprocess32 import Popen, PIPE
from datetime import datetime
import time

class TorchOptim:

    bestResult = 1000
    f_eval_count = 0
    seed = 139
    server = 'unset'

    # Hyperparameter to optimise:
    hyper_map = {
            'mean' : 0, # mean of Gaussian initializing function 
            'std' : 1, # std of Gaussian initializing function
            'learnRate' : 2, # SGD learning rate
            'momentum' : 3, # SGD momentum coefficient
            'epochs' : 4, # (integer) Number of epochs
            'hiddenNodes' : 5 # (integer) Number of nodes in the second to last layer
    }

    def __init__(self, seed, server, dim=6):

        self.seed = seed
        self.server = server
        self.f_eval_count = 0
        m = self.hyper_map 

        self.xlow = np.zeros(dim)
        self.xup = np.zeros(dim)

        # human value 0.0000
        self.xlow[m['mean']] = 0.0000001
        self.xup[m['mean']] =  0.0100000

        # human value 0.0100
        self.xlow[m['std']] = 0.0010
        self.xup[m['std']] =  0.1000

        # human value 0.1
        self.xlow[m['learnRate']] = 0.001000
        self.xup[m['learnRate']] =  0.200000

        # human value 0.9
        self.xlow[m['momentum']] = 0.80000
        self.xup[m['momentum']] =  0.99999

        # human value 10
        self.xlow[m['epochs']] = 8
        self.xup[m['epochs']] = 20

        # human value 50
        self.xlow[m['hiddenNodes']] = 50
        self.xup[m['hiddenNodes']] = 200

        self.dim = dim
        self.info = 'Optimise a simple MLP network over MNIST dataset'
        self.continuous = np.arange(0, 4)
        self.integer = np.arange(4, dim)

    def print_result_directly(self, x, result):
        self.f_eval_count = self.f_eval_count + 1
        experimentId = 'p-'+str(len(x))+'-'+str(self.f_eval_count)+'-'+self.seed+'-'+self.server
        fileId = 'p-'+str(len(x))+'-'+self.seed+'-'+self.server
        millis = int(round(time.time() * 1000))

        if self.bestResult > result:
            self.bestResult = result
        row = [self.bestResult, -1, result, -1, self.f_eval_count, millis] 
        for xi in range(0, len(x)):
            row.append(x[xi])
        with open('logs/'+fileId+'-output.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        


    def objfunction(self, x):
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')

        self.f_eval_count = self.f_eval_count + 1
        experimentId = 'p-'+str(len(x))+'-'+str(self.f_eval_count)+'-'+self.seed+'-'+self.server
        fileId = 'p-'+str(len(x))+'-'+self.seed+'-'+self.server
        m = self.hyper_map

        exp_arg = []
        exp_arg.append('th'),
        exp_arg.append('eval_mnist_GPU.lua')
        exp_arg.append('--mean')
        exp_arg.append(str(x[m['mean']]))
        exp_arg.append('--std')
        exp_arg.append(str(x[m['std']]))
        exp_arg.append('--learnRate')
        exp_arg.append(str(x[m['learnRate']]))
        exp_arg.append('--momentum')
        exp_arg.append(str(x[m['momentum']]))
        exp_arg.append('--epochs')
        exp_arg.append(str(x[m['epochs']]))
        exp_arg.append('--hiddenNodes')
        exp_arg.append(str(x[m['hiddenNodes']]))
        exp_arg.append('--experimentId')
        exp_arg.append(experimentId)
        exp_arg.append('--seed')
        exp_arg.append(self.seed)
        
        millis_start = int(round(time.time() * 1000))
        proc = Popen(exp_arg, stdout=PIPE)
        out, err = proc.communicate()
        
        if proc.returncode == 0:
            results = out.split('###')
            result = float(results[0])
            testResult = float(results[1])
            millis = int(round(time.time() * 1000))
            f_eval_time = millis - millis_start

            if self.bestResult > result:
                self.bestResult = result
            
            row = [self.bestResult, f_eval_time, result, testResult, self.f_eval_count, millis] 
            for xi in range(0, len(x)):
                row.append(x[xi])
            with open('logs/'+fileId+'-output.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            return result
        else:
            print err
            raise ValueError('Function evaluation error')
    
    def print_parameters(self, x):

        print current_thread()
        m = self.hyper_map
        print ''
        for p in m:
             print p+'\t : %g' % float(x[m[p]])

