#!/usr/bin/env python
import numpy as np
from threading import Thread, current_thread
from subprocess32 import Popen, PIPE
from datetime import datetime
import time
import csv

class TorchOptim:

    bestResult = 1000
    f_eval_count = 0
    seed = 139
    server = 'unset'

    # Hyperparameter to optimise:
    hyper_map = {
            'learningRate' : 0, # best 0.29622
            'momentum' : 1, #0.75 
            'weightDecay': 2, # 0.0015
            'learningRateDecay' : 3, # 0.000349 
            'leakyReLU_fc1' : 4,
            'leakyReLU_fc2' : 5,
            'std_fc1' : 6,
            'std_fc2' : 7, 
            'std_conv1' : 8,
            'std_conv2' : 9,
            'drop_rate1' : 10,
            'drop_rate2' : 11,
            'drop_rate3' : 12,
            'drop_rate4' : 13,
            'batchSize' : 14, 
            'hiddenNodes_fc1' : 15, # 282
            'hiddenNodes_fc2' : 16, # 320
            'hiddenNodes_conv1' : 17, # 124
            'hiddenNodes_conv2' : 18 # 117
    }

    def __init__(self, seed, server, dim=19):

        self.seed = seed
        self.server = server
        self.f_eval_count = 0
        m = self.hyper_map 

        self.xlow = np.zeros(dim)
        self.xup = np.zeros(dim)

        # manual setting 0.1
        self.xlow[m['learningRate']] = 0.005  #0
        self.xup[m['learningRate']] = 0.300

        # manual setting 0.9
        self.xlow[m['momentum']] = 0.60  #1
        self.xup[m['momentum']] =  0.99

        # manual setting 5e-4
        self.xlow[m['weightDecay']] = 0 #2
        self.xup[m['weightDecay']] = 0.01

        # human value 0.0002
        self.xlow[m['learningRateDecay']] = 0  #3
        self.xup[m['learningRateDecay']] = 0.01

        # 0.01
        self.xlow[m['leakyReLU_fc1']] = 0   #4
        self.xup[m['leakyReLU_fc1']] = 0.5

        # 0.01
        self.xlow[m['leakyReLU_fc2']] = 0  #5
        self.xup[m['leakyReLU_fc2']] = 0.5

        # 0.01
        self.xlow[m['std_fc1']] = 0.0   #6
        self.xup[m['std_fc1']] = 0.5

        # 0.01
        self.xlow[m['std_fc2']] = 0.0   #7
        self.xup[m['std_fc2']] = 0.5

        # 0.01
        self.xlow[m['std_conv1']] = 0.0   #8
        self.xup[m['std_conv1']] = 0.5

        # 0.01
        self.xlow[m['std_conv2']] = 0.0     #9
        self.xup[m['std_conv2']] = 0.5

        self.xlow[m['drop_rate1']] = 0.0  #10
        self.xup[m['drop_rate1']] = 0.8

        self.xlow[m['drop_rate2']] = 0.0  #11
        self.xup[m['drop_rate2']] = 0.8

        self.xlow[m['drop_rate3']] = 0.0    #12
        self.xup[m['drop_rate3']] = 0.8

        self.xlow[m['drop_rate4']] = 0.0    #13
        self.xup[m['drop_rate4']] = 0.8

        # 128
        self.xlow[m['batchSize']] = 32      #14
        self.xup[m['batchSize']] = 512

        # 200
        self.xlow[m['hiddenNodes_fc1']] = 100  #15
        self.xup[m['hiddenNodes_fc1']] = 400

        # 256
        self.xlow[m['hiddenNodes_fc2']] = 100    #16
        self.xup[m['hiddenNodes_fc2']] = 400

        # 32
        self.xlow[m['hiddenNodes_conv1']] = 16  #17
        self.xup[m['hiddenNodes_conv1']] = 128

        # 64
        self.xlow[m['hiddenNodes_conv2']] = 16   #18
        self.xup[m['hiddenNodes_conv2']] = 128

        self.dim = dim
        self.info = 'Optimise a CNN on CIFAR-10 dataset'
        self.continuous = np.arange(0, 14)
        self.integer = np.arange(14, dim)

    def objfunction(self, x):
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')

        self.f_eval_count = self.f_eval_count + 1
        experimentId = 'p-'+str(len(x))+'-'+str(self.f_eval_count)+'-'+self.seed+'-'+self.server
        fileId = 'p-'+str(len(x))+'-'+self.seed+'-'+self.server
        m = self.hyper_map

        exp_arg = []
        exp_arg.append('th'),
        exp_arg.append('eval_CIFAR_CNN_dropout.lua')

        exp_arg.append('--learningRate')
        exp_arg.append(str(x[m['learningRate']]))

        exp_arg.append('--momentum')
        exp_arg.append(str(x[m['momentum']]))

        exp_arg.append('--weightDecay')
        exp_arg.append(str(x[m['weightDecay']]))

        exp_arg.append('--learningRateDecay')
        exp_arg.append(str(x[m['learningRateDecay']])) 

        exp_arg.append('--batchSize')
        exp_arg.append(str(x[m['batchSize']])) 

        exp_arg.append('--leakyReLU_fc1')
        exp_arg.append(str(x[m['leakyReLU_fc1']])) 

        exp_arg.append('--leakyReLU_fc2')
        exp_arg.append(str(x[m['leakyReLU_fc2']])) 

        exp_arg.append('--std_fc1')
        exp_arg.append(str(x[m['std_fc1']])) 

        exp_arg.append('--std_fc2')
        exp_arg.append(str(x[m['std_fc2']])) 

        exp_arg.append('--std_conv1')
        exp_arg.append(str(x[m['std_conv1']])) 

        exp_arg.append('--std_conv2')
        exp_arg.append(str(x[m['std_conv2']])) 

        exp_arg.append('--drop_rate1')
        exp_arg.append(str(x[m['drop_rate1']])) 

        exp_arg.append('--drop_rate2')
        exp_arg.append(str(x[m['drop_rate2']])) 

        exp_arg.append('--drop_rate3')
        exp_arg.append(str(x[m['drop_rate3']])) 

        exp_arg.append('--drop_rate4')
        exp_arg.append(str(x[m['drop_rate4']])) 

        exp_arg.append('--hiddenNodes_fc1')
        exp_arg.append(str(x[m['hiddenNodes_fc1']])) 

        exp_arg.append('--hiddenNodes_fc2')
        exp_arg.append(str(x[m['hiddenNodes_fc2']])) 
        
        exp_arg.append('--hiddenNodes_conv1')
        exp_arg.append(str(x[m['hiddenNodes_conv1']])) 

        exp_arg.append('--hiddenNodes_conv2')
        exp_arg.append(str(x[m['hiddenNodes_conv2']])) 

        exp_arg.append('--experimentId')
        exp_arg.append(experimentId)
        exp_arg.append('--seed')
        exp_arg.append(self.seed)

        millis_start = int(round(time.time() * 1000))
        row = [self.f_eval_count, millis_start]
        for xi in range(0, len(x)):
            row.append(x[xi])
        with open('logs/'+fileId+'-input.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(row)

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
            return 100.0
    
    def print_parameters(self, x):

        print current_thread()
        m = self.hyper_map
        print ''
        for p in m:
             print p+'\t : %g' % float(x[m[p]])

