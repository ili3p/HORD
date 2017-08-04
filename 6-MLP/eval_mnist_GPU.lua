require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'optim'
require 'sys'
mnist = require 'mnist'
logger = require "log"

local opt = lapp[[
   --mean             (default 0.0)       The mean of init function HP1
   --std              (default 0.01)      The std of init function HP2
   --learnRate        (default 0.1)       The SGD learning rate HP3 
   --momentum         (default 0.9)       The SGD momentum HP4
   --epochs           (default 10)        Number of epochs to train HP5
   --hiddenNodes      (default 100)       Number of nodes in the hidden layer HP6
   --batchSize        (default 128)       Number of instances per batch (NOT optimized) 
   --seed             (default 2275)      Random seed number
   --deviceId         (default 1)         GPU device ID
   --experimentId     (default "ad_hock") Experiment ID, used to create unique log
   --threads          (default 16)        Number of CPU threads
   --save             (default "logs")    Log folder number
   --showProgress                         Output progress to console
]]


torch.manualSeed(opt.seed)
torch.setnumthreads(opt.threads)
cutorch.manualSeed(opt.seed)
cutorch.setDevice(opt.deviceId)
fullset = mnist.traindataset()
testset = mnist.testdataset()

valSize = 10000
trainSize = fullset.size - valSize
-- split datasets and convert to float 
trainset = {
  size = trainSize,
  data = fullset.data[{{1, trainSize}}]:float(),
  label = fullset.label[{{1, trainSize}}]
}


validationset = {
  size = valSize,
  data = fullset.data[{{trainSize + 1, fullset.size}}]:float(),
  label = fullset.label[{{trainSize + 1, fullset.size}}]
}

-- to float
testset.data = testset.data:float()

-- normalize datasets
trainset.data:add(-trainset.data:mean())
trainset.data:mul(1/trainset.data:std())

validationset.data:add(-validationset.data:mean())
validationset.data:mul(1/validationset.data:std())

testset.data:add(-testset.data:mean())
testset.data:mul(1/testset.data:std())

imgDim = 28
classes = 10

-- create the MLP model
model = nn.Sequential()

model:add(nn.View(imgDim * imgDim))
model:add(nn.Linear(imgDim * imgDim, opt.hiddenNodes))
model:add(nn.ReLU())
model:add(nn.Linear(opt.hiddenNodes, classes))

-- loss function and criterion - cross entropy loss
model:add(nn.LogSoftMax())
criterion = nn.CrossEntropyCriterion()

model = model:cuda()
criterion = criterion:cuda()

model.modules[2].weight:normal(opt.mean, opt.std)
model.modules[2].bias:normal(opt.mean, opt.std)

model.modules[4].weight:normal(opt.mean, opt.std)
model.modules[4].bias:normal(opt.mean, opt.std)

parameters, gradParameters = model:getParameters()

logger.outfile = paths.concat(opt.save, opt.experimentId..'.log')
logger.logToConsole = false

train = function(dataset)

  local batchLoss = 0 
  local b = 0
  while b ~= dataset.size do
    
    local currentBatchSize = math.min(opt.batchSize, dataset.size - b)
    -- create new batch
    local inputs = torch.CudaTensor(currentBatchSize, imgDim, imgDim)
    local targets = torch.CudaTensor(currentBatchSize)

    -- load batch with data
    local j = 1
    for i = b + 1, math.min(b + opt.batchSize, dataset.size) do
      local input = dataset.data[i]:clone()
      local target = dataset.label[i]
      inputs[j] = input
      targets[j] = target + 1 -- plus one since lua is 1-index based
      j = j + 1
    end

    -- one batch iteration
    local feval = function(x)

      if x ~= parameters then
        parameteres:copy(x)
     end

      gradParameters:zero()

      local outputs = model:forward(inputs)
      local loss = criterion:forward(outputs, targets)
      batchLoss = batchLoss + loss

      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)

      return loss, gradParameters
    end

    sgdState = sgdState or {
      learningRate = opt.learnRate,
      momentum = opt.momentum
    }

    optim.sgd(feval, parameters, sgdState)
    
    b = b + currentBatchSize
    if opt.showProgress then
      xlua.progress(math.floor(b/opt.batchSize),math.floor(dataset.size/opt.batchSize))
    end
  end
  
  if opt.showProgress then
    batchLoss = batchLoss/(math.floor(dataset.size/opt.batchSize)+1)
    print('Epoch: '..epoch..' mean loss: ' .. batchLoss)
  end
end

test = function(dataset, setName)

  local correct = 0
  local b = 0
  while b ~= dataset.size do

    local currentBatchSize = math.min(opt.batchSize, dataset.size - b)

    local inputs = torch.CudaTensor(currentBatchSize, imgDim, imgDim)
    local targets = torch.CudaTensor(currentBatchSize)

    local j = 1
    for i = b + 1, math.min(b + opt.batchSize, dataset.size) do
      local input = dataset.data[i]:clone()
      local target = dataset.label[i]
      inputs[j] = input
      targets[j] = target + 1
      j = j + 1
    end

    local predictions = model:forward(inputs)

    for i = 1, currentBatchSize do
      local _, predicted = predictions[i]:max(1)
      if predicted[1] == targets[i] then
        correct = correct + 1
      end
    end

    b = b + currentBatchSize
    if opt.showProgress then
      xlua.progress(b, dataset.size)
    end
    
  end

  errorRate = 1 - (correct / dataset.size)

  if opt.showProgress then
    print('Epoch: '..epoch..' '..setName..' error: ' .. tostring(errorRate * 100))
  end

  return errorRate
end

epoch = 1
while epoch <= opt.epochs do
  train(trainset)
  epoch = epoch + 1
end

epoch = epoch - 1
local result = test(validationset, 'validation')
local testResult = test(testset, 'test')
logger.info(opt)
logger.info('validation: '..tostring(result)..'\t test: '..testResult)
print(tostring(result)..'###'..tostring(testResult))
