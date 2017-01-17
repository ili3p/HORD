require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'optim'
logger = require "log"
require 'loader'
-- require 'cudnn'

-- cudnn.fastest = true
-- cudnn.benchmark = true

local o = lapp[[
   --learningRate      (default 0.1)       The SGD learning rate ### OPTIM 1 ###
   --momentum          (default 0.9)       The SGD momentum ### OPTIM 2 ###
   --weightDecay       (default 0.0005)    Weight ### OPTIM 3 ### 
   --batchSize         (default 128)       Number of instances per batch ### OPTIM 4 ###
   --learningRateDecay (default 0.0001)    Learn rate decay ### OPTIM 5 ###
   --leakyReLU_fc1     (default 0.01)      LeakyReLU ### OPTIM 6 ###
   --leakyReLU_fc2     (default 0.01)      LeakyReLU ### OPTIM 7 ###
   --std_fc1           (default 0.01)      The std of init function ### OPTIM 8 ###
   --std_fc2           (default 0.01)      The std of init function ### OPTIM 9 ###
   --hiddenNodes_fc1   (default 200)       Number of nodes in the hidden layer ### OPTIM 10 ### 
   --hiddenNodes_fc2   (default 256)       Number of nodes in the hidden layer ### OPTIM 11 ###
   --std_conv1         (default 0.01)      The std of init function ### OPTIM 12 ### 
   --std_conv2         (default 0.01)      The std of init function ### OPTIM 13 ###
   --hiddenNodes_conv1 (default 32)        Number of nodes in the hidden layer ### OPTIM 14 ###
   --hiddenNodes_conv2 (default 64)        Number of nodes in the hidden layer ### OPTIM 15 ###
   --kernelSize_conv1  (default 5)         Kernel size 
   --kernelSize_conv2  (default 3)         Kernel size
   --sBNormEps_conv1   (default 1e-5)      SpatialBatchNormalization
   --sBNormEps_conv2   (default 1e-5)      SpatialBatchNormalization
   --sBNMomentum_conv1 (default 0.1)       SpatialBatchNormalization Momentum
   --sBNMomentum_conv2 (default 0.1)       SpatialBatchNormalization Momentum
   --maxEpochs         (default 10)        Number of epochs to train 
   --mean              (default 0.0)       The mean of init function 
   --dropoutRate_fc1   (default 0.5)       Drop
   --dropoutRate_fc2   (default 0.5)       Drop
   --seed              (default 2275)      Random seed number
   --experimentId      (default "ad_hock") Experiment ID, used to create unique log
   --save              (default "logs")    Log folder number
   --showProgress                         Output progress to console
]]


classes = 10
imgDim = 28

torch.manualSeed(o.seed)
torch.setnumthreads(12)
cutorch.manualSeed(o.seed)

local trainset, validationset, testset = loadMnist()

model = nn.Sequential()
------------------------------------------------------------
-- First convolutional layer
model:add(nn.SpatialConvolutionMM(1, o.hiddenNodes_conv1, o.kernelSize_conv1, o.kernelSize_conv1))
model:add(nn.SpatialBatchNormalization(o.hiddenNodes_conv1,o.sBNormEps_conv1, o.sBNMomentum_conv1))
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(3, 3, 3, 3))
-- Second convolutional layer
model:add(nn.SpatialConvolutionMM(o.hiddenNodes_conv1, o.hiddenNodes_conv2, o.kernelSize_conv2, o.kernelSize_conv2))
model:add(nn.SpatialBatchNormalization(o.hiddenNodes_conv2, o.sBNormEps_conv2, o.sBNMomentum_conv2))
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
-- First fully-connected layer
model:add(nn.Reshape(o.hiddenNodes_conv2*9))
model:add(nn.Linear(o.hiddenNodes_conv2*9, o.hiddenNodes_fc1))
model:add(nn.LeakyReLU(o.leakyReLU_fc1, true))
-- -- Second fully-connected layer
model:add(nn.Linear(o.hiddenNodes_fc1, o.hiddenNodes_fc2))
model:add(nn.LeakyReLU(o.leakyReLU_fc2, true))

model:add(nn.Linear(o.hiddenNodes_fc2, classes))
-- cudnn.convert(model, cudnn)
model:apply(function(m)
   if m.setMode then m:setMode(1, 1, 1) end
end)
------------------------------------------------------------
criterion = nn.CrossEntropyCriterion()

model = model:cuda()
criterion = criterion:cuda()


model.modules[1].weight:normal(0, o.std_conv1)
model.modules[1].bias:zero()

model.modules[5].weight:normal(0, o.std_conv2)
model.modules[5].bias:zero()

model.modules[10].weight:normal(0, o.std_fc1)
model.modules[10].bias:zero()

model.modules[12].weight:normal(0, o.std_fc2)
model.modules[12].bias:zero()

parameters, gradParameters = model:getParameters()

logger.outfile = paths.concat(o.save, o.experimentId..'.log')
logger.logToConsole = false

train = function(dataset)

  local batchLoss = 0 
  local b = 0
  while b ~= dataset.size do
    
    local currentBatchSize = math.min(o.batchSize, dataset.size - b)
    -- create new batch
    local inputs = torch.CudaTensor(currentBatchSize, 1, imgDim, imgDim)
    local targets = torch.CudaTensor(currentBatchSize)

    -- load batch with data
    local j = 1
    for i = b + 1, math.min(b + o.batchSize, dataset.size) do
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
      learningRate = o.learningRate,
      momentum = o.momentum, 
      learningRateDecay = o.learningRateDecay, 
      weightDecay = o.weightDecay
    }

    optim.sgd(feval, parameters, sgdState)
    
    b = b + currentBatchSize
    if o.showProgress then
      xlua.progress(math.floor(b/o.batchSize),math.floor(dataset.size/o.batchSize))
    end
  end
  
  if o.showProgress then
    batchLoss = batchLoss/(math.floor(dataset.size/o.batchSize)+1)
    print('Epoch: '..epoch..' mean loss: ' .. batchLoss)
  end
end

test = function(dataset, setName)

  local correct = 0
  local b = 0
  while b ~= dataset.size do

    local currentBatchSize = math.min(o.batchSize, dataset.size - b)

    local inputs = torch.CudaTensor(currentBatchSize, 1, imgDim, imgDim)
    local targets = torch.CudaTensor(currentBatchSize)

    local j = 1
    for i = b + 1, math.min(b + o.batchSize, dataset.size) do
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
    if o.showProgress then
      xlua.progress(b, dataset.size)
    end
    
  end

  errorRate = 1 - (correct / dataset.size)

  if o.showProgress then
    print('Epoch: '..epoch..' '..setName..' error: ' .. tostring(errorRate * 100))
  end

  return errorRate
end

epoch = 1
while epoch <= o.maxEpochs do
  train(trainset)
  epoch = epoch + 1
end

epoch = epoch - 1
local result = test(validationset, 'validation')
local testResult = test(testset, 'test')
logger.info(o)
logger.info('validation: '..tostring(result)..'\t test: '..testResult)
print(tostring(result)..'###'..tostring(testResult))
