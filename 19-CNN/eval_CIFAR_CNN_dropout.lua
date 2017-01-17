require 'nn'
require 'paths'
require 'cunn'
require 'optim'
logger = require "log"
require 'cudnn'
paths.dofile'augmentation.lua'

cudnn.fastest = true
cudnn.benchmark = true

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
   --maxEpochs         (default 100)        Number of epochs to train 
   --mean              (default 0.0)       The mean of init function 
   --drop_rate1        (default 0.5)       Drop
   --drop_rate2        (default 0.5)       Drop
   --drop_rate3        (default 0.5)       Drop
   --drop_rate4        (default 0.5)       Drop
   --seed              (default 2275)      Random seed number
   --experimentId      (default "ad_hock") Experiment ID, used to create unique log
   --save              (default "logs")    Log folder number
   --dataset           (default  "./datasets/cifar10_train-val-white.t7") 
   --nThreads          (default 12)
   --gen               (default "gen")
   --showProgress                         Output progress to console
]]


torch.manualSeed(o.seed)
torch.setnumthreads(12)
cutorch.manualSeed(o.seed)

logger.outfile = paths.concat(o.save, o.experimentId..'.log')
logger.logToConsole = false

local provider = torch.load(o.dataset)
classes = provider.testData.labels:max()
imgDim = 32


model = nn.Sequential()
------------------------------------------------------------
-- First convolutional layer
model:add(nn.SpatialConvolutionMM(3, o.hiddenNodes_conv1, o.kernelSize_conv1, o.kernelSize_conv1))
model:add(nn.SpatialBatchNormalization(o.hiddenNodes_conv1,o.sBNormEps_conv1, o.sBNMomentum_conv1))
model:add(nn.ReLU(true))
model:add(nn.Dropout(o.drop_rate1))
model:add(nn.SpatialMaxPooling(3, 3, 3, 3))

-- Second convolutional layer
model:add(nn.SpatialConvolutionMM(o.hiddenNodes_conv1, o.hiddenNodes_conv2, o.kernelSize_conv2, o.kernelSize_conv2))
model:add(nn.SpatialBatchNormalization(o.hiddenNodes_conv2, o.sBNormEps_conv2, o.sBNMomentum_conv2))
model:add(nn.ReLU(true))
model:add(nn.Dropout(o.drop_rate2))
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

-- First fully-connected layer
model:add(nn.Reshape(o.hiddenNodes_conv2*9))
model:add(nn.Linear(o.hiddenNodes_conv2*9, o.hiddenNodes_fc1))
model:add(nn.LeakyReLU(o.leakyReLU_fc1, true))
model:add(nn.Dropout(o.drop_rate3))

-- -- Second fully-connected layer
model:add(nn.Linear(o.hiddenNodes_fc1, o.hiddenNodes_fc2))
model:add(nn.LeakyReLU(o.leakyReLU_fc2, true))
model:add(nn.Dropout(o.drop_rate4))
model:add(nn.Linear(o.hiddenNodes_fc2, classes))
cudnn.convert(model, cudnn)
model:apply(function(m)
   if m.setMode then m:setMode(1, 1, 1) end
end)
------------------------------------------------------------
criterion = nn.CrossEntropyCriterion()

model = model:cuda()
criterion = criterion:cuda()


model.modules[1].weight:normal(0, o.std_conv1)
model.modules[1].bias:zero()

model.modules[6].weight:normal(0, o.std_conv2)
model.modules[6].bias:zero()

model.modules[12].weight:normal(0, o.std_fc1)
model.modules[12].bias:zero()

model.modules[15].weight:normal(0, o.std_fc2)
model.modules[15].bias:zero()

parameters, gradParameters = model:getParameters()

function f(inputs, targets)
   model:forward(inputs)
   local loss = criterion:forward(model.output, targets)
   local df_do = criterion:backward(model.output, targets)
   model:backward(inputs, df_do)
   return loss
end


function train()
  model:training()

  local targets = torch.CudaTensor(o.batchSize)
  local indices = torch.randperm(provider.trainData.data:size(1)):long():split(o.batchSize)
  -- remove last batch that may not have exactly batchSize elements
  indices[#indices] = nil

  local loss = 0

  local start = 1
  for t,v in ipairs(indices) do
    if o.showProgress then xlua.progress(start, #indices) end
    local inputs = provider.trainData.data:index(1,v):cuda()
    targets:copy(provider.trainData.labels:index(1,v))

    sgdState = sgdState or {
      learningRate = o.learningRate,
      momentum = o.momentum, 
      learningRateDecay = o.learningRateDecay, 
      weightDecay = o.weightDecay
    }

    optim.sgd(function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      loss = loss + f(inputs, targets)
      return f,gradParameters
    end, parameters, sgdState)
    start = start+1
  end

  return loss / #indices
end

function testError()
  model:evaluate()
  local confusion = optim.ConfusionMatrix(classes)
  local data_split = provider.testData.data:split(o.batchSize,1)
  local labels_split = provider.testData.labels:split(o.batchSize,1)

  for i,v in ipairs(data_split) do
    confusion:batchAdd(model:forward(v:cuda()), labels_split[i])
  end

  confusion:updateValids()
  return (1-confusion.totalValid) * 100
end
function valError()
  model:evaluate()
  local confusion = optim.ConfusionMatrix(classes)
  local data_split = provider.valData.data:split(o.batchSize,1)
  local labels_split = provider.valData.labels:split(o.batchSize,1)

  for i,v in ipairs(data_split) do
    confusion:batchAdd(model:forward(v:cuda()), labels_split[i])
  end

  confusion:updateValids()
  return (1-confusion.totalValid) * 100
end

epoch = 1
while epoch <= o.maxEpochs do
  local loss = train()
  if o.showProgress then 
     print('Epoch:', epoch, 'Loss:', loss)
     print('Val error:',valError())
  end
  epoch = epoch + 1
end

epoch = epoch - 1
local val_error = valError()
local test_error = testError()
local resultMsg = 'validation: '..tostring(val_error)..'\t test: '..test_error
logger.info(o)
logger.info(resultMsg)
print(val_error..'###'..test_error)
