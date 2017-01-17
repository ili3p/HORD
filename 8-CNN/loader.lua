mnist = require 'mnist'
function loadMnist()

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

   return trainset, validationset, testset
end
