The code implements experiments on Tiny ImageNet

Running iCaRL:
---------------------------

Versions
Python 3+
PyTorch v0.1.12



First run data_prep.py to download and assemble the dataset
Then run main.py


Running ExpertNet:
---------------------------


Generate data through:

```sh
cd data_utils
python3 data_prep_tin.py
python3 data_prep_mninst.py
cd ../

Generate autoencoders through:

```sh 
python3 generate_models.py 
```

The file takes the following arguments

* ***init_lr***: Initial learning rate for the model. The learning rate is decayed every 5 epochs.**Default**: 0.1 
* ***num_epochs_encoder***: Number of epochs you want to train the encoder model for. **Default**: 5
* ***num_epochs_model***: Number of epochs you want to train the model for. **Default**: 15
* ***batch_size***: Batch Size. **Default**: 16
* ***use_gpu***: Set the GPU flag to ``True`` to use the GPU. **Default**: ``False``

Running MemoryAwareSynapses:
---------------------------

Generate data through

```sh
python3 data_prep.py
```

Run the following command to complete training:

```sh
python3 main.py
```

The file takes the following arguments

* ***use_gpu***: Set the flag to true to train the model on the GPU **Default**: False
* ***batch_size***: Batch Size. **Default**: 8
* ***num_freeze_layers***: The number of layers in the feature extractor (features) of an Alexnet model, that you want to train. The rest are frozen and they are not trained. **Default**: 2
* ***num_epochs***: Number of epochs you want to train the model for. **Default**: 10
* ***init_lr***: Initial learning rate for the model. The learning rate is decayed every 20th epoch.**Default**: 0.001 
* ***reg_lambda***: The regularization parameter that provides the trade-off between the cross entropy loss function and the penalty for changes to important weights. **Default**: 0.01