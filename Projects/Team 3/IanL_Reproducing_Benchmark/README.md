## This Purpose of Having This Folder 
This folder is created to document all the necessary steps, code and data to reproduce the results for the EoE project.


## Code in Jupyter Notebooks



## Data 
- Data is stored on Google Drive. Due to confidential and IP concerns, please contact Kevin Lin(pex7ps), Jason Wang(jyw5hw) for permission to access data.

- If you have successfully obtained the dat access, please have the data stored this way: data_path = "YOUR_DIR/EoE_data/Full_Model/", and the under the Full_Model directory, you should also have "Images/all_imgs" that contains the raw imgaes for training, and "/Masks" that contains masked images.

## Key components or parameters that need to adjusted in order to get the experiments running:

### Environments
- Tensorflow 2.8.0/Keras Py3.9
- A100 and V100 GPUs on Rivanna, other GPUs would not work due to OOM error
- DS--6013 allocation
- RAM: 150Gb

### Important functions/lines of code
- def createModel(input_tensor, bnorm_axis, n_filters, drop_rate, drop_train) #this is the original UNet model, or the basedline model
- def createDilatedModel(input_tensor, bnorm_axis, n_filters, drop_rate, drop_train, dilation) #this is the dilated UNet model, remeber to modify the parameter **dialtion = 2/4/8/16** to meet with your experiment plan.
- def createPyramidDilatedModel(input_tensor, bnorm_axis, n_filters, drop_rate, drop_train) #this is the pyramid dilated model
- def createSeparableModel(input_tensor, bnorm_axis, n_filters, drop_rate, drop_train) #this is the separable model
- def createDepthwiseModel(input_tensor, bnorm_axis, n_filters, drop_rate, drop_train) #this si the depthwise model

### Which model to run? (comment out the model you want to run) 
- model=createSeparableModel(input_tensor, bnorm_axis, n_filters, drop_rate, drop_train)
- model=createModel(input_tensor, bnorm_axis, n_filters, drop_rate, drop_train)
- model=reateDilatedModel(input_tensor, bnorm_axis, n_filters, drop_rate, drop_train, dilation)
- model= createPyramidDilatedModel(input_tensor, bnorm_axis, n_filters, drop_rate, drop_train)
- model=createDepthwiseModel(input_tensor, bnorm_axis, n_filters, drop_rate, drop_train) # Depthwise model is not working

## Important (hyper)parameters 
- epoch = 100
- lr = 0.01
- drop_rate = 0.5
- Batch_size = 4 and 5 (unclear comments)
- dilation = 2, 4, 8, or 16 this only needs to be modified when running the dilated models.
