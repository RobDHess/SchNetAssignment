# SchNetAssignment
Hi, this is a quick implementation of SchNet for a protein dataset. It uses Facebook's ESM2 model to create node features based on the amino acid sequence. In the nn folder you can find the implemented SchNet architecture. I have added some customisability, for example to the architecture of the mlp that parameterises the depth-separable convolution. 

You can run the code using ```python main.py```. This will run the base architecture described in the original paper. See the ```main.py``` file for optional arguments. 

The conda environment can be found in ```environment.yml```. Let me know if you have any questions. 

