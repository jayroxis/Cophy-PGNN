# Learning Neural Networks with Competing Physics Objectives: An Application in Quantum Mechanics

This is the supplementary codes for arxiv paper *Learning Neural Networks with Competing Physics Objectives: An Application in Quantum Mechanics*: https://arxiv.org/abs/2007.01420.

**IMPORTANT NOTE:** 
Pretrained models and result files have been omitted in this repository due to the space limits of Github. Particularly the pretrained models and stored variables are too big for gradient analysis, but meanwhile they are essential for gradient analysis to run. A complete project can be found by navigating to the anonymous link: https://osf.io/ps3wx/?view_only=9681ddd5c43e48ed91af0db019bf285a (cophy-pgnn.tar.gz). 
However, you can still be able to inspect the code and notebooks where we have the results displayed. Since the dataset is included, you are also able to run the experiments and generate new results for evaluations.

---------------------------------------------------------------
The project is written in *Python* and algorithms are implemented using *PyTorch*.

The structure of the folders

- **datasets**: where the datasets are placed.
- **notebooks**: where jupyter/ipython notebooks are placed.
- **scripts**: where most of the python scripts are placed.
- **models**: where by default saved models (PyTorch *state_dict*) are (temporary) saved.
- **results**: where by default generated losses *.csv* and results *.txt* files are (temporary) saved.
- **logs**: where by default generated log *.txt* files are (temporary) saved.
- **gradients**: where the results of gradient anaysis experiments should be placed.
- **hyper_saved**: where the results of hyperparameter search experiments should be placed.
- **eval_saved**: where the results for evaluation experiments should be placed.
- **figures**: where by default the generated figures will be placed.
- **loss_surface_vis**: where all the files needed for loss landscape visualization are placed.

For details, check the *README.md* file in each directory.

-----------------------------------------------------------------
# Instructions

Tested OS: *Ubuntu 18.04 LTS*.

Since almost all figures are generated using ***Matplotlib*** with ***Latex*** plugin, please make sure a running ***LaTex*** install on your machine before running any evaluations.
Please follow the instructions to install ***MikTex***:
https://miktex.org/download

Or a safer way is to install ``texlive-full``.

For GPU acceleration, please check you have installed CUDA version >= 10.1 and have GPU devices that support CUDA.

To help you understand the stucture of the repository, here we provide a simple example of usage:
- Go to *notebooks* folder and read the *README.md*. Set up parameters (or use default settings as provided), and based on the desired experiments, pick a correspoding notebook to run. (Description can be found in the *notebooks* folder).
- When finished, go to root directory and the logs, results and trained models can be found in the *logs*, *results* and *models* folder. Move the contents to where you want to save and organize them properly, e.g., *eval_saved* and *hyper_saved* are two examples of how we store and organize the results. You may also want to do the same for your own set of experiments.
- When you move the results, logs etc to where you want to save them (for long-term use). Then set up evaluation pipelines, e.g., you can find the notebooks for evaluations in the *notebooks* folder where they read stored results from *eval_saved* and *hyper_saved*. You may also follow the same fashion. Generated figures were saved under the *figures* folder.

Some important notes:
- Pretrained models can be founded in the folders/subfolders named models. However, it may be difficult to identify them just from their file names. To load a pretrained model, you need to find the result text file which contains the information of the model including the path of corresponding trained model.
- Results and pretrained models for hyperparameter search are omitted here due to the huge amount of space needed to store them for a large number of experiments.
- After running any notebooks that generate results for some experiments, the results, trained models and logs (by default) will always be stored in **results**, **models**, and **logs** at the root directory (of this repo). To confirm the results, you need to manually move the contents of the three folders into other folders, e.g., **hyper_saved** and **eval_saved**. This is for more flexible management of the results that can be organized better, and can allow users to reject unexpected results (e.g., generated under wrong settings) before they mixed with the 'good' results.
- Most folders under the root directory of this repo have README.mds file that provide more information about the files and functions related to the contents in the folder.
-----------------------------------------------------------------

**IMPORTANT**: Make sure *pip* is installed on your machine, all codes are tested on *Ubuntu 18.04 LTS*. 

Please install the following dependencies

- *numpy*: `pip install numpy`
- *pandas*: `pip install pandas`
- *tqdm*: `pip install tqdm`
- *matplotlib*: `pip install matplotlib`
- *seaborn*: `pip install seaborn`
- *fastprogress*: `pip install fastprogress`
- *PyTorch*: `pip install torch torchvision`
- *scikit-learn*: `pip install scikit-learn`
- *scipy*: `pip install scipy`
- *jupyterlab*: `pip install jupyterlab`
- *progressbar*: `pip install progressbar`
- *loss-landscapes*: `pip install loss-landscapes`

-----------------------------------------------------------------

We create a naming system that can encode the settings of a model into its alias, which is very different from the names used in the paper.
Here is a table of model information including their alias used in the codes:

|          Models          	|     Alias    	| Training 	| Training 	| Training 	|   Test  	|  Test  	| Adaptive  |
|:------------------------:	|:------------:	|:--------:	|:--------:	|:--------:	|:------:	|:------:	|:------:	|
|          Names          	|     Abbr.    	|    MSE   	|  S-Loss  	|  E-Loss  	| S-Loss 	| E-Loss 	|  Lambda	|
| *CoPhy*-PGNN   	        | cNSE-NNex    	|     +    	|     +    	|     +    	|    +   	|    +   	| c.s. & a. |
| Black-box Neural Network  | NN            |     +    	|     -    	|     -    	|    -   	|    -   	|     -     |
| PINN-*analogue*      	    | NSE-NNex-LF 	|     -    	|     +    	|     +    	|    +   	|    +   	|     -     |
| PGNN-*analogue*   	    | NSE-NNex    	|     +    	|     +    	|     +    	|    +   	|    +   	|     -     |
| *MTL*-PGNN     	        | vNSE-NNex     |     +    	|     +    	|     +    	|    +   	|    +   	|    alt.   |
| *CoPhy*-PGNN (only-D_Tr)  | cNSE-NN       |     +    	|     +    	|     +    	|    -   	|    -   	| c.s. & a. |
| *CoPhy*-PGNN (w/o E-Loss) | cNS-NNex     	|     +    	|     +    	|     -    	|    +   	|    -   	| c.s. & a. |
| *CoPhy*-PGNN (Label-free) | cNSE-NNex-LF  |     -    	|     +    	|     +    	|    +   	|    +   	| c.s. & a. |

'+': presence, '-': absence, 'c.s. & a.': cold-start and annealing, 'alt.': alternating. 

Some additional alias that you may see in the codes:
- 'NN' <=> 'DNN' (the most common one)
- '_{ex}' <=> 'ex'
- 'LB' <=> 'LF'
- 'C-' <=> 'c'
For example, 'C-NSE-DNN_{ex}-LB' <=> 'cNSE-NNex-LF'.
