# Notebooks

----------------------------------------------------

We use **jupyter notebooks** to for better visualization and the users can try out different settings on the code we provided more easily.

The notebooks under this root directory shows examples of how to set up experiments using the resources in the repo.

---------------------------------------------------
## Hyperparameters

We use around 300 runs of random search [1] for hyperparamters search on *lambda_s*, *lambda_e0* and *sigma_e*, then choose top-5 models with lowest MSE on the validation set, and calculate the mean value of each hyperparamter as our final choice. Here are the results:

|          Models          	| Models (Alias)| *lambda_s*| *lambda_e0*   | *sigma_e* |
|--------------------------	|--------------	|----------	|-----------	|----------	|
| *CoPhy*-PGNN   	        | cNSE-DNNex    | 0.433467 	| 2.297982  	| 0.861043 	|
| Black-box Neural Network  | NN          	| N/A      	| N/A       	| N/A      	|
| PINN-*analogue*      	    | NSE-NNex-LF 	| 0.566698 	| 3.680050  	| 0.786220 	|
| PGNN-*analogue*   	    | NSE-NNex    	| 0.433467 	| 2.297982  	| 0.861043 	|
| *MTL*-PGNN     	        | vNSE-NNex     | 0.433467 	| 2.297982  	| 0.861043 	|
| *CoPhy*-PGNN (only-D_Tr)  | cNSE-NN       | 0.274606 	| 3.046513  	| 0.672920 	|
| *CoPhy*-PGNN (w/o E-Loss) | cNS-NNex      | 0.041193 	| N/A       	| N/A      	|
| *CoPhy*-PGNN (Label-free) | cNSE-NNex-LF  | 0.566698 	| 3.680050  	| 0.786220 	|

The hyperparameters for different modes:

<table border="0">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>threshold</th>
      <th>lambda s</th>
      <th>smooth</th>
      <th>epoch</th>
      <th>overlap</th>
      <th>mse</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>quick-start</th>
      <td>14.0</td>
      <td>0.936669</td>
      <td>0.073074</td>
      <td>497.0</td>
      <td>0.994515</td>
      <td>0.003987</td>
    </tr>
    <tr>
      <th>sigmoid</th>
      <td>51.0</td>
      <td>0.846349</td>
      <td>0.171778</td>
      <td>495.0</td>
      <td>0.994536</td>
      <td>0.004112</td>
    </tr>
    <tr>
      <th>inverse-sigmoid</th>
      <td>59.2</td>
      <td>0.939779</td>
      <td>0.020170</td>
      <td>499.0</td>
      <td>0.983904</td>
      <td>0.008927</td>
    </tr>
    <tr>
      <th>quick-drop</th>
      <td>61.2</td>
      <td>0.836881</td>
      <td>0.062851</td>
      <td>499.0</td>
      <td>0.962256</td>
      <td>0.018643</td>
    </tr>
  </tbody>
</table>

-----------------------------------------

## Usage

The following *notebooks* are used to generate experiment results, which by default will be saved at **results**, **logs** and **models**: 
- **Evaluation Experiments**: experiments for different models.
- **Evaluation Experiments - Four Modes.ipynb**: experiments for different adaptive modes.
- **Generate Gradient Data.ipynb**: experiments for analysing the competing loss terms.

By replacing the constant hyperparameters with random variables, and set the running times from 10 to a large number, the above notebooks can be used for hyperparameter search.

Before run any following evaluations, please make sure you have generated results and put into corresponding folders, e.g., *eval_saved*, *gradients*.
- **eval_train_size**: includes a notebook *visualization_train_sizes.ipynb* that comprehensively evaluates the performances of different models.
- **eval_cold_start_modes**: includes a notebook *four-modes.ipynb* that plots the performances using different adaptive strategies.
- **gradient_analysis**: includes a notebook *gradient_analysis.ipynb* that is used for gradient analysis for competing loss terms.
- **others**: includes a notebook *example_of_competing_loss.ipynb* that plots an example figure of competing loss term.

-----------------------------------------
## Reference
[1] J. Bergstra, Y. Bengio, Random search for hyper-parameter optimization (2012), Journal of Machine Learning Research.
