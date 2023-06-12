# twinstat

[![pipeline status](https://gitlab.aws.dev/twinstat)](https://gitlab.aws.dev/twinstat/badges/main/pipeline.svg)

A data science python library (TwinStat) is needed as a toolbox to be used by a variety of other AWS services ranging from Twinmaker, do-pm, SimSpace Weaver, SageMaker, etc.  TwinStat was designed as a toolbox for the TwinFlow Level 4 Digital Twin Framework. The leveling framework can be read about here: https://aws.amazon.com/blogs/iot/digital-twins-on-aws-unlocking-business-value-and-outcomes/


## Features

TwinStat includes the following methods:

* Bayesian Estimation and Posterior Distribution Calculation
  - Standard Kalman Filters
  - Smoothing Kalman Filters
  - Adaptive Kalman Filters
  - Unscented Kalman Filters
  - Particle Filters
* Global Heuristic Optimization
    - Genetic Algorithm 
* Sensitivity Analysis
    - Using shapely sensitivities 
* Uncertainty Propagation 
  - polynomial chaos expansion
  - brute monte carlo
* Time Series Analysis
  - AutoArch
  - Auto-regressive neural networks
    - feedforward architecture generation
    - both mean and quantiles
* Quantile K-Nearest Neighbor
* Outlier removal
* Linear Regressions (CPU / GPU)
* AutoML wrappers around (AutoGluon)
* Probablistic IoT deviation checks
  - Example: determine if incoming IoT data is no longer supported by an existing training set
* Gaussian Process Models
  - includes templates for using physics based mean functions
  - update functionality not requiring retraining for streaming data

### Requirements: 

- Supported Operating Systems: Linux, Windows
- Python 3.10+

## Installation

```
git clone git@ssh.gitlab.aws.dev:autonomouscomputesateam/twinstat.git
cd twinstat/dist
pip install ./*.whl

```

## API Documentation

Auto-documentation can be found here:

https://gitlab.aws.dev/autonomouscomputesateam/twinstat/-/tree/main/docs/_build/html

Users can:

```
git clone git@ssh.gitlab.aws.dev:autonomouscomputesateam/twinstat.git
cd twinstat/docs/_build/html
```
View the index.html to review API documentation.

## Example

Full tutorials to be published on AWS Samples in Q4 2023.

Short example:
```
from twinstat.ts_forecast.AR_NN_models import AR_quantile_neural_network
ANN = AR_quantile_neural_network(AR=ar,
                                 n_exog_variables= n_exo,
                                 hidden_units=16,
                                 tau=safety_bias,
                                 include_endog=False)
ANN.train(var, X=X, patience=50)
print("Training complete")
ANN.save_model(weights_path + '/corrector_weights')
print("model and figures saved")
ANN.plot(save_fig=True)
```

## License

This repository is released under the MIT-0 License. See the LICENSE file for details.

## Credits

This open source framework was developed by the Autonomous Computing Team within Amazon Web Services (AWS) Worldwide Specialist Organization (WWSO). Developers include Ross Pivovar, Satheesh Maheswaran, Vidyasagar Ananthan, and Cheryl Abundo. Authors would like to thank Alex Iankoulski for his detailed guidance and expertise in reviewing the code.
