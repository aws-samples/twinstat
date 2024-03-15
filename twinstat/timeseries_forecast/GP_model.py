# -*- coding: utf-8 -*-
######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: MIT-0                                     #
######################################################################


import torch
import gpytorch
import matplotlib.pyplot as plot
import numpy as np

# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
scaler = torch.cuda.amp.GradScaler()

# class GPDataset(Dataset):
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y

#     def __len__(self):
#         return self.x.shape[0]

#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx]


class PhysicsKernel_CrackGrowth(gpytorch.means.mean.Mean):
    def __init__(self, initial_crack_length,
                       periodic_load=0.05,
                       C_constraint = None,
                       m_constraint = None,
                       batch_shape=torch.Size(), **kwargs):
        '''
        This object serves as an example of including a physics function in the
        Gaussian Process.  Pytorch will optimize the registered parameters
        including the defined constraints.

        Parameters
        ----------
        initial_crack_length : float
            Initial crack size to be inserted as a constant in Paris equation.
        periodic_load : float, optional
            Cyclic load causing crack propagation. The default is 0.05.
        batch_shape : int, optional
            Size of batches to train on in SGD. The default is torch.Size().

        Returns
        -------
        None.

        '''
        super(PhysicsKernel_CrackGrowth, self).__init__(**kwargs)

        self.a0 = initial_crack_length
        self.sigma = periodic_load

          # register the raw parameter
        self.register_parameter(name="raw_C", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))
        self.register_parameter(name="raw_m", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))

        # register the constraint
        if C_constraint is None:
            C_constraint = gpytorch.constraints.Positive()
        elif isinstance(C_constraint,tuple):
            C_constraint = gpytorch.constraints.Interval(C_constraint[0],
                                                         C_constraint[1])

        if m_constraint is None:
            m_constraint = gpytorch.constraints.Positive()
        elif isinstance(m_constraint,tuple):
            m_constraint = gpytorch.constraints.Interval(m_constraint[0],
                                                         m_constraint[1])

        self.register_constraint("raw_C", C_constraint)
        self.register_constraint("raw_m", m_constraint)

    # now set up the 'actual' paramter
    @property
    def C(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_C_constraint.transform(self.raw_C)

    @C.setter
    def C(self, value):
        return self._set_C(value)

    def _set_C(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_C)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_C=self.raw_C_constraint.inverse_transform(value))


    @property
    def m(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_m_constraint.transform(self.raw_m)

    @m.setter
    def m(self, value):
        return self._set_m(value)


    def _set_m(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_m)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_m=self.raw_m_constraint.inverse_transform(value))

    def forward(self, x):
        '''
        Using the Paris equation for crack propgation under a cyclic
        load.

        Parameters
        ----------
        x : cycles of fatigue

        Returns
        -------
        at : crack size at time t

        '''
        a0 = self.a0
        m = self.m
        C = self.C
        sigma = self.sigma

        #rename to N for cycles, for clarity
        N = x

        at_1 = N.matmul(C * (1 - m/2) * (sigma *np.sqrt(np.pi))**m  )
        at_1 += a0**(1-m/2)

        at_1 = torch.abs(at_1)
        at = torch.pow(at_1, 2/(2-m)   )

        return at


#----------------------------------------------------------------------------------

class _ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,
                 mean_function='constant',
                 periodic=False,
                 lengthscale_constraint=None,
                 lengthscale_direction='greaterthan',
                 outputscale_constraint=None,
                 periodic_period_constraint=None,
                 periodic_lengthscale_constraint=None,
                 base_kernel='RBF'):
        super(_ExactGPModel, self).__init__(train_x, train_y, likelihood)

        if mean_function =='constant':
            self.mean_module = gpytorch.means.ConstantMean()
        elif mean_function =='linear':
            if len(train_x.shape) > 1:
                 self.mean_module = gpytorch.means.LinearMean(train_x.shape.shape[1])
            else:
                 self.mean_module = gpytorch.means.LinearMean(1)
        else:
            self.mean_module = mean_function

        if lengthscale_constraint is not None:
            if lengthscale_direction =='greaterthan':
                lengthscale_constraint= gpytorch.constraints.GreaterThan(lengthscale_constraint)
            else:
                lengthscale_constraint= gpytorch.constraints.LessThan(lengthscale_constraint)
        if outputscale_constraint is not None:
            outputscale_constraint= gpytorch.constraints.GreaterThan(outputscale_constraint)

        if base_kernel.lower() == 'rbf':
            kernel = gpytorch.kernels.RBFKernel(
                                                lengthscale_constraint = lengthscale_constraint
                                                )
        elif base_kernel.lower() == 'matern':
            kernel = gpytorch.kernels.MaternKernel(
                                                    #TODO: user input
                                                    nu=2.5,
                                                    lengthscale_constraint = lengthscale_constraint
                                                  )
        else:
            if periodic:
                kernel = None
            else:
                raise ValueError(f"ERROR: unsupported kernel type {base_kernel}")

        if periodic:
            if periodic_period_constraint is not None:
                if isinstance(periodic_period_constraint, tuple):
                    periodic_constraint = gpytorch.constraints.Interval(periodic_period_constraint[0],
                                                                        periodic_period_constraint[1])
                else:
                    raise ValueError("ERROR: Periodic constraint must be a 2x1 tuple.")

            if periodic_lengthscale_constraint is not None:
                if isinstance(periodic_period_constraint, tuple):
                    periodic_lengthscale_constraint = \
                        gpytorch.constraints.Interval(periodic_lengthscale_constraint[0],
                                                      periodic_lengthscale_constraint[1])
                else:
                    raise ValueError("ERROR: Periodic constraint must be a 2x1 tuple.")

            kernel+=gpytorch.kernels.PeriodicKernel(
                                        period_length_constraint=periodic_constraint,
                                        lengthscale_constraint=periodic_lengthscale_constraint
                                        )

        self.covar_module = gpytorch.kernels.ScaleKernel(kernel,
                                            outputscale_constraint=outputscale_constraint
                                            )
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



class _SpectralMixtureGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_mixtures = 4, lengthscale_constraint=None):
        super(_SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.LinearMean(1)

        if lengthscale_constraint is not None:
            lengthscale_constraint= gpytorch.constraints.GreaterThan(lengthscale_constraint)

        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(
                                                                num_mixtures=n_mixtures,
                                                                lengthscale_constraint = lengthscale_constraint)
        self.covar_module.initialize_from_data(train_x, train_y)
    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPModel(object):
    '''

    Using gpytorch as the backend solver, this object constructs a
    Gaussian Process of varying levels of complexity, which includes
    incorporation of physics objects to both resolve the covaraince
    and calibrate the model inputs.

    The main goal here is bring scikit ease of use to the powerful scaling
    and flexibility of gpytorch.


    '''
    def __init__(self, train_x:np.array, train_y:np.array, lr:float=0.1,
                 model_type:str = 'basic',
                 mean_function:str = 'constant',
                 likelihood_function:str = 'gaussian',
                 lengthscale_constraint:float = None,
                 lengthscale_direction:str = 'greaterthan',
                 outputscale_constraint:float = None,
                 periodic_period_constraint:float = None,
                 periodic_lengthscale_constraint:float = None,
                 likelihood_noise_constraint:float = 1e-4,
                 use_cholesky:bool=True,
                 periodic:bool=False,
                 n_mixtures:int = 4,
                 base_kernel:str='RBF',
                 train_auto_stop:bool = True,
                 auto_stop_tol:float = 1e-5,
                 early_stoppage_iterations:int = 100,
                 gpu:bool = False):
        '''


        Parameters
        ----------
        train_x : np.array
        train_y : np.array
        lr : float, optional
            Learning rate used in SGD. The default is 0.1.
        model_type : str, optional
            basic or spectral_mixing. The default is 'basic'.
        mean_function : str, optional
            'constant','linear', or custom object such as the 'PhysicsKernel_CrackGrowth'.
            The default is 'constant'.
        likelihood_function : str, optional
            Only supports gaussian at the moment. The default is 'gaussian'.
        lengthscale_constraint : float, optional
            Constraint used in the lengthscale of the kernel. The default is None.
        outputscale_constraint : float, optional
            Constraint used in the prediction output. The default is None.
        periodic_period_constraint : float, optional
            Constraint used in the period of the periodic kernel if present. The default is None.
        periodic_lengthscale_constraint : float, optional
            Constraint used in the lengthscale of the periodic kernel if present. The default is None.
        likelihood_noise_constraint : float, optional
            Constraint used in the magnitude of the noise adder to the covariance matrix. The default is 1e-4.
        use_cholesky : bool, optional
            Use cholesky decompisition during training. The default is True.
        periodic : bool, optional
            If true add a periodic kernel to the base_kernel. The default is False.
        n_mixtures : int, optional
            The default is 4.
        base_kernel : str, optional
            Base kernel to use in the covariance matrix. The default is 'RBF'.
        train_auto_stop : bool, optional
            If true, stop training after early_stoppage_iterations of no improvement. The default is True.
        auto_stop_tol : float, optional
            If absolute change in loss is below this value, stop traning. The default is 1e-5.
        early_stoppage_iterations : int, optional
            If during training the loss does not improve within this many iterations, stop training. The default is 100.

        Returns
        -------
        None.

        '''

        if not isinstance(train_x, torch.Tensor):
            train_x = torch.Tensor(train_x)
        if not isinstance(train_y, torch.Tensor):
            train_y = torch.Tensor(train_y)

        # initialize likelihood and model
        #TODO: come back to this later if you want discrete distributions
        #if likelihood_function == 'gaussian':

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
                            noise_constraint=gpytorch.constraints.GreaterThan(likelihood_noise_constraint))

        if model_type == 'basic':
            self.model = _ExactGPModel(train_x, train_y, self.likelihood,
                                      mean_function=mean_function,
                                      periodic=periodic,
                                      lengthscale_constraint=lengthscale_constraint,
                                      lengthscale_direction=lengthscale_direction,
                                      outputscale_constraint=outputscale_constraint,
                                      periodic_period_constraint=periodic_period_constraint,
                                      periodic_lengthscale_constraint=periodic_lengthscale_constraint,
                                      base_kernel=base_kernel
                                     )
        elif model_type == 'spectral_mixture':
            self.model = _SpectralMixtureGPModel(train_x, train_y, self.likelihood, n_mixtures = n_mixtures )
        else:
            raise ValueError("ERROR: Unsupported model type.")


        self.lr = lr
        self.train_x = train_x
        self.train_y = train_y
        self.model_type = model_type
        self.use_cholesky = use_cholesky
        self.base_kernel = base_kernel
        self.train_auto_stop = train_auto_stop
        self.auto_stop_tol = auto_stop_tol
        self.early_stoppage_iterations = early_stoppage_iterations
        self.gpu = gpu

        # Use the adam optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)


    def _iterate_training(self, training_iter):

        early_stop_cnt = 0
        best_min = 1e9
        for i in range(training_iter):
            # Zero gradients from previous iteration
            self.optimizer.zero_grad()
            # Output from model
            #with torch.cuda.amp.autocast():
            output = self.model(self.train_x)
            # Calc loss and backprop gradients
            loss = -self.mll(output, self.train_y)
            loss.backward()
            #scaler.scale(loss).backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
            self.losses.append(loss.item())
            self.optimizer.step()
            #scaler.step(self.optimizer)
            #scaler.update()

            if self.train_auto_stop:
                if len(self.losses) > 2:
                    change = self.losses[-1] - self.losses[-2]
                    if abs(change) < self.auto_stop_tol:
                        break

            early_stop_cnt+=1
            if loss.item() < best_min:
                best_min = loss.item()
                early_stop_cnt = 0
            if early_stop_cnt > self.early_stoppage_iterations:
                break


    def train(self, training_iter:int
              #batch_size:int = 32
              ) -> None:
        '''

        Parameters
        ----------
        training_iter : int
            Number of training iterations to perform

        Returns
        -------
        None

        '''

        if self.gpu:
            self.model = self.model.to(0)

            #TODO: gpytorch not working with multi-GPU, need to investigate
            #world_size = torch.cuda.device_count()
            #self.model = torch.nn.DataParallel(self.model, device_ids=list(range(world_size)))

            self.train_x = self.train_x.to(0)
            self.train_y = self.train_y.to(0)

        # ds = GPDataset(self.train_x, self.train_y)
        # self.dataloader = DataLoader(
        #                             dataset=ds,
        #                             batch_size=batch_size,
        #                             shuffle=True
        #                             )


        self.model.train()
        self.likelihood.train()
        self.losses = []

        if not self.use_cholesky:
            with gpytorch.settings.max_cholesky_size(0):
                self._iterate_training(training_iter)
        else:
            self._iterate_training(training_iter)

        self.model.eval()
        self.likelihood.eval()

        #determine model average performance
        observed_pred = self.get_estimate(self.train_x)
        r = np.array(self.train_y.cpu() - observed_pred.mean.cpu())
        self.mse = np.var(r)
        self.rmse = np.sqrt(self.mse)
        self.rmse_rel = self.rmse / np.mean(np.array(self.train_y.cpu()))

    def update_training_data(self, newX:np.array, newy:np.array, replace:bool=False) -> None:
        '''
        Update the training data stored within the object.  When training,
        the GP will use this data to refine the fitting parameters. Can
        be used to refine existing models instead of retraining from
        scratch.

        Parameters
        ----------
        newX : np.array
        newy : np.array
        replace : bool, optional
            Append to the existing training set or replace.
            The default is False.

        Returns
        -------
        None

        '''

        if not isinstance(newX, np.ndarray):
            newX = np.array(newX)
        if not isinstance(newy, np.ndarray):
            newy = np.array(newy)

        if replace:
            self.train_x = torch.Tensor(newX)
            self.train_y = torch.Tensor(newy)
        else:
            self.train_x = torch.Tensor(np.append(self.train_x, newX))
            self.train_y = torch.Tensor(np.append(self.train_y, newy))
        self.model.set_train_data(self.train_x,self.train_y,strict=False)

    def get_estimate(self, X:np.array, uncertainty:str = 'confidence') -> object:
        '''

        Parameters
        ----------
        X : np.array
        uncertainty : str, optional
            Either sets the uncertainty to be based
            on the confidence (95%) of the mean prediction or the
            prediction interval (95%), which bounds were we would expect
            to see a newly measured value. The default is 'confidence'.

        Returns
        -------
        observed_pred : object
            gpytorch object that includes the attributes/methods:

                confidence_region
                mean
                variance
            Example:
                observed_pred = GP.get_estimate(X, uncertainty='confidence')

                # Get upper and lower confidence bounds
                lower, upper = observed_pred.confidence_region()

                # Get the expectation
                mean = observed_pred.mean.numpy()

                # Get the variance
                mean = observed_pred.variance.numpy()
        '''

        if not isinstance(X, torch.Tensor):
            X = torch.Tensor(X)

        if self.gpu:
            X = X.to(0)

        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if uncertainty == 'confidence':
                observed_pred = self.model(X)
            else:
                observed_pred = self.likelihood(self.model(X))
        return observed_pred

    def plot(self, X:np.array,y:np.array=None, uncertainty:str = 'confidence') -> None:
        '''
        Generate figures of the GP performance with the X,y data.

        Parameters
        ----------
        X : np.array
        y : np.array, optional
             The default is None.
        uncertainty : str, optional
            confidence or prediction. The default is 'confidence'.

        Returns
        -------
        None

        '''

        if not isinstance(X, torch.Tensor):
            X = torch.Tensor(X)
        if y is not None and not isinstance(y, torch.Tensor):
            y = torch.Tensor(y)

        with torch.no_grad():
            # Initialize plot
            f, ax = plot.subplots(1, 1, figsize=(12,5))

            observed_pred = self.get_estimate(X, uncertainty=uncertainty)

            # Get upper and lower confidence bounds
            lower, upper = observed_pred.confidence_region()
            # Plot training data as black stars

            if y is not None:
                ax.scatter(X.cpu().numpy(), y.cpu().numpy(), s=40, facecolors='none', edgecolors='black')

            # Plot predictive means as blue line
            ax.plot(X.cpu().numpy(), observed_pred.mean.cpu().numpy(), 'b')
            # Shade between the lower and upper confidence bounds
            ax.fill_between(X.cpu().numpy(), lower.cpu().numpy(),
                            upper.cpu().numpy(), alpha=0.5)
            #ax.legend(['Observed Data', 'Mean', 'Confidence'])
            if uncertainty == 'confidence':
                ax.legend(['GP Mean', '95% Confidence\nInterval'])
            else:
                ax.legend(['GP Mean', '95% Prediction\nInterval'])

    def print_parameters(self):
        '''
        Print to screen the raw GP parameters,
        the constraints on each of these parameters,
        and the transformed actual values of these
        parameters.

        Returns
        -------
        None.

        '''
        params = {}
        print("Raw Parameters: ")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print (name, param.data)
                params[name] = param.data

        print("\nParameter Constraints: ")
        constraints = {}
        for name, constraint  in self.model.named_constraints():
            print (name, constraint )
            constraints[name] = constraint

        print("\nTransformed Parameters: ")
        for name, constraint  in constraints.items():
            name = name.replace('_constraint','')
            print (name, constraint.transform(params[name]) )



#----------------------------------------------------------
#%% main, GP examples
if __name__ == '__main__':

    n_data = 100
    x = np.linspace(0,1000,num=n_data)
    y_org = 0.001*x
    #gaussian noise
    np.random.seed(0)
    y = y_org + np.random.normal(0, 0.1, size=n_data)

    GP = GPModel( x, y.ravel(), lr=0.1,
                  base_kernel='matern',
                  #lengthscale_constraint=200,
                  auto_stop_tol=1e-6,
                  mean_function=PhysicsKernel_CrackGrowth(0.01),
                  gpu=False
                  )
    GP.train(50000)

    GP.plot(GP.train_x, GP.train_y)
    GP.print_parameters()