# -*- coding: utf-8 -*-
######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: MIT-0                                     #
######################################################################


from scipy.stats import f as Fdist
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
import numpy as np
from collections import Counter
from scipy.stats import multivariate_normal
from tqdm import tqdm
import matplotlib.pyplot as plot

def _hotelling_t2(bgm1, bgm2, dist1, dist2, alpha = 0.05):
    #https://en.m.wikipedia.org/wiki/Hotelling%27s_T-squared_distribution#Two-sample_statistic

    #dist1 = 1
    #dist2 = 0
    xbar = bgm1.means_[dist1]
    ybar = bgm2.means_[dist2]
    cov_x = bgm1.covariances_[dist1]
    cov_y = bgm2.covariances_[dist2]

    nx = bgm1.dof[dist1]
    ny = bgm2.dof[dist2]

    if nx <5 or ny < 5:
        return 1e7, 0.0

    pooled_cov = ( (nx - 1)*cov_x + (ny - 1)*cov_y ) / (  nx + ny - 2  )
    pooled_cov_inv = np.linalg.inv(pooled_cov)
    bar_diff = (xbar - ybar)
    bar_diff = np.expand_dims(bar_diff, -1)

    t2= np.matmul( bar_diff.T , pooled_cov_inv)
    t2 = np.matmul(t2, bar_diff)
    t2 = np.squeeze(t2)
    t2 = nx*ny/(nx+ny) *  t2
    #convert to F
    p = xbar.shape[0]
    Fstar = (nx + ny - p -1)/ ((nx + ny -2)*p) * t2
    Fcrit = Fdist.ppf(1-alpha, p, nx + ny-1-p )

    p_value = 1 - Fdist.cdf(Fstar, p, nx + ny-1-p )

    #print( Fstar, Fcrit )
    return Fstar/Fcrit, p_value


#------------------------------------------------------------------------------------------

def _MC_integrate_subdistributions(bgm1, bgm2, n_samples = 500000):
    '''
    We have a training set distribution bgm1 which is a GMM.

    We have new data in bgm2 being modelled with new GMM. Going to
    check if each subdistribution in bgm2 could have come from bgm1.

    Use monte carlo integration to determine probability of new
    distribution given training distribution.

    Note that scikit is giving an estimate of the probability. The
    MC needs to integrate the probability density.  Thus, the
    multivariate normals from scipy are used instead.

    '''

    P_given_f = {}
    N = n_samples
    nmm2 = bgm2.means_.shape[0]
    for dist in tqdm(range(nmm2)):
        #N=500000
        samples = multivariate_normal.rvs( mean=bgm2.means_[dist],
                                           cov = bgm2.covariances_[dist],
                                           size=N)
        #sample most likely came from this parent GMM sub dist
        component_responsibility = bgm1.predict_proba(samples)
        component_responsibility = np.sum(component_responsibility, axis=0)
        dist_parent = np.argmax( component_responsibility )

        #get parent density-----

        # when dealing with large number of variables, numerical stability
        # becomes an issue due to precision.  The pdf needs to have a
        # symmetric positive definite matrix, but the eigenvalues become very
        # very slightly negative in some large variable cases.  The solution here
        # is to attempt to slightly increase the variances in an attempt to
        # resolve the numerical issues.  We should maybe consider a for loop that
        # continuously increases the variance to find a stable point.  However,
        # this is not the best idea in that we are deviating from the measured covariance
        # of the data.
        A = bgm1.covariances_[dist_parent]
        B = bgm2.covariances_[dist]
        xbar_diff = bgm1.means_[dist_parent] - bgm2.means_[dist]

        min_eig_A = np.min(np.real(np.linalg.eigvals(A)))
        min_eig_B = np.min(np.real(np.linalg.eigvals(B)))

        for _ in range(30):
            try:
                f = multivariate_normal.pdf(samples,
                                            mean= bgm1.means_[dist_parent],
                                            cov = A
                                            )
                #print("success")
                break
            except np.linalg.LinAlgError:
                #print("failed")
                A += 10*min_eig_A * np.eye(*A.shape)


        for _ in range(30):
            try:
                g = multivariate_normal.pdf(samples,
                                        mean= bgm2.means_[dist],
                                        cov = B
                                        )
                #print("success")
                break
            except np.linalg.LinAlgError:
                #print("failed")
                B += 10*min_eig_B * np.eye(*B.shape)


        P = np.mean(f/(g+1e-10))
        sigmaP = np.sqrt( np.var(f/ (g+1e-10) ) / N )
        P_given_f[dist] = [P, sigmaP, N, xbar_diff]

    return P_given_f

#------------------------------------------------------------------------------------------

def distribution_difference_MC_test(X:np.array, Y:np.array,
                                    n_mixtures_X:int = 50,
                                    n_mixtures_Y:int = 50,
                                    gmm_type:str = "standard",
                                    n_samples:int=500000) -> dict and            \
                                                             GaussianMixture and \
                                                             GaussianMixture:
    '''

    Fit a Gaussian Mixture Model to the input data sets.  Then use Monte Carlo
    integration to determine the volume of overlap in probablity space. Small overlap
    volumes indicate there is little evidence that Y was generated from
    the same distribution as X.

    Large dimensional problems will require normalization of the results due to overall
    parameter space volume.  Example: if we have X data for 50 IoT data streams, if a
    few points are taken from X and used as Y, due to the vastness of the hyper-volume
    when the overlap is integreated, the result is a small value.  Therefore, when using
    this algorithm, we are looking for small values on the order of 1e-9, not standard
    statistical inferrencing values such as 0.01.


    Parameters
    ----------
    X : np.array
        Data set 1 (n data X n features)
    Y : np.array
        Data set 2 (n data X n features)
    n_mixtures_X : int, optional
        If using a standard Gaussian Mixture Model, the number of clusters
        must be inputed.  When using "bayesian", this is the max clusters that could be used.
        The default is 50.
    n_mixtures_Y : int, optional
        If using a standard Gaussian Mixture Model, the number of clusters
        must be inputed.  When using "bayesian", this is the max clusters that could be used.
        The default is 50.
    gmm_type : str, optional
        "standard" or "bayesian".
        The default is "standard".
    n_samples : int, optional
        Number of monte carlo samples to use in the integration. The default is 500000.

    Returns
    -------
    dict and GaussianMixture and GaussianMixture
        The dictionary includes the pvalues for each clustered distribution, in which
        a large p-value means it is likely data from Y came from X and a low pvalue
        means there is little evidence that Y came from X.

    '''

    if gmm_type.lower() =="bayesian":
        print("Fitting BGMM 1")
        bgm1 = BayesianGaussianMixture(n_components=n_mixtures_X, verbose = 2,
                                        random_state=0, max_iter=5000).fit(X)
        print("Fitting BGMM 2")
        bgm2 = BayesianGaussianMixture(n_components=n_mixtures_Y, verbose = 2,
                                        random_state=0, max_iter=5000).fit(Y)
    else:
        print("Fitting BGMM 1")
        bgm1 = GaussianMixture(n_components=n_mixtures_X, verbose = 2,
                                        random_state=0, max_iter=5000).fit(X)
        print("Fitting BGMM 2")
        bgm2 = GaussianMixture(n_components=n_mixtures_Y, verbose = 2,
                                        random_state=0, max_iter=5000).fit(Y)

    P_given_f = _MC_integrate_subdistributions(bgm1, bgm2, n_samples = n_samples)

    return P_given_f, bgm1, bgm2

#------------------------------------------------------------------------------------------

def distribution_difference_hotelling_test(X:np.array, Y:np.array, alpha:float = 0.05,
                                           n_mixtures:int = 50,
                                           gmm_type:str = "standard") -> dict and            \
                                                                         GaussianMixture and \
                                                                         GaussianMixture:
    '''

    Fit a Gaussian Mixture Model to the input data sets.  Determine the Hotelling T^2 and
    p-values to determine if two data sets are different.

    Note this method is determining if the means of multivariate distributions are difference
    and accounting for the variance in both distributions.  It will likely flag all IoT data
    has being different from the original X data if you looking at sub regions of the original
    sample set.  Hence, this can be used for comparing overal distributions, not individual
    data from peripherial regions.

    Reference:
        https://en.wikipedia.org/wiki/Hotelling%27s_T-squared_distribution


    Parameters
    ----------
    X : np.array
        Data set 1 (n data X n features)
    Y : np.array
        Data set 2 (n data X n features)
    alpha : float, optional
        Statistical significance level. The default is 0.05.
    n_mixtures_Y : int, optional
        If using a standard Gaussian Mixture Model, the number of clusters
        must be inputed.  When using "bayesian", this is the max clusters that could be used.
        The default is 50.
    gmm_type : str, optional
        "standard" or "bayesian".
        The default is "standard".

    Returns
    -------
    dict and GaussianMixture and GaussianMixture
        The dictionary includes the pvalues for each clustered distribution, in which
        a large p-value means it is likely data from Y came from X and a low pvalue
        means there is little evidence that Y came from X.

    '''



    if gmm_type.lower() =="bayesian":
        print("Fitting BGMM 1")
        bgm1 = BayesianGaussianMixture(n_components=n_mixtures, verbose = 2,
                                        random_state=0, max_iter=5000).fit(X)
        print("Fitting BGMM 2")
        bgm2 = BayesianGaussianMixture(n_components=n_mixtures, verbose = 2,
                                        random_state=0, max_iter=5000).fit(Y)
    else:
        print("Fitting BGMM 1")
        bgm1 = GaussianMixture(n_components=n_mixtures, verbose = 2,
                                        random_state=0, max_iter=5000).fit(X)
        print("Fitting BGMM 2")
        bgm2 = GaussianMixture(n_components=n_mixtures, verbose = 2,
                                        random_state=0, max_iter=5000).fit(Y)

    nmm1 = bgm1.means_.shape[0]
    nmm2 = bgm2.means_.shape[0]

    pred = bgm1.predict(X)
    bgm1.dof = Counter(pred)
    pred = bgm2.predict(Y)
    bgm2.dof = Counter(pred)


    t2_tests = {}

    print("Checking for statistically significant difference between distributions "
          + " at the {} significance level.".format(alpha))
    # for dist1 in range(nmm1):
    #     f_ratio = 1e7
    #     p_value = 0.0
    #     for dist2 in range(nmm2):
    #         fr,pv = hotelling_t2(bgm1, bgm2, dist1, dist2, alpha = alpha)
    #         #print(fr)
    #         f_ratio = min(f_ratio, fr)
    #         p_value = max(p_value, pv)
    #     t2_tests[dist1] = [f_ratio, p_value]
    for dist2 in range(nmm2):
        f_ratio = 1e7
        p_value = 0.0
        for dist1 in range(nmm1):
            fr,pv = _hotelling_t2(bgm1, bgm2, dist1, dist2, alpha = alpha)
            #print(fr)
            f_ratio = min(f_ratio, fr)
            p_value = max(p_value, pv)
        t2_tests[dist2] = [f_ratio, p_value]

    #get rid of the defaulted values
    t2_tests = {key:value for key,value in t2_tests.items() if value[0] !=1e7}

    return t2_tests, bgm1, bgm2

#------------------------------------------------------------------------------------------

def get_optimal_n_cluster(X:np.array, max_cluster:int=20, make_plot:bool=False) -> int:
    '''

    Use the silhouette score to find the optimal number of
    clusters in the data.  This function leverages KMeans unsupervised
    clustering.

    This method looks for 2 or more clusters

    Parameters
    ----------
    X : np.array
        Data to look for clustering.
    max_cluster : int, optional
        The default is 20.
    make_plot : bool, optional
        Plot the silhouette score for all number of clusters.
        The default is False.

    Returns
    -------
    int
        Number of clusters with the maximum silhouette score.

    '''


    #clustering
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    sil_score = []
    for nc in range(2,max_cluster):
        clusterer = KMeans(n_clusters=nc, n_init="auto", random_state=10)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        sil_score.append(silhouette_avg)

    if make_plot:
        plot.figure()
        plot.scatter(range(2,max_cluster), sil_score)

    best = np.argmax(sil_score) + 2
    return best


#%% main
if __name__ == '__main__':


    import pprint

    n_var = 2
    n_data = 2000
    X= None
    for i in range(8):
        means1 = np.random.uniform(size=n_var)
        means1 *= np.random.randint(1,15)
        x = np.random.multivariate_normal(means1, np.eye(n_var)*1.5, size=n_data)
        if i >0:
            X = np.append(X,x, axis=0)
        else:
            X = x

    Y = None
    for i in range(1):
        means2 = np.random.uniform(size=n_var)
        #means2 *= np.random.randint(1,50)
        y = np.random.multivariate_normal(means2, np.eye(n_var)*1.5, size=n_data)
        if i >0:
            Y = np.append(Y,y, axis=0)
        else:
            Y = y


    n_mix = int(np.sqrt(n_data))
    n_mix = 100

    # using t2 hotelling, break the distributions into tiny sub-distributions and
    # check if possibly to find sub distirbutions that match each other.  Since we
    # are assuming tiny gaussian distibutions, this is algebraic and much, much
    # faster than the MC method, but inherently less accurate
    max_p = []
    for n_mix in range(10,150,20):
        t2_tests, bgm1, bgm2 = distribution_difference_hotelling_test(X, Y,
                                                                      alpha=1e-5,
                                                                      n_mixtures=n_mix)
        max_t2_pvalues = np.max([x[1] for x in t2_tests.values()])
        max_p.append(max_t2_pvalues)

    # using MC integration to calculate probability Y came from X

    #this check uses a silhouette score, for 2+ clusters
    #users can also use bayesian GMM to find best cluster number
    Xn = get_optimal_n_cluster(X, max_cluster=20, make_plot=True)
    Yn = 1
    P_given_f, bgm1, bgm2 = distribution_difference_MC_test(X, Y,
                                                            alpha=1e-5,
                                                            n_mixtures_X=Xn,
                                                            n_mixtures_Y=Yn )
    max_MC_pvalues = np.max([x[0] for x in P_given_f.values()])

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(t2_tests)
    pp.pprint(max_p)
    print(f"max pvalues {max_t2_pvalues}")
    pp.pprint(P_given_f)
    print(f"max pvalues {max_MC_pvalues}")

    plot.close('all')

    plot.figure()
    plot.scatter(X[:,0], X[:,1], label='Data Set 1', s=30, facecolors='none', edgecolors='b')
    plot.scatter(Y[:,0], Y[:,1], label='Data Set 2', s=30, facecolors='none', edgecolors='r')
    plot.ylabel('Var1')
    plot.xlabel('Var2')
    plot.tight_layout()


    pred1 = bgm1.predict(X)
    pred2 = bgm2.predict(Y)

    plot.figure()
    plot.scatter(X[:,0], X[:,1], label='Data Set 1', c=pred1)
    plot.ylabel('Var1')
    plot.xlabel('Var2')
    plot.tight_layout()

    plot.figure()
    plot.scatter(Y[:,0], Y[:,1], label='Data Set 2', c=pred2)
    plot.ylabel('Var1')
    plot.xlabel('Var2')
    plot.tight_layout()

















