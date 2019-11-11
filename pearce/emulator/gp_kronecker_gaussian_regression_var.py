# This code subclasses GPy's kronecker module to include a variance vector instead of a constant.

import numpy as np
from GPy.models import GPKroneckerGaussianRegression


class GPKroneckerGaussianRegressionVar(GPKroneckerGaussianRegression):
    """
    See GPKroneckerGaussianRegression for documentation
    """

    def __init__(self, X1, X2, Y, Y_var, kern1, kern2, noise_var=1., name='KGPR'):

        assert Y_var.shape == Y.shape, "Y_var does not have the same shape as Y. "

        super(GPKroneckerGaussianRegressionVar,self).__init__(X1, X2, Y, kern1, kern2, noise_var=noise_var, name=name)

        # NOTE this is not optimal, but should roughly work
        self.Y_var = Y_var

    def parameters_changed(self):
        (N1, D1), (N2, D2) = self.X1.shape, self.X2.shape
        K1, K2 = self.kern1.K(self.X1), self.kern2.K(self.X2)

        # eigendecompositon
        S1, U1 = np.linalg.eigh(K1)
        S2, U2 = np.linalg.eigh(K2)
        # only change ###
        W = np.kron(S2, S1) + self.Y_var.flatten(order = 'F')+ self.likelihood.variance
        #################

        Y_ = U1.T.dot(self.Y).dot(U2)

        # store these quantities: needed for prediction
        Wi = 1. / W
        Ytilde = Y_.flatten(order='F') * Wi

        self._log_marginal_likelihood = -0.5 * self.num_data1 * self.num_data2 * np.log(2 * np.pi) \
                                        - 0.5 * np.sum(np.log(W)) \
                                        - 0.5 * np.dot(Y_.flatten(order='F'), Ytilde)

        # gradients for data fit part
        Yt_reshaped = Ytilde.reshape(N1, N2, order='F')
        tmp = U1.dot(Yt_reshaped)
        dL_dK1 = .5 * (tmp * S2).dot(tmp.T)
        tmp = U2.dot(Yt_reshaped.T)
        dL_dK2 = .5 * (tmp * S1).dot(tmp.T)

        # gradients for logdet
        Wi_reshaped = Wi.reshape(N1, N2, order='F')
        tmp = np.dot(Wi_reshaped, S2)
        dL_dK1 += -0.5 * (U1 * tmp).dot(U1.T)
        tmp = np.dot(Wi_reshaped.T, S1)
        dL_dK2 += -0.5 * (U2 * tmp).dot(U2.T)

        self.kern1.update_gradients_full(dL_dK1, self.X1)
        self.kern2.update_gradients_full(dL_dK2, self.X2)

        # gradients for noise variance
        dL_dsigma2 = -0.5 * Wi.sum() + 0.5 * np.sum(np.square(Ytilde))
        self.likelihood.variance.gradient = dL_dsigma2

        # store these quantities for prediction:
        self.Wi, self.Ytilde, self.U1, self.U2 = Wi, Ytilde, U1, U2

    def predict(self, X1new, X2new):

        mu, var = super(GPKroneckerGaussianRegressionVar, self).predict(X1new, X2new)
        # TODO i think the diag should be added to var, not sure how tho
        return mu, var
