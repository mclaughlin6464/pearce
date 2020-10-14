# This code subclasses GPy's kronecker module to include a variance vector instead of a constant.

import numpy as np
from GPy.models import GPKroneckerGaussianRegression


class GPKroneckerGaussianRegressionVar(GPKroneckerGaussianRegression):
    """
    See GPKroneckerGaussianRegression for documentation
    """

    def __init__(self, X1, X2, Y, Y_var, kern1, kern2, noise_var=1.,
                 name='KGPR', additional_Xs = [],
                 additional_kerns=[]):

        assert Y_var.shape == Y.shape, "Y_var does not have the same shape as Y. "

        super(GPKroneckerGaussianRegressionVar,self).__init__(X1, X2, Y, kern1, kern2, noise_var=noise_var, name=name,
                                                              additional_Xs=additional_Xs,
                                                              additional_kerns=additional_kerns)

        # NOTE this is not optimal, but should roughly work
        self.Y_var = Y_var

    def parameters_changed(self):
        # lifted whole cloth from the super class to add one line.

        dims = len(self.Y.shape)
        Ss, Us = [], []
        for i in range(dims):
            X = getattr(self, "X%d" % i)
            kern = getattr(self, "kern%d" % i)

            K = kern.K(X)
            S, U = np.linalg.eigh(K)

            # if i==1:
            #    Ss.insert(0, S)
            #    Us.insert(0, U)
            # else:
            Ss.append(S)
            Us.append(U)

        # ^^^ swap the orders of the first and second elements
        # this is only necessary to make sure things are consistent with non-kronecker kernels
        # mathematically theyr're the same.

        W = reduce(np.kron, reversed(Ss))
        W+= self.Y_var.flatten(order = 'F') # This is the only step that's changed.
        W += self.likelihood.variance

        # rotated_Y = np.swapaxes(self.Y, 0,1)
        Y_list = [self.Y]
        Y_list.extend(Us)

        Y_ = reduce(lambda x, y: np.tensordot(x, y.T, axes=[[0], [1]]), Y_list)
        Wi = 1. / W
        Ytilde = Y_.flatten(order='F') * Wi
        num_data_prod = np.prod([getattr(self, "num_data%d" % i) for i in range(len(self.Y.shape))])

        self._log_marginal_likelihood = -0.5 * num_data_prod * np.log(2 * np.pi) \
                                        - 0.5 * np.sum(np.log(W)) \
                                        - 0.5 * np.dot(Y_.flatten(order='F'), Ytilde)

        # gradients for data fit part
        Yt_reshaped = np.reshape(Ytilde, self.Y.shape, order='F')
        Wi_reshaped = np.reshape(Wi, self.Y.shape, order='F')

        for i in range(dims):
            U = Us[i]
            tmp = np.tensordot(U.T, Yt_reshaped, axes=[[0], [i]])
            S = reduce(np.multiply.outer, [s for j, s in enumerate(Ss) if i != j])

            tmps = tmp * S
            # NOTE not pleased about the construction of these axes. Should be able to use a simpler
            # integer input to axes, but in practice it didn't seem to work.

            axes = [[k for k in range(dims - 1, 0, -1)], [j for j in range(dims - 1)]]
            dL_dK = .5 * (np.tensordot(tmps, tmp.T, axes=axes))

            axes = [[k for k in range(dims - 1, -1, -1) if k != i], [j for j in range(dims - 1)]]
            tmp = np.tensordot(Wi_reshaped, S.T, axes=axes)

            dL_dK += -0.5 * np.dot(U * tmp, U.T)

            getattr(self, "kern%d" % i).update_gradients_full(dL_dK, getattr(self, "X%d" % i))

        # gradients for noise variance
        dL_dsigma2 = -0.5 * Wi.sum() + 0.5 * np.sum(np.square(Ytilde))
        self.likelihood.variance.gradient = dL_dsigma2

        # store these quantities for prediction:
        self.Wi, self.Ytilde = Wi, Ytilde

        for i, u in enumerate(Us):
            setattr(self, "U%d" % i, u)

    def predict(self, X1new, X2new, mean_only=False,additional_Xnews=[]):

        return super(GPKroneckerGaussianRegressionVar, self).predict(X1new, X2new,mean_only=mean_only,\
                                                                     additional_Xnews=additional_Xnews)
        # TODO i think the diag should be added to var, not sure how tho
        #return mu, var
