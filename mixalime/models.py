# -*- coding: utf-8 -*-
"""Window models of betanegbinfit with a rescaled objective."""
from betanegbinfit.models import ModelWindow as _ModelWindow, ModelWindowRec as _ModelWindowRec
from scipy.optimize import minimize
import numpy as np
import logging


class ModelWindow(_ModelWindow):

    def prepare_start(self, start: np.ndarray, standoff=0.04,
                      eps=0.01) -> np.ndarray:
        start = start.copy()
        for i, ((a, b), v) in enumerate(zip(self.bounds, start)):
            if a is not None and b is not None:
                # Capped, as 4% of a (0, kappa_right) range is far above a typical kappa.
                d = min((b - a) * standoff, standoff)
                start[i] = min(max(v, a + d), b - d)
            elif a is not None:
                start[i] = max(v, a + eps)
            elif b is not None:
                start[i] = min(v, b - eps)
        return start

    def _fit(self, data, use_prev='only', calc_std=False, stop_first=False, prior_arg=1.0,
             param_mask=None, _other_opt=False, try_all_opts=False, **kwargs):
        """
        Fit model to data.

        Parameters
        ----------
        data : np.ndarray
            Array of shape (number of observations x 3), where columns are ref
            counts, alt counts and number of occurences.
        use_prev : str, optional
            If 'add', then a previous optima will be added to the pool of
            starting points. If 'only', then only the previous point will be
            used in optimization. If None or empty string, then previous points
            are ignored.
        calc_std : bool, optional
            If True, then standard deviations of parameter estimates are
            calculated. The default is False.
        stop_first : bool, optional
            If True, the procedure stops at the first successful optimization.
            The default is False.
        **kwargs : dict
            Extra arguments to be passed to the optimizer (if applicable).

        Returns
        -------
        params : dict
            Dictionary of estimated parameters and loglikelihood.

        """
        assert data.shape[1] == 3, data.shape
        orig_data = data.copy()
        weights = data[:, -1]
        data = data[:, :-1]
        df, weights, mask = self.update_mask(data, weights)
        # Optimizers are given the mean loglikelihood. Gradients of the sum grow with
        # the number of observations and SLSQP diverges at millions of them.
        n = weights.sum()
        args = {'data': df, 'mask': mask, 'weights': weights, 'prior_arg': prior_arg,
                'param_mask': param_mask}
        jac = lambda x: self.grad_loglik(x, **args)[0] / n
        fun = lambda x: self.negloglikelihood(x, **args) / n
        best_r = None
        best_loglik = -np.inf
        x0s = self.x0
        if use_prev and hasattr(self, 'params'):
            prev = [self.prepare_start(self.last_result.x)]
            x0s = prev + x0s
            prev_succ = self.last_result.success
            prev = prev[0]
        else:
            prev_succ = False
            prev = None
        if _other_opt:
            optimizer = 'L-BFGS-B' if self.optimizer == 'SLSQP' else 'SLSQP'
        else:
            optimizer = self.optimizer
        options = {'maxiter': 1000, 'disp': False, 'ftol': 1e-12}
        if optimizer == 'L-BFGS-B':
            options['gtol'] = 1e-8
        for i, x0 in enumerate(x0s):

            logging.debug(f'Fitting model to data: starting point {i}/{len(x0s)}.')
            r = minimize(fun, x0, jac=jac, method=optimizer,
                         bounds=self.bounds, options=options)
            r.fun = float(r.fun) * n
            loglik = -r.fun
            if loglik > best_loglik:
                best_loglik = loglik
                best_r = r
                if r.success and ((prev_succ and use_prev == 'only') or stop_first) and (not prev_succ or np.abs( (r.x - prev) / prev).max() < 10.0):
                    break
        if best_r is None:
            if not _other_opt:
                return self._fit(orig_data, use_prev, calc_std, stop_first, prior_arg, param_mask, _other_opt=True)
            logging.debug("Optimizer didn't converge.")
        elif not _other_opt and try_all_opts:
            _, r = self._fit(orig_data, use_prev, False, stop_first, prior_arg, param_mask, _other_opt=True)
            try:
                if r.fun < best_r.fun:
                    best_r = r
            except AttributeError:
                pass
        if best_r is None:
            return np.nan, None
        params = self.vec_to_dict(best_r.x)
        params['loglik'] = float(best_loglik) / weights.sum()
        params['success'] = best_r.success
        self.params = params
        self.last_result = best_r
        self.prev_res = self.last_result.x
        if calc_std:
            cov = self.calc_cov(df, weights=weights, params=best_r.x,
                                mask=mask)
            stds = {p: v for p, v in zip (self.params_active,
                                          np.sqrt(np.diag(cov)))}
            params['std'] = stds
        return params, best_r


class ModelWindowRec(ModelWindow, _ModelWindowRec):
    pass
