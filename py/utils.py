#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
import emcee
import george
from george import kernels

__all__ = ["double_gaussian", "model", "fit_gp", "fit_ind", "single_gaussian"]


def double_gaussian(wl, amp1, amp2, sig, const, redshift):
    cen1 = 3727.09 * (1. + redshift)
    cen2 = 3729.88 * (1. + redshift)
    return amp1 * np.exp(-(wl - cen1)**2. / sig) + amp2 * np.exp(-(wl - cen2)**2. / sig) + const


def single_gaussian(wl, amp, sig, redshift, linecenter=3727.09):
    cen = linecenter * (1. + redshift)
    return amp * np.exp(-(wl - cen)**2. / sig)


def model(params, wl):
    amp1, amp2, sig, const, redshift = params
    cen1 = 3727.09 * (1. + redshift)
    cen2 = 3729.88 * (1. + redshift)
    return amp1 * np.exp(-(wl - cen1)**2. / sig) + amp2 * np.exp(-(wl - cen2)**2. / sig) + const


def lnprior(params):
    amp1, amp2, sig, const, redshift = params
    if (0 < amp1 < np.inf and 0 < amp2 < np.inf and 0 < sig < np.inf and -np.inf < const < np.inf and 0.5 < redshift < 1.5):
        return 0.0
    return -np.inf


def lnlike_ind(params, wl, flux, fluxerror):
    m = model(params, wl)
    return -0.5 * np.sum((flux - m) ** 2. / fluxerror ** 2.)


def lnprob_ind(params, wl, flux, fluxerror):
    lp = lnprior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_ind(params, wl, flux, fluxerror)


def fit_ind(initial, initial_error, data, nwalkers=32):
    ndim = len(initial)
    p0 = [np.random.normal(initial, initial_error)
          for i in xrange(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_ind, args=data)

    print("Running burn-in")
    p0, _, _ = sampler.run_mcmc(p0, 500)
    sampler.reset()

    print("Running production")
    p0, _, _ = sampler.run_mcmc(p0, 1000)

    return sampler


def lnlike_gp(params, wl, flux, fluxerror):
    a, tau = np.exp(params[:2])
    gp = george.GP(a * kernels.Matern32Kernel(tau))
    gp.compute(wl, fluxerror)
    return gp.lnlikelihood(flux - model(params[2:], wl))


def lnprior_gp(params):
    lna, lntau = params[:2]
    if not -15 < lna < 15:
        return -np.inf
    if not -150 < lntau < np.inf:
        return -np.inf
    return lnprior(params[2:])


def lnprob_gp(params, wl, flux, fluxerror):
    lp = lnprior_gp(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_gp(params, wl, flux, fluxerror)


def fit_gp(initial, initial_error, data, nwalkers=32):
    ndim = len(initial)
    p0 = [np.random.normal(initial, initial_error)
          for i in xrange(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_gp, args=data)

    print("Running burn-in")
    p0, lnp, _ = sampler.run_mcmc(p0, 1500)
    # sampler.reset()

    # print("Running second burn-in")
    # p = p0[np.argmax(lnp)]
    # p0 = [np.random.normal(p, 1e-2*np.array(initial_error)) for i in xrange(nwalkers)]
    # p0, _, _ = sampler.run_mcmc(p0, 500)
    # sampler.reset()

    # print("Running production")
    # p0, _, _ = sampler.run_mcmc(p0, 1000)

    return sampler