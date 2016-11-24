#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import matplotlib.pylab as pl
import seaborn as sns; sns.set_style('ticks')
import numpy as np
import scipy.optimize as op
from utils import *


import emcee
import corner
import george
from george import kernels
from matplotlib.ticker import MaxNLocator




def main():
    np.random.seed(12345)
    dat = np.genfromtxt("../data/VIS_combinedstdext.dat")
    mask = ~np.isnan(dat[:, 2]) & (dat[:, 1] <= 7495) & (dat[:, 1] > 7460)
    wl = dat[:, 1][mask][::1]
    x = np.arange(min(wl), max(wl), 0.01)
    flux = (dat[:, 2][mask]/1e-17)[::1]
    error = (abs(dat[:, 3][mask])/1e-17)[::1]
    pl.errorbar(wl, flux, yerr=error, fmt=".k", capsize=0, elinewidth=0.5, ms=3, rasterized=True)
    init_vals = [1.0e-18 / 1e-17, 1.5e-18 / 1e-17, 1, 2e-19 / 1e-17, 1.005]


    popt, pcov = op.curve_fit(double_gaussian, wl, flux, sigma=error, p0 = init_vals, maxfev = 5000)
    pl.plot(x, double_gaussian(x, *popt), color=sns.xkcd_rgb["purple red"])
    # print(popt)

    # Fit assuming independent.
    print("Fitting independent")
    data = (wl, flux, error)
    sampler = fit_ind(popt, np.sqrt(np.diag(pcov)), data)

    # # Plot the samples in data space.
    # print("Making plots")
    samples = sampler.flatchain
    # # x = np.linspace(-5, 5, 500)
    # for s in samples[np.random.randint(len(samples), size=24)]:
    #     pl.plot(x, model(s, x), color="#4682b4", alpha=0.3)

    # pl.title("results assuming uncorrelated noise")
    # pl.savefig("../figs/ind-results.pdf", dpi=150)
    # pl.clf()

    # # Make the corner plot.
    labels = [r"$A1$", r"$A2$", r"$\sigma$", r"$const$", r"$z$"]
    fig = corner.corner(samples[:, :], labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
    fig.savefig("../figs/ind-corner.pdf", dpi=150)
    pl.clf()


    # Compute the quantiles.
    # samples[:, 2] = np.exp(samples[:, 2])
    amp1, amp2, sig, const, redshift = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                 zip(*np.percentile(samples, [16, 50, 84],
                                                    axis=0)))
    print("""MCMC result:
        amp1 = {0[0]} +{0[1]} -{0[2]}
        amp2 = {1[0]} +{1[1]} -{1[2]}
        sig = {2[0]} +{2[1]} -{2[2]}
        const = {3[0]} +{3[1]} -{3[2]}
        redshift = {4[0]} +{4[1]} -{4[2]}
    """.format(amp1, amp2, sig, const, redshift))


    print(np.sqrt(redshift[1]**2. + 3e-5**2.))
    print(np.sqrt(redshift[2]**2. + 3e-5**2.))

    sample = []
    l1_flux, l2_flux = [], []
    OIIratio = []
    for s in samples:
        sample.append(model(s, x))
        OII1 = np.trapz(single_gaussian(x, s[0], s[2], s[4], linecenter=3727.09), x)
        OII2 = np.trapz(single_gaussian(x, s[1], s[2], s[4], linecenter=3729.88), x) 
        l1_flux.append(OII1)
        l2_flux.append(OII2)
        OIIratio.append(OII2/OII1)
    lower, mean, upper = np.percentile(sample, [16, 50, 84], axis=0)
    lower = mean - 3*(mean - lower)
    upper = mean + 3*(upper - mean)


    OII1_l, OII1_m, OII1_h = np.percentile(l1_flux, [16, 50, 84])
    OII2_l, OII2_m, OII2_h = np.percentile(l2_flux, [16, 50, 84])
    OIIratio_l, OIIratio_m, OIIratio_h = np.percentile(OIIratio, [16, 50, 84])
    print(OII1_m, OII1_h - OII1_m, OII1_m - OII1_l)
    print(OII2_m, OII2_h - OII2_m, OII2_m - OII2_l)
    print(OIIratio_m, OIIratio_h - OIIratio_m, OIIratio_m - OIIratio_l)
    # print(np.percentile(OIIratio, [16, 50, 84]))
    # print(np.percentile(OIIratio, [16, 50, 84]))
    # # Fit assuming GP.
    # print("Fitting GP")
    # data = (wl, flux, error)
    # popt_gp = [-5, 150.0] + list(popt)
    # popt_err_gp = [2, 50] + list(np.sqrt(np.diag(pcov)))
    # sampler = fit_gp(popt_gp, popt_err_gp, data)


    # fig, axes = pl.subplots(3, 1, sharex=True, figsize=(8, 9))
    # axes[0].plot(sampler.chain[:, :, 0].T, color="k", alpha=0.4)
    # axes[0].yaxis.set_major_locator(MaxNLocator(5))
    # axes[0].axhline(-5, color="#888888", lw=2)
    # axes[0].set_ylabel("$m$")

    # axes[1].plot(sampler.chain[:, :, 1].T, color="k", alpha=0.4)
    # axes[1].yaxis.set_major_locator(MaxNLocator(5))
    # axes[1].axhline(0, color="#888888", lw=2)
    # axes[1].set_ylabel("$b$")

    # axes[2].plot(np.exp(sampler.chain[:, :, 2]).T, color="k", alpha=0.4)
    # axes[2].yaxis.set_major_locator(MaxNLocator(5))
    # axes[2].axhline(popt[0], color="#888888", lw=2)
    # axes[2].set_ylabel("$f$")
    # axes[2].set_xlabel("step number")

    # fig.tight_layout(h_pad=0.0)
    # fig.savefig("../figs/line-time.pdf")


    # # Plot the samples in data space.
    # print("Making plots")
    # samples = sampler.flatchain

    # pl.figure()
    # pl.errorbar(wl, flux, yerr=error, fmt=".k", capsize=0)
    # for s in samples[np.random.randint(len(samples), size=24)]:
    #     gp = george.GP(np.exp(s[0]) * kernels.Matern32Kernel(np.exp(s[1])))
    #     gp.compute(wl, error)
    #     m = gp.sample_conditional(flux - model(s[2:], wl), x) + model(s[2:], x)
    #     pl.plot(x, m, color="#4682b4", alpha=0.3)
    # # pl.ylabel(r"$y$")
    # # pl.xlabel(r"$t$")
    # # pl.xlim(-5, 5)
    # pl.title("results with Gaussian process noise model")
    # # pl.show()
    # pl.savefig("../figs/gp-results.pdf", dpi=150)

    # # Make the corner plot.
    # fig = corner.corner(samples[:, 2:], labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
    # # pl.show()
    # fig.savefig("../figs/gp-corner.pdf", dpi=150)







    pl.clf()
    fig, ax = pl.subplots()
    ax.errorbar(wl, flux, yerr=error, fmt=".k", capsize=0, elinewidth=0.5, ms=3, zorder=10, label="Spock")

    # ax.plot(x, double_gaussian(x, *popt))
    ax.plot(x, mean, zorder=1, label="Best-fit")
    ax.fill_between(x, lower, upper, alpha = 0.5, rasterized=True)
    ax.set_xlabel(r'Observed Wavelength [$\mathrm{\AA}$]')
    ax.set_ylabel(r'Flux density [10$^{-17}$ erg s$^{-1}$ cm$^{-1}$ $\mathrm{\AA}$$^{-1}$]')

    # #Overplot lines
    import lineid_plot
    fit_line_positions = np.genfromtxt('../data/linelist.dat', dtype=None)

    # pl.tight_layout()
    linelist = []
    linenames = []
    for n in fit_line_positions:
        linelist.append(float(n[0])*(1 + redshift[0]))
        linenames.append(str(n[1]))
    lineid_plot.plot_line_ids(x, mean, linelist, linenames, ax = ax)
    # pl.gcf().subplots_adjust(bottom=0.15)

    ax2 = ax.twiny()
    ax2.plot(x/(1 + redshift[0]), mean, alpha = 0)
    ax2.set_xlabel(r'Rest Wavelength [$\mathrm{\AA}$]')
    ax.set_xlim((7465, 7490))
    ax.set_ylim((-1.5e-18/1e-17, 7e-18/1e-17))
    handles, labels = ax.get_legend_handles_labels()
    print(handles)
    print(labels)
    ax.legend([handles[3]] + [handles[0]], [labels[3]] + [labels[0]])
    # ax.legend()
    fig.savefig("../figs/OIIfit_xshoo.pdf", dpi=150)
    # pl.show()

if __name__ == '__main__':
    main()

