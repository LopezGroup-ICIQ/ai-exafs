"""
Copyright (c) 2026 - ICIQ (Institute of Chemical Research of Catalonia)

Developed at ICIQ and funded by NCCR Catalysis.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

"""
Code adapted from xraylarch python package.
LINK: https://github.com/xraypy/xraylarch/blob/master/larch/xafs/xafsft.py
"""

import numpy as np

import larch
from larch.xafs import (
    ftwindow,
    xftf_fast,
)

from larch import Group, parse_group_args, Journal
from larch.math import complex_phase
from larch.xafs.xafsutils import etok, ktoe


def set_xafsGroup(group, _larch=None):
    """set _sys.xafsGroup to the supplied group (if not None)

    return _sys.xafsGroup.

    if needed, a new, empty _sys.xafsGroup may be created.
    """
    if group is None:
        if _larch is None:
            group = Group()
        else:
            group = getattr(_larch.symtable._sys, "xafsGroup", Group())
    if not hasattr(group, "journal"):
        group.journal = Journal()

    if _larch is not None:
        _larch.symtable._sys.xafsGroup = group
    return group


def xftf_prep(
    k,
    chi,
    kmin=0,
    kmax=20,
    kweight=2,
    dk=1,
    dk2=None,
    window="kaiser",
    nfft=2048,
    kstep=0.05,
    _larch=None,
):
    """
    calculate weighted chi(k) on uniform grid of len=nfft, and the ft window

    Returns weighted chi, window function which can easily be multiplied and
    used in xftf_fast.
    """
    if dk2 is None:
        dk2 = dk
    kweight = int(kweight)
    npts = int(1.01 + max(k) / kstep)
    k_max = max(max(k), kmax + dk2)
    k_ = kstep * np.arange(int(1.01 + k_max / kstep), dtype="float64")
    chi_ = np.interp(k_, k, chi)
    win = ftwindow(k_, xmin=kmin, xmax=kmax, dx=dk, dx2=dk2, window=window)
    return (chi_[:npts], win[:npts])


def xas_fft(
    k,
    chi=None,
    group=None,
    kmin=0,
    kmax=20,
    kweight=2.0,
    dk=1,
    dk2=None,
    with_phase=False,
    window="kaiser",
    rmax_out=10,
    nfft=2048,
    kstep=None,
    _larch=None,
    **kws
):
    """
    Forward XAFS Fourier Transform, from chi(k) to chi(R), using
    common XAFS conventions.

    Args:
        k: 1-d array of photo-electron wavenumber in Ang^-1 of group
        chi: 1-d array of chi
        group: output Group
        rmax_out: highest R for output data (10 Ang)
        kweight: exponent for weighting spectra by k**kweight [2]
        kmin: starting k for FT window
        kmax: ending k for FT window
        dk: tapering parameter for FT window
        dk2: second tapering parameter for FT window
        window: name of window type
        nfft: value to use for N_fft
        kstep: value to use for delta_k
        with_phase: output the phase as well as magnitude, real, img
    """
    # Assuming kweight equal to 2.0
    if kweight is None:
        kweight = 2.0

    k, chi, group = parse_group_args(
        k, members=("k", "chi"), defaults=(chi,), group=group, fcn_name="xftf"
    )

    if kstep is None:
        kstep = k[1] - k[0]

    cchi, win = xftf_prep(
        k,
        chi,
        kmin=kmin,
        kmax=kmax,
        kweight=kweight,
        dk=dk,
        dk2=dk2,
        nfft=nfft,
        kstep=kstep,
        window=window,
        _larch=_larch,
    )

    # FFT
    out = xftf_fast(cchi * win, kstep=kstep, nfft=nfft)
    rstep = np.pi / (kstep * nfft)

    irmax = int(min(nfft / 2, 1.01 + rmax_out / rstep))

    group = set_xafsGroup(group, _larch=_larch)
    r = rstep * np.arange(irmax)
    mag = np.sqrt(out.real**2 + out.imag**2)
    group.kwin = win[: len(chi)]
    group.r = r[:irmax]
    group.chir = out[:irmax]
    group.chir_mag = mag[:irmax]
    group.chir_re = out.real[:irmax]
    group.chir_im = out.imag[:irmax]
    if with_phase:
        group.chir_pha = complex_phase(out[:irmax])
    return
