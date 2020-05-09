# -*- coding: utf-8 -*-
"""

How-to example

"""
import numpy as np

from HarmSqrdRMS import HarmonicsRMS

sigLen = 2e5
xx = np.arange(0, sigLen)
frq = 5e4
delT = 1/frq
timA = delT*xx
sin10Hz = 3*np.sin(2*np.pi * 10 * timA)
sin20Hz = 7.3*np.sin(2*np.pi * 20 * timA)
sin120Hz = 12.3*np.sin(2*np.pi * 120 * timA)
sin240Hz = 2.3*np.sin(2*np.pi * 240 * timA)
sin300Hz = 1.5*np.sin(2*np.pi * 300 * timA)
baseLinNoise = 2.8 + 5*np.random.random(sigLen)
signal = baseLinNoise + sin10Hz + sin20Hz + sin120Hz + sin240Hz + sin300Hz
rmsSignal = np.sqrt(sum((signal - np.mean(signal))**2)/sigLen)
#
rmsHarmEstimate = HarmonicsRMS(signal, timA, 10.0, 40, 50e3, 0.1)
sigRmsHarmonics = rmsHarmEstimate.distribRMS()
#
