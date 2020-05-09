# -*- coding: utf-8 -*-
"""

Development package for RMS distribution on a set of Fourier Harmonics

"""
#
# Idea is: input a time history, the acquisition frequency and the set of
# harmonics to investigate and get the estimate of the RMS^2 contribution
# of each Harmonic slot.
# Take the SQRT for individual RMS slots - sum and SQRT for the whole Harmonic
# set/Class.
#
import numpy as np


class HarmonicsRMS:

    """ Harmonic RMS distribution class """

    def __init__(self, sigX, timeX, hrmOne, numHrm, acqFrq, frqTol=2):
        """Harmonic RMS distribution.
        #
        Parameters:
            - [sigX, timeX ]    : input time-history (signal, time-axis)
            - hrmOne            : frequency of 1st Harmonic component [Hz]
            - numHrm            : maximum Harmonic order (freq=numHrm*hrmOne)
            - acqFrq            : acquisition frequency [Hz]
            - frqTol(optional)  : tolerance on harmonic frequency values [%]
        Attributes (from Parameters):
            - signal            : sigX      - signal array
            - time              : timeX     - time array
            - hrmOne            : hrmOne    - 1st Harmonic Frequency
            - numHrm            : numHrm    - maximum Harmonic order
            - acqFreq           : acquisition frequency
            - frqTol            : tolerance level on Harmonic position
        Attributes (Additional):
            - pow2              : power-of-2 for FFT padding
            - fftAxlen          : length of positive Frequency semi-axis
            - rmsSqrScale       : RMS^2 scale factor
            - fftCoeffs         : signal FFT over 2^(pow2) points
            - frqAx             ; positive Frequency semi-axis
            - harmonicRMSsq     : output array - Harmonics RMS^2 values
        """
        # input parameters
        sigLen = len(sigX)
        if (abs(len(timeX) - sigLen) > 0):
            raise ValueError("signal and time arrays mus have same size")
        if (hrmOne < 1):
            raise ValueError("frequency of 1st Harmonic is < 1Hz")
        # rise Error on stupidly high harmonic order ?
        # if (numHrm > 30):
        #   raise ValueError("Maximum Harmonic Order Higher than 30")
        #
        if (frqTol < 0 or frqTol > 100):
            raise ValueError("Frequency Tolerance [%] must be in [0+, 100-]")
        # frequency plausibility test:
        # {it's actually 1% of 1/3rd but, not OK anyway}
        ptsTest = np.ndarray.astype(int(sigLen/3+1) *
                                    np.random.random(int(sigLen/3)), 'int')+2
        idFrqChk = abs(1/(timeX[ptsTest]-timeX[ptsTest-1]) - acqFrq) > 1e-6
        if (sum(1*idFrqChk) > 0.01*sigLen):
            raise ValueError("More than 1% samples not matching frequency")
        # input done
        self.sigLen = sigLen
        self.signal = sigX
        self.time = timeX
        self.hrmOne = hrmOne
        self.numHrm = numHrm
        self.acqFreq = acqFrq
        self.frqTol = frqTol
        # calculate additional variables
        # - power-of-2 for padding
        self.pow2 = int(np.round(np.log(self.sigLen)/np.log(2) + 0.5))
        #
        # fft Spectrum is 1/2 siglen for usual simmetry
        self.fftAxLen = 2**(self.pow2-1)+1
        #
        # RMS**2 scaling factor
        self.rmsSqrScale = self.sigLen/2**(self.pow2)
        #
        # get the FFT done
        self.fftCoeffs = np.fft.fft(self.signal - np.mean(self.signal),
                                    2**self.pow2)/self.sigLen
        self.frqAx = 0.5*self.acqFreq *\
            np.arange(0, self.fftAxLen)/self.fftAxLen

    def distribRMS(self):
        """ Estimate the RMS^2 distribution on the set of Harmonic components
            defined in input object.

            Output:     harmonicRMSsq - Array of RMS^2 attributed to the
                        Harmonic components under analysis.
        """

        #
        # from frequency axis: get frequency resolution and sampling frequency
        fftFrqRes = self.frqAx[1]
        fftCoeffs = self.fftCoeffs
        #
        # find Harmonic separation AND compare to a 5Hz frequency interval
        # assuming 5Hz is enough - input parameter ?
        nfrqHrm = np.floor(self.hrmOne/fftFrqRes)
        nfrq5Hz = np.round(5.0/fftFrqRes)
        #
        #  choose the larger number of frequency points as averaging range
        nffrqSum = max(nfrqHrm, nfrq5Hz)
        #
        # identify FFT components with amplitude > 0.1
        # - "flgFrqLoads"  : binary mask/stencil to pluck frequency components
        # - "idxFrqLoads"  : selected iterable index
        # - "frqLoadArray" : selected frequency array to feed the for-loop
        idxFrqSpctrm = np.arange(0, self.fftAxLen)
        flgFrqLoads = (2*abs(fftCoeffs[0:self.fftAxLen]) > 0.1)
        idxFrqLoads = idxFrqSpctrm[flgFrqLoads]
        frqLoadArray = self.frqAx[idxFrqLoads]
        #
        # arrays of Harmonic components, range to to "maxHrm"
        # frequencies and amplitudes
        hrmRange = np.arange(1, self.numHrm)
        frqHrmArray = self.hrmOne * hrmRange
        amplHrmArray = 0.0 * frqHrmArray
        #
        # tolerance on frequency slots
        frqDelta = 1e-2 * self.frqTol
        #
        # Initialise Array of "Next Frequency Slot"
        idSltNext = []
        #
        # groundwork done
        #
        # loop over the array of Harmonic slots :
        # - check if frequency slot is to be counted
        # - calculate the average sum-of-squares over designated range
        for nFrq in hrmRange:
            #
            # - find candidate frequency slot : idSlt
            # - find next frequency slot : idSltNext
            # - check frequency separation
            # - get the max peak location
            # - calculate the local RMS by averaging around the peak without overlap
            #   with next slot
            if (nFrq > 1):
                #
                # already initialised : just update running slot
                idSlt = idSltNext.copy()
            else:
                #
                # 1st step
                idSlt = (abs(frqLoadArray - frqHrmArray[nFrq-1])/frqHrmArray[nFrq-1] <= frqDelta)
            #
            # compute next frequency slot - if not finishing
            if (nFrq < self.numHrm-1):
                idSltNext = (abs(frqLoadArray - frqHrmArray[nFrq])/frqHrmArray[nFrq] <= frqDelta)
            else:
                idSltNext = []
            #
            # Compute average squared RMS when frequency interval identified
            idxFrqSlt = idxFrqLoads[idSlt]
            idxFrqSltNext = idxFrqLoads[idSltNext]
            if (len(idxFrqSlt) > 0):
                #
                # find the peak
                testArray = abs(fftCoeffs[idxFrqSlt])
                # mxAmpl = np.max(testArray)
                mxIdx = np.argmax(testArray)
                if (len(idxFrqSltNext) == 0):
                    #
                    # no close-by slot, use 5Hz interval
                    deltaSlots = nffrqSum + 1
                else:
                    #
                    # slot separation : min(nextFrqSlot) - freq(peak)
                    deltaSlots = min(idxFrqSltNext) - idxFrqSlt[mxIdx]
                #
                # if freq separation wide, average on 5Hz around peak
                # if next slot close-by, take 1/2 freq separation
                if(deltaSlots > nffrqSum):
                    nptsRMS = min(nffrqSum, nfrq5Hz)
                else:
                    nptsRMS = np.round(0.5*deltaSlots)
                #
                # compute the average: if the peak is not outside the 1st nptsRMS
                # points use this interval from 0
                if (idxFrqSlt[mxIdx] > nptsRMS):
                    amplHrmArray[nFrq-1] = \
                        sum(2*abs(
                            fftCoeffs[idxFrqSlt[mxIdx] - int(nptsRMS):
                                      idxFrqSlt[mxIdx] + int(nptsRMS)]
                                )**2) * self.rmsSqrScale
                else:
                    amplHrmArray[nFrq-1] = sum(2*abs(
                        fftCoeffs[0: int(nptsRMS)])**2) * self.rmsSqrScale
                #
                # cleanup
                idSlt = []
        self.harmonicRMSsq = amplHrmArray
        return self.harmonicRMSsq
