import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class Translator:
    def __init__(self, rmin, rmax, nbins, verbose=False):
        self.rmin = rmin
        self.rmax = rmax
        self.nbins = nbins
        self.verbose = verbose
        self.binwidth = float(self.rmax - self.rmin)/self.nbins
        
    def bin(self, energy):
        b = int((energy - self.rmin)/self.binwidth)
        if b > self.nbins - 1:
            if self.verbose:
                print("Energy out of range: ", energy)
            b = self.nbins - 1
        elif b < 0:
            if self.verbose:
                print("Energy out of range: ", energy)
            b = 0
            
        return b

class DataManager:
    def __init__(self, filename):
        histData = np.genfromtxt(filename, delimiter=',', skip_header=1)
        xmax, ymax, _ = np.max(histData, axis=0) # get column-wise maxes
        if xmax != ymax:
            print('Maxes not equal:', xmax, ymax)

        maxbin = max(xmax, ymax)
        self.nbins = int(maxbin)

        self.hCoincidences = np.zeros((self.nbins, self.nbins), dtype="int64")
        for row in histData:
            x = int(row[0])-1
            y = int(row[1])-1
            count = int(row[2])
            self.hCoincidences[x,y] = count
            self.hCoincidences[y,x] = count

        # Project 2D histogram into a single dimension
        self.hProjection = np.sum(self.hCoincidences, axis=0)

        # Make PDF
        totalCount = float(np.sum(self.hProjection))
        self.pdf = self.hProjection.astype(float) / totalCount
        self._makeCDF()
        # print("pdf", self.pdf)
        # print("cdf", self.cdf)

        # Create vocabulary of bins
        self._makeVocab()
        return

    def _makeVocab(self):
        # Convert 1D histogram to dict for sorting
        binToFreq = {}
        for i in range(self.nbins):
            binToFreq.update({i:self.hProjection[i]})   
        
        order = 0
        self.binToOrder = np.zeros((self.nbins,), dtype="int64")
        self.orderToBin = np.zeros((self.nbins,), dtype="int64")
        for b in sorted(binToFreq, key=binToFreq.get, reverse=True):
            self.binToOrder[b] = order
            self.orderToBin[order] = b
            order += 1

    def _makeCDF(self):
        self.cdf = np.zeros((self.nbins,), dtype="float")
        self.cdf[0] = 0
        for i in range(1, self.nbins):
            self.cdf[i] = self.cdf[i-1] + self.pdf[i-1]

    def getBinOrder(self):
        return self.binToOrder

    def getSortedVocab(self):
        return self.orderToBin

    def plot1D(self):
        fig, ax = plt.subplots()
        ax.plot(self.hProjection, linewidth=0.3, label="Projection")
        ax.semilogy()
        ax.legend()

    def plotPDF(self):
        fig, ax = plt.subplots()
        ax.plot(self.pdf, linewidth=0.3, label="Probability Distribution")
        ax.semilogy()
        ax.legend()

    def plotCDF(self):
        fig, ax = plt.subplots()
        ax.plot(self.cdf, linewidth=0.3, label="Cumulative Distribution Function")
        # ax.semilogy()
        ax.legend()
    
    def plot2D(self, subsample=1):
        """Plot the 2D histogram as a sanity check.
         
        What we actually plot is a subsampling of the histogram data, because
        in many cases there will be more bins than available pixels, and the
        resulting images are hard to interpret. Increasing the 'subsample' value
        will collapse multiple bins into increasingly large new bins, which are
        then plotted."""
        
        # n.b. this does not do anything graceful when 'subsample'
        #      does not divide evenly into 'nb', but that's probably fine
        #      because there is very little data in the highest bin.
        nbNew = math.ceil(self.nbins/subsample)
        hCoincSub = np.zeros((nbNew, nbNew))
        
        # fyi: this loop can take 10-20 seconds
        for i in range(self.nbins):
            for j in range(self.nbins):
                isub = int(i/subsample)
                jsub = int(j/subsample)
                # print(self.hCoincidences[i,j])
                hCoincSub[isub, jsub] += self.hCoincidences[i, j]
        
        # Create alpha (transparency) channel
        # This plots a transparent pixel wherever the bin count is 0
        # As a result, it's much easier to see where the nonzero data is
        alpha = (hCoincSub != 0).astype(float)
        
        # Plot 2D histogram
        fig, ax = plt.subplots()
        im = ax.imshow(hCoincSub, origin='lower', alpha=alpha, cmap='cool')
        fig.colorbar(im)
        plt.show()

    def getCoincidences(self):
        return self.hCoincidences

    def getProjection(self):
        return self.hProjection

    # def randomSkipGram(self):
