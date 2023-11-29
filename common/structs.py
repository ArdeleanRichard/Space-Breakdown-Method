class Spectrum2D:
    def __init__(self, timeValues, frequencyValues, powerValues):
        """

        :param timeValues:      vector - time values at each sample (has the length equal the x axis of the spectrogram)
        :param frequencyValues: vector - frequency values at each sample (has the length equal the y axis of the spectrogram)
        :param powerValues:     matrix - power values, represents the value at each pixel of the spectrogram
        """
        self.timeValues = timeValues
        self.frequencyValues = frequencyValues
        self.powerValues = powerValues

    def binToFrequency(self, row_index):
        return self.frequencyValues[row_index]

    def binToTimepoint(self, col_index):
        return self.timeValues[col_index]




class BoundingBox:
    def __init__(self, L, T, R, B):
        """
        The bounding box in image space.
        :param L: int - left, start value on the x axis
        :param T: int - top, start value on the y axis
        :param R: int - right, end value on the x axis
        :param B: int - bottom, end value on the y axis
        """
        self.L = L
        self.T = T
        self.R = R
        self.B = B
