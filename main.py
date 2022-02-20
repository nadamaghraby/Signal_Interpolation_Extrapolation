import math
import pathlib
import os
import pyqtgraph as pg
import pandas as pd
from gui import Ui_MainWindow
from threading import *
from PyQt5 import QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5 import QtWidgets
from PyQt5.QtCore import QObject, QThread, pyqtSignal
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')


# class to intalize a canvas for the error map and latex equation

class MatplotlibCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, dpi=10):
        self.fig = Figure(dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)
        self.fig.tight_layout()

    def returnAxes(self, w, h):
        axes = self.fig.add_subplot(111)
        return axes

# class for the parallel thread used for calculating the err map


class Worker(QObject):
    finished = pyqtSignal
    prograss = pyqtSignal(int)

    def calculateErrMap(self, data, time):
        self.ui.GenerateErrorMapButton.setText("Cancel")
        # calc err map using 2 loops interms of X parameter & y parameter
        maxNumberOfChunks = self.getMaxNoOFChunks()
        maxDegreeOfPolynomial = self.getHighestDegOfPolynomial()
        maxOverlapPercentage = self.getMaxOverlapPercentage()
        if maxNumberOfChunks > 1 and maxDegreeOfPolynomial > 1:
            ErrorMatrix = []
            timeValues = time
            dataValues = data
            # print(self.ui.OverlappingChoice.value())

            x_axis, _, _ = switchAxes(self)
            _, y_axis, _ = switchAxes(self)

            if(self.ui.ErrorMapXaxis.currentText() != "overlapping percentage" or self.ui.ErrorMapYaxis.currentText() != "overlapping percentage"):
                for i in range(1, x_axis):
                    list = []
                    for j in range(1, y_axis):
                        if(self.ui.OverlappingChoice.value() != 0):
                            timeChunksValues = self.chunks_overlap(
                                timeValues, self.ui.NumberOfChunks.value(), self.ui.OverlappingChoice.value())
                            dataChunksValues = self.chunks_overlap(
                                dataValues, self.ui.NumberOfChunks.value(), self.ui.OverlappingChoice.value())
                            #print(f"overaplapTime: {timeChunksValues}")

                        else:
                            timeChunksValues = np.array_split(timeValues, i)
                            dataChunksValues = np.array_split(dataValues, i)

                        # print(f"x values length = {len(self.time)}")
                        # print(f"y values length = {len(self.data)}")

                        errlist, _, _ = self.interpolate_signal(
                            timeChunksValues, dataChunksValues, j)
                        print(f"the err of {j} is{errlist}")

                        # print(
                        #     f"x values length After Interpolation = {len(self.time)}")
                        # print(
                        #     f"y values length After Interpolation= {len(self.data)}")
                        if(len(errlist) != 0):
                            err = sum(errlist)/len(errlist)
                            err *= 100
                            list.append(err)
                    ErrorMatrix.append(list)
                    self.ui.progressBar.setValue(
                        int((i/(x_axis-1))*100))
                ErrorMatrix = np.array(ErrorMatrix)

            else:

                for i in range(x_axis):
                    list = []
                    for j in range(y_axis):
                        err = self.getOverlapError(
                            x_axis, y_axis)
                        err *= 100
                        list.append(err)
                    ErrorMatrix.append(list)
                    if((x_axis == maxDegreeOfPolynomial or x_axis == maxDegreeOfPolynomial) and y_axis == maxOverlapPercentage):
                        ErrorMatrix = ErrorMatrix
                    elif(y_axis == maxDegreeOfPolynomial and (x_axis == maxOverlapPercentage or x_axis == maxDegreeOfPolynomial)):
                        ErrorMatrix = np.array(ErrorMatrix).transpose()
                    self.ui.progressBar.setValue(int((i/(x_axis-1))*100))
            print(x_axis, y_axis)
            return np.array(ErrorMatrix)


def switchAxes(self):
    maxNumberOfChunks = self.ui.ErrorNumberOfChunks.value()
    maxDegreeOfPolynomial = self.ui.ErrorPolynomialOrder.value()
    maxPercentageOfOverlap = self.ui.OverlappingChoice.value()
    casesDictionary = {"number of chunks": maxNumberOfChunks,
                       "order of the fitting polynomial": maxDegreeOfPolynomial, "overlapping percentage": maxPercentageOfOverlap}
    for axis in casesDictionary:
        if(self.ui.ErrorMapXaxis.currentText() == axis):
            x_axis = casesDictionary[axis]
        if(self.ui.ErrorMapYaxis.currentText() == axis):
            y_axis = casesDictionary[axis]
        else:  
            z_axis = casesDictionary[axis]
    return x_axis, y_axis, z_axis


# class definition for application window like ui


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.Open.triggered.connect(self.open)

        self.latexEquationCanvas = MatplotlibCanvas(self)
        self.ErrorMapCanvas = MatplotlibCanvas(self, 60)
        self.latexEquation = ""
        self.maxNumberOfChunks = 20
        self.maxDegreeOfPolynomial = 10
        self.errorMapToggle = 0

        self.ui.NumberOfChunks.valueChanged.connect(
            self.draw_interpolated_signal)
        self.ui.OrderOfFittingPolynomial.valueChanged.connect(
            self.draw_interpolated_signal)
        self.ui.EquationChoice.valueChanged.connect(self.equation_displayed)
        self.ui.PercentageOfSignalInFitting.valueChanged.connect(
            self.extrapolate_signal)

        self.ui.GenerateErrorMapButton.setCheckable(True)
        self.ui.progressBar.setVisible(True)

        self.ui.GenerateErrorMapButton.clicked.connect(
            self.generateOrCancelErrMap)

    def open(self):
        files_name = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open only CSV', os.getenv('HOME'), "csv(*.csv)")
        path = files_name[0]

        if pathlib.Path(path).suffix == ".csv":
            fileData = pd.read_csv(path)
            self.time = list(fileData.values[:, 0])
            self.data = list(fileData.values[:, 1])
            self.ui.SignalGraph.clear()
            self.ui.SignalGraph.plot(
                self.time, self.data, pen=pg.mkPen(color=(0, 0, 255)))

    def getOverlapError(self, polynomialDegree, percentageOfOverlap):
        time_chunks_ov = []
        time_chunks_wov = []  # wov->without overlap
        yerrors_ov = []
        yerrors_wov = []  # wov->without overlap
        total_error = []
        polynomialCoefficients = []
        time = np.copy(self.time)
        dataChunksValues = self.getChunksValues()
        timeChunksValues = np.array_split(time, len(dataChunksValues))
        for i in range(len(timeChunksValues)-1):
            ratio = math.ceil(
                ((percentageOfOverlap/100)(len(timeChunksValues[i]))))
            index = -1-ratio
            if i == 0:
                time_chunk1_wov = timeChunksValues[i][:index]
                polynomialCoefficients = np.polyfit(
                    timeChunksValues[i], dataChunksValues[i], polynomialDegree, full=True)
                yerror_ch1 = np.polyval(
                    polynomialCoefficients[i], time_chunk1_wov)
                residuals_ch1 = sum(
                    math.pow(yerror_ch1 - dataChunksValues[i][:index], 2))
                std_ch1 = np.std(dataChunksValues[i][:index])
                deviation_ch1 = math.pow(std_ch1, 2)*len(dataChunksValues[i])
                rSquared_ch1 = (deviation_ch1 - residuals_ch1)/deviation_ch1
                yerror1_wov = float(1 - rSquared_ch1)
                yerrors_wov.append(yerror1_wov)

            time_chunks_ov = timeChunksValues[i][index:]
            time_chunks_wov = timeChunksValues[i+1][ratio:index]
            yerror_eq1 = np.polyval(polynomialCoefficients[i], time_chunks_ov)
            yerror_eq2 = np.polyval(
                polynomialCoefficients[i+1], time_chunks_ov)
            yerror_ch2 = np.polyval(
                polynomialCoefficients[i+1], time_chunks_wov)
            yerror = (yerror_eq1+yerror_eq2)/2
            std_ov1 = np.std(dataChunksValues[i][index:])
            residuals_ov1 = sum(
                math.pow(std_ov1 - dataChunksValues[i][index:], 2))
            deviation_ov1 = math.pow(std_ov1, 2) * \
                len(dataChunksValues[i][index:])
            rSquared_ov1 = (deviation_ov1 - residuals_ov1)/deviation_ov1
            yerror_ov = float(1 - rSquared_ov1)
            yerrors_ov.append(yerror_ov)
            std_ch2 = np.std(dataChunksValues[i+1][ratio:index])
            residuals_ch2 = sum(
                math.pow(yerror_ch2 - dataChunksValues[i+1][ratio:index], 2))
            deviation_ch2 = math.pow(std_ch2, 2) * \
                len(dataChunksValues[i+1][ratio:index])
            rSquared_ch2 = (deviation_ch2 - residuals_ch2)/deviation_ch2
            yerror2_wov = float(1 - rSquared_ch2)
            yerrors_wov.append(yerror2_wov)

        total_error = (sum(yerrors_wov)+sum(yerrors_ov)) / \
            (2*len(dataChunksValues)-1)
        return total_error

    def getChunksValues(self):
        data = np.copy(self.data)
        self.dataChunksValues = np.array_split(
            data, self.ui.NumberOfChunks.value())
        return self.dataChunksValues

    def interpolate_signal(self, timeChunks, dataChunks, OrderOfPolynomial):

        self.polynomialCoefficientsValues = []
        self.InterpolatedDataValues = []
        self.errorOfChunks = []
        for chunk in range(len(dataChunks)):
            polynomialCoefficients, residuals, _, _, _ = np.polyfit(
                timeChunks[chunk], dataChunks[chunk], OrderOfPolynomial, full=True)
            interpolatedData = np.polyval(
                polynomialCoefficients, timeChunks[chunk])
            self.InterpolatedDataValues = np.concatenate(
                (self.InterpolatedDataValues, interpolatedData))
            self.polynomialCoefficientsValues.append(polynomialCoefficients)
            if not len(residuals) > 0:
                residuals = 0
            error = math.sqrt(residuals/len(dataChunks[chunk]))
            self.errorOfChunks.append(error)
        return (self.errorOfChunks, self.InterpolatedDataValues, self.polynomialCoefficientsValues)

    def draw_interpolated_signal(self):
        time = np.copy(self.time)
        data = np.copy(self.data)
        self.timeChunksValues = np.array_split(
            time, self.ui.NumberOfChunks.value())
        self.dataChunksValues = np.array_split(
            data, self.ui.NumberOfChunks.value())
        _, self.newYvalues, _ = self.interpolate_signal(
            self.timeChunksValues, self.dataChunksValues, self.ui.OrderOfFittingPolynomial.value())
        self.ui.SignalGraph.clear()
        self.ui.SignalGraph.addLegend()
        self.ui.SignalGraph.plot(self.time, self.data,
                                 pen=pg.mkPen(color=(0, 0, 255)), name="Original Signal")
        self.ui.SignalGraph.plot(
            self.time, self.newYvalues, pen=pg.mkPen(color=(255, 0, 0)), name="Interpolated Signal")

    def getMaxNoOFChunks(self):
        return self.ui.ErrorNumberOfChunks.value()

    def getHighestDegOfPolynomial(self):
        return self.ui.ErrorPolynomialOrder.value()

    def getMaxOverlapPercentage(self):
        return self.ui.OverlappingChoice.value()

    def plotErrorMap(self):

        self.maxNumberOfChunks = self.ui.ErrorNumberOfChunks.value()
        self.maxDegreeOfPolynomial = self.ui.ErrorPolynomialOrder.value()
        self.ErrorMapdata = Worker.calculateErrMap(
            self, np.copy(self.data), np.copy(self.time))

        self.ErrorMapCanvas.axes.cla()
        self.im = self.ErrorMapCanvas.axes.imshow(self.ErrorMapdata, cmap='Reds', extent=[
            0.5, (self.maxNumberOfChunks+0.5), 0.5, (self.maxDegreeOfPolynomial+0.5)], origin='lower')
        self.ErrorMapCanvas.axes.set_yticks(
            np.arange(1, self.maxDegreeOfPolynomial+1))
        self.ErrorMapCanvas.axes.set_yticklabels(
            np.arange(1, self.maxDegreeOfPolynomial+1))
        self.ErrorMapCanvas.axes.set_xticks(
            np.arange(1, self.maxNumberOfChunks+1))
        self.ErrorMapCanvas.axes.set_xticklabels(
            np.arange(1, self.maxNumberOfChunks+1))
        self.ErrorMapCanvas.axes.set_xlabel(
            self.ui.ErrorMapXaxis.currentText())
        self.ErrorMapCanvas.axes.set_ylabel(
            self.ui.ErrorMapYaxis.currentText())
        self.cbar = self.ErrorMapCanvas.fig.colorbar(self.im)
        self.ErrorMapCanvas.draw()

    def generateOrCancelErrMap(self):

        self.errorMapToggle ^= 1

        if self.errorMapToggle:
            self.ui.GenerateErrorMapButton.setText("Cancel")
            self.plotErrorMap()
            self.ui.ErrorMapLayout.addWidget(self.ErrorMapCanvas)
            self.ErrorMapCanvas.setVisible(True)


        else:
            self.ui.GenerateErrorMapButton.setText("Generate ErrorMap")
            self.ui.progressBar.setValue(0)
            self.ErrorMapCanvas.setVisible(False)
            self.cbar.remove()
            self.ui.ErrorMapLayout.removeWidget(self.ErrorMapCanvas)

    def chunks_overlap(self, array, numberOfChunks, overlappingPercentage):
        chunks = []
        startIndex = 0
        chunkLength = int(len(array)/(numberOfChunks -
                          ((overlappingPercentage/100)*(numberOfChunks - 1))))
        stepSize = math.ceil(chunkLength*(1-(overlappingPercentage/100)))
        for startIndex in range(numberOfChunks):
            chunk = array[startIndex:startIndex+stepSize]
            chunks.append(np.array(chunk))
            startIndex += stepSize

        return chunks

    def equation_displayed(self):
        coefficients = self.polynomialCoefficientsValues[self.ui.EquationChoice.value(
        )-1]
        equation = "Y = "
        for value in range(len(coefficients)):
            coefficientString = str(round(coefficients[value], 2))
            if coefficientString[0] == "-":
                equation += f"{coefficientString} X^{len(coefficients)-value-1} "
            else:
                equation += f"+ {coefficientString} X^{len(coefficients)-value-1} "
       
        if equation[4] == '+':
            equation = equation[:4] + equation[5:]
        
        equation = equation[:-4]

        error = round(
            self.errorOfChunks[self.ui.EquationChoice.value()-1]*100, 2)
        self.errorPercentage = f"Error = "+str(error) + "%"
        self.latexEquation = "$" + str(equation) + "$"
        self.ui.LatexEquationLayout.addWidget(self.latexEquationCanvas)
        self.latexEquationCanvas.fig.clear()
        self.latexEquationCanvas.fig.suptitle(f"{self.latexEquation}   {self.errorPercentage}", x=0, y=0.5,
                                              horizontalalignment='left', verticalalignment='center', fontsize=70)
        self.latexEquationCanvas.draw()

    def extrapolate_signal(self):
        numberOfPoints = round(
            (self.ui.PercentageOfSignalInFitting.value()/100) * len(self.time))
        self.extrapolationPolynomialCoefficients = np.polyfit(
            self.time[:numberOfPoints+1], self.data[:numberOfPoints+1], self.ui.OrderOfFittingPolynomial.value())
        self.extrapolatedData = np.polyval(
            self.extrapolationPolynomialCoefficients, self.time)
        self.ui.SignalGraph.clear()
        self.ui.SignalGraph.addLegend()
        self.ui.SignalGraph.plot(self.time, self.data, pen=pg.mkPen(
            color=(0, 0, 255)), name="Original Signal")
        self.ui.SignalGraph.plot(self.time, self.newYvalues, pen=pg.mkPen(
            color=(255, 0, 0)), name="Interpolated Signal")
        self.ui.SignalGraph.plot(self.time, self.extrapolatedData, pen=pg.mkPen(
            color=(255, 255, 100)), name="Extrapolated Signal")


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = ApplicationWindow()
    MainWindow.show()
    sys.exit(app.exec_())
