# signal-interpolation-extrapolation


- Curve interpolation and extrapolation using the number of chunks and polynomial degrees chosen by the user with a graphical representation of the error.
- Interpolation & Curve Fitting An application that illustrates the efficacy of different curve fitting and interpolation models where:

- The user can open and display an arbitrary signal (of reasonable length of 1000 points).
- The user can choose if the fitting to be done as one chunk or multiple ones and choose how many chunks the curve should be divided into.
- The user can choose the order of the fitting polynomial (either for the big chunk or the multiple ones).
- Upon any change in the selections, the fitting result is displayed in dotted line on the same graph that displays the original signal.
- The fitted equation and percentage error sis shown in its mathematical form above the main graph using Latex format.
- The user can generate an error map for the fitting process via clicking a button. The user can choose the x- and y- axis of the error map among: 1- number of chunks 2- - order of the fitting polynomial 3- overlapping between consecutive chunks (0-25% of the chunk size). The error values in the map are normalized.
- When the user clicks a button to initiate the error map, a progress bar should show the status of the generation process, and the clicked button turns into a “cancel” button where the user can click it to stop the process and return to the normal status.
- To illustrate the extrapolation efficacy, there is an option to clip the curve fitting process into only portion of the open signal. For example, apply the fitting process on only 50% or 60%,...or 90% of the signal. Then, the fitted curve should be plotted for the whole signal. i.e., the last portion is practically extrapolated
