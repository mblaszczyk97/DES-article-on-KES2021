import PySimpleGUI as sg
import method_checker as mc
from irregular_set import irregular_set as irrs
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
# Define the window layout
layout = [
          [sg.Text('Sprawdzanie zbiorów niezbalansowanych')],
          [sg.Frame('Techinka balansowania:',[[sg.Button('Yes'), sg.Button('No')]]), sg.Frame('Wykres:',[[sg.Button('Ok'), sg.Canvas(key='-CANVAS-')]])],
          ]

# Create the form and show it without the plot


matplotlib.use("TkAgg")

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg

# Add the plot to the window
set=irrs("glass-0-1-5_vs_2.csv")
X=set.X
y=set.y
plot1=mc.plotter(X, y, "SMOTE")

window = sg.Window('Wykresy', layout, finalize=True, element_justification='center', font='Helvetica 18')
draw_figure(window["-CANVAS-"].TKCanvas, plot1.fig())

event, values = window.read()
window.close()
