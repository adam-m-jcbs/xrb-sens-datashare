#!/usr/bin/env python3

import matplotlib

from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler


from matplotlib.figure import Figure

import sys
import tkinter as Tk


def window():
    matplotlib.use('TkAgg')

    root = Tk.Tk()
    root.wm_title("Embedding in TK")
#root.bind("<Destroy>", destroy)

    f = Figure(figsize=(5,4))
    a = f.add_subplot(111)
    t = arange(0.0,3.0,0.01)
    s = sin(2*pi*t)

    a.plot(t,s)


    # a tk.DrawingArea
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.show()
    canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

    #toolbar = NavigationToolbar2TkAgg( canvas, root )
    #toolbar.update()
    canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

    def on_key_event(event):
        print('you pressed %s'%event.key)
        key_press_handler(event, canvas, toolbar)

    canvas.mpl_connect('key_press_event', on_key_event)

    def _quit():
        root.quit()     # stops mainloop
        root.destroy()  # this is necessary on Windows to prevent
                        # Fatal Python Error: PyEval_RestoreThread: NULL tstate

    button = Tk.Button(master=root, text='Quit', command=_quit)
    button.pack(side=Tk.BOTTOM)

    def OnClick(event):
        if event.dblclick:
            print("DBLCLICK", event)
        else:
            print("DOWN    ", event)


    def OnRelease(event):
        print("UP      ", event)


    r2 = Tk.Tk()
    r2.wm_title("Another Window")
    b2 = Tk.Button(r2, text="QUIT", fg="red",
         command = r2.destroy)
    b2.pack()


    cid_up = canvas.mpl_connect('button_press_event', OnClick)
    cid_down = canvas.mpl_connect('button_release_event', OnRelease)





    Tk.mainloop()
    # If you put root.destroy() here, it will cause an error if
    # the window is closed with the window manager.
    # root.destroy()


if __name__ == '__main__':
    window()
