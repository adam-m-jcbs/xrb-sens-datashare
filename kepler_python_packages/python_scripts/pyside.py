#!/usr/bin/python3
 
import sys
# from PySide.QtCore import *
from PySide.QtGui import *
 
class Window(QMainWindow):
   
    def __init__(self, parent = None):
        super().__init__(parent)
        self.setWindowTitle("My Form")
        self.edit = QLineEdit("Write my name here..")
        self.button = QPushButton("Show Greetings")

        layout = QVBoxLayout()
        layout.addWidget(self.edit)
        layout.addWidget(self.button)
        
        # Set dialog layout
        self.setLayout(layout)

        # Add button signal to greetings slot
        self.button.clicked.connect(self.greetings)

    # Greets the user
    def greetings(self):
        print("Hello %s" % self.edit.text())
        self.done(True)

def run():
    # Create the Qt Application
    try:
        app = QApplication(sys.argv)
    except:
        # qApp.exit()
        # app = QApplication(['x'])
        app = qApp
    # Create and show the form
    w = Window()
    w.show()
    # Run the main Qt loop
    # sys.exit(app.exec_())
    app.exec_()


if __name__ == '__main__':
    run()
    
