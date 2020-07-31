Nucleosyn Tutorial
==================

Create directories /home/user/python/project and home/user/python/source. The source one should be created when following Heger's setup method to get all of his .py files.

Within the project folder, create a dumps folder and place all 9.6SM and above znuc dump files within it.

Also place the znuc2012.S4.star.deciso.y.stardb.gz and znuc2012.S4.star.el.y.stardb.gz files within the project folder.

Import nucleosyn and project plots

To run nucleosyn simply execute ex=nucleosyn.Main()

Then the user can enter their own initial mass function by entering

def yourimf(m):
    return m**1.35  or other function
    
into the command line and running ex.enterIMF(IMF=yourimf)

or use the default Salpeter IMF by ex.enterIMF()

To run the project plots module just run ex.plots() and assign a variable, say y, like y=ex.plottingtool and then run y.desiredplothere

