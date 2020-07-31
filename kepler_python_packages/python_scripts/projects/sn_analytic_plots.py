#plotting
import sn_analytic
import matplotlib
import matplotlib.pyplot as plt
matplotlib.pyplot.ioff()


#to avoid reloading models
#need to set some constant a  = plots.initialise(), then include a as an argument in 	plotting function
def initialise():
    s = sn_analytic.SN()
    p = s.dumps
    return p

def plotting(p, eta_outflow = None, threshold = None, zeta = None):

	if not zeta:
		for i in range(10):
		 zeta = 0.20 + 0.01*i
		 s = sn_analytic.SN(p)
		 s.plotsave(zeta = zeta,
				eta_outflow = eta_outflow,
				threshold = threshold,
				)
		 print("plotted for zeta = {}".format(zeta))
		 filename = 'z{}_eo{}_t{}'.format(round(zeta,2),eta_outflow,threshold)
		 plt.savefig(filename, format='png')
		 plt.close()


	if not eta_outflow:
		for i in range(61):
		 eta_outflow = 0.30 + 0.01*i
		 s = sn_analytic.SN(p)
		 s.plotsave(zeta = zeta,
				eta_outflow = eta_outflow,
				threshold = threshold,
				)
		 print("plotted for eta_outflow = {}".format(eta_outflow))
		 filename = 'z{}_eo{}_t{}'.format(round(zeta,2),round(eta_outflow,2),round(threshold,2))
		 plt.savefig(filename, format='png')
		 plt.close()


	if not threshold:
		for i in range():
		 threshold = 0.60 + 0.01*i
		 s = sn_analytic.SN(p)
		 s.plotsave(zeta = zeta,
				eta_outflow = eta_outflow,
				threshold = threshold,
				)
		 print("plotted for threshold = {}".format(threshold))
		 filename = 'z{}_eo{}_t{}'.format(round(zeta,2),round(eta_outflow,2),round(threshold,2))
		 plt.savefig(filename, format='png')
		 plt.close()
