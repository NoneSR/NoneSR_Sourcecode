import matplotlib.pyplot as plt
import matplotlib
import numpy as np
n=1024
Y=26.71,26.54,26.47,26.04,26.39,26.27,26.25,26.56,26.14,26.2,26.11
X=922,601,897,715,751,742,777,589,659,643,550

plt.scatter(X,Y,s=80,c='royalblue')#生成一个离散的 size=50,透明度为50%
plt.scatter(690,26.98,s=80,marker='p',c='r')
matplotlib.pyplot.text(700,26.98, s="Our New Work",fontsize=10)
# matplotlib.pyplot.text(736,26.98, s="(ours)",fontsize=8)
matplotlib.pyplot.text(892,26.71, s="EDT",fontsize=10)
# matplotlib.pyplot.text(850,26.7, s="MAN",fontsize=10)
matplotlib.pyplot.text(611,26.54, s="ELAN",fontsize=10)
matplotlib.pyplot.text(852,26.47, s="SwinIR",fontsize=10)
matplotlib.pyplot.text(725,26.04, s="IMDN",fontsize=10)
matplotlib.pyplot.text(761,26.39, s="ESRT",fontsize=10)
matplotlib.pyplot.text(752,26.285, s="LBNet",fontsize=10)
matplotlib.pyplot.text(787,26.25, s="LatticeNet",fontsize=10)
matplotlib.pyplot.text(595,26.59, s="ESWT",fontsize=10)
matplotlib.pyplot.text(669,26.14, s="LAPAR",fontsize=10)
matplotlib.pyplot.text(653,26.2, s="RFDN-L",fontsize=10)
matplotlib.pyplot.text(560,26.11, s="RFDN",fontsize=10)

# matplotlib.pyplot.text(340,25.79, s="NLRG",fontsize=10)

plt.title('PSNR vs. Params[K]', fontsize=15)
plt.xlabel("Parameters (K)",fontsize=12)
plt.ylabel("PSNR(dB)",fontsize=12)
plt.grid(True)
plt.savefig(fname='lsr')
plt.show()

