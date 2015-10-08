#author Frank Patterson
import matplotlib.pyplot as plt
f_c_file = 'examples/optLog_5Oct15_cDist_50g_popX15_65_07_cross3.log'
f_h_file = 'examples/optLog_5Oct15_hDist_50g_popX15_65_07_cross3.log'

f_c = open(f_c_file, 'r')
f_h = open(f_h_file, 'r')
#[[bests], [worsts], [avgs]]


cDists_c = [] #data for log for crowding dist
hDists_c = []
pPcts_c = []
cDists_h = [] #data for log for hamming dist
hDists_h = []
pPcts_h = []

for line in f_c:    
    if "Average Crowding Distance (parents): " in line:
        bf = float(line.split(":")[1].strip())
        cDists_c.append(bf)
    if "Average Hamming Distance (parents): " in line:
        bf = float(line.split(":")[1].strip())
        hDists_c.append(bf)
    if "Frontiers: 1: " in line and (not "Hamming" in line) and (not "Crowding" in line):
        bf = float(line.split(":")[2].split(",")[0].strip())
        print bf
        pPcts_c.append(bf)
pPcts_c = [p/max(pPcts_c) for p in pPcts_c]

for line in f_h:    
    if "Average Crowding Distance (parents): " in line:
        bf = float(line.split(":")[1].strip())
        cDists_h.append(bf)
    if "Average Hamming Distance (parents): " in line:
        bf = float(line.split(":")[1].strip())
        hDists_h.append(bf)
    if "Frontiers: 1: " in line and (not "Hamming" in line) and (not "Crowding" in line):
        bf = float(line.split(":")[2].split(",")[0].strip())
        print bf
        pPcts_h.append(bf)
pPcts_h = [p/max(pPcts_h) for p in pPcts_h]

plt.figure()
plt.subplot(3,1,1)
plt.plot(range(1,len(hDists_c)+1), hDists_c)
plt.plot(range(1,len(hDists_h)+1), hDists_h)
plt.text(10, 16, 'Pareto Frontier Avg. of Mean Hamming Distance', fontsize=11)
plt.ylabel('Hamming Dist.', fontsize=11) 
#plt.xlabel('Generation')
plt.legend(['Opt. w/ Crowding Dist.', 'Opt. w/ Hamming Dist.'], bbox_to_anchor=(0.8, 1.19), ncol=2, fontsize=11)
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(range(1,len(cDists_c)+1), cDists_c)
plt.plot(range(1,len(cDists_h)+1), cDists_h)
plt.text(10, (max(cDists_c)+min(cDists_c))/2, 'Pareto Frontier Average of Crowding Distance', fontsize=11)
plt.ylabel('Crowding Dist.', fontsize=11) 
#plt.xlabel('Generation')
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(range(1,len(pPcts_c)+1), pPcts_c)
plt.plot(range(1,len(pPcts_h)+1), pPcts_h)
plt.text(10, 0.5, 'Pareto Frontier as Percentage of Population', fontsize=11)
plt.ylabel('%')
plt.ylim([0,1.03])
plt.xlabel('Generation')
plt.grid(True)

plt.draw()
plt.show()