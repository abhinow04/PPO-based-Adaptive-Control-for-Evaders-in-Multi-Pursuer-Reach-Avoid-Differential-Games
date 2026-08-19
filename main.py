import numpy as np
import assign


N = 2 
n = 2 
m = 1 
v = np.ones(m) 
u = np.ones(n) 
r = 1 


pur_pos = 200 * np.random.randn(n,N)
ev_pos = 300 * np.random.randn(m,N)
#cas = int(input("Enter the case number (1, 2, or 3): "))
'''if(cas== 1):
    pur_pos = np.array([[-50,-50],[45,110]])
    ev_pos = np.array([[620,240]])

elif(cas== 2):
    pur_pos = np.array( [[  -8.48774851,-304.63018555],[-165.37522171 ,138.47735059]])
    ev_pos = np.array([[-158.37671857,-141.99147185]])
elif(cas== 3):
    pur_pos = np.array([[ 157.24053755,1.90787654],[  68.40871248,-220.1231542 ]])
    ev_pos = np.array([[ 145.01982303,-742.14543981]])'''

target = np.array([0,0])
pur_sp = np.array([[30]] * n)
eva_sp = np.array([[29]] * m)
print("Pursuer speed: \n",pur_sp)
print("Evader speed: \n",eva_sp)
print("pursuer positon: \n",pur_pos)
print("Evader positon: \n",ev_pos)

asgn = assign.assignment(pur_pos,ev_pos,target,v,pur_sp,eva_sp)
asgn.check_win()
    
