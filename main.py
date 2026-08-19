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

target = np.array([0,0])
pur_sp = np.array([[30]] * n)
eva_sp = np.array([[29]] * m)
print("Pursuer speed: \n",pur_sp)
print("Evader speed: \n",eva_sp)
print("pursuer positon: \n",pur_pos)
print("Evader positon: \n",ev_pos)

asgn = assign.assignment(pur_pos,ev_pos,target,v,pur_sp,eva_sp)
asgn.check_win()
    
