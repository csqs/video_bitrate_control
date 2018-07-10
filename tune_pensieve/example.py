import numpy as np

prob =[1, 0, 6, 0.5, 6, 0.6]
index = np.min(np.argmax(prob))
state = np.zeros([6,8])
state[0]=[1,2,3,4,5,6,7,8]
state[1]=[11,12,13,14,15,16,17,18]
state[2]=[21,22,23,24,25,26,27,28]
state[3]=[31,32,33,34,35,36,37,38]
state[4]=[41,42,43,44,45,46,47,48]
state[5]=[51,52,53,54,55,56,57,58]

state = np.roll(state, -1, axis=1)
print state
print state[0:1, -1] # 0 last one
print state[1:2, -1] # 1 last one
print state[2:3, :]  # 2 all
print state[3:4, :]  # 3 all
print state[4:5, :6]  # 4 start 6 elements
print state[4:5, -1]  # 4 last one
print state[5:6, -1]  # 5 last one
state[5, -1] =50
print state[4:5, -1]  # 4 last one
print state[5:6, -1]  # 5 last one

# v_batch = np.zeros([6,1])
v_batch =[[1],[2],[3],[4],[5],[6]]
print v_batch
v_batch = np.arange(6).reshape(6,1)
print v_batch
print v_batch[-1, 0]
