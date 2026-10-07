import time, mujoco, numpy as np, warp as wp, mujoco_warp as mjw, torch, sys
from reactor.robot import build_model, RobotInfo
sc = sys.argv[1]=='1'
m=build_model(self_collision=sc); r=RobotInfo(m)
N=4096
d0=mujoco.MjData(m); d0.qpos[:]=m.qpos0; d0.qpos[7:]=r.default_q; d0.qpos[2]=0.78; d0.ctrl[:]=r.default_q
mujoco.mj_forward(m,d0)
wm=mjw.put_model(m); wd=mjw.put_data(m,d0,nworld=N,nconmax=48,njmax=256)
def step4():
    for _ in range(4): mjw.step(wm,wd)
step4(); wp.synchronize()
with wp.ScopedCapture() as cap: step4()
g=cap.graph
t=time.time()
for i in range(200): wp.capture_launch(g)
wp.synchronize(); dt=time.time()-t
q=wp.to_torch(wd.qpos)
print('selfcol',sc,'ctrl steps/s', 200*N/dt, 'z mean', q[:,2].mean().item(), 'nacon', wd.nacon.numpy(), 'overflow?')
