import sys
from sklearn.model_selection import train_test_split
sys.path.append(".")
import numpy as np
import torch
from torch.autograd import grad
from network import DNN
from scipy.io import loadmat
import pandas as pd
import torch.nn as nn
import numpy as np
import os

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(1234)
np.random.seed(1234)

data = pd.DataFrame(pd.read_csv(r"D:\02_github\01_RF_sites\121_SOSdaymet568_OK.csv"))
u = data['SOS']
x = data.iloc[:,-46:]
# print(x.columns)
# 特征选择
# x = data.iloc[:, [2, 3, 4, 5, 6,7, -46, -45, -44, -43, -42, -41, -40, -39, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1]]
t = data.iloc[:, [2, 3, 4, 5, 6,7]]
# print(t.columns)

dayl = data.iloc[:,2]
# print(dayl)
prcp = data.iloc[:,3]
srad = data.iloc[:,4]
tmax = data.iloc[:,5]
tmin = data.iloc[:,6]
vp = data.iloc[:,7]


print(x.shape,t.shape)
print(u.shape)

ub = np.array([np.array(x).max(), np.array(t).max()])
lb = np.array([np.array(x).min(), np.array(t).min()])
print(ub,lb)



x_, t_ = x, t
# x_, t_ = np.asarray(x),np.asarray(t)
# print(x_.shape, t_.shape)

x_ = x_.values.reshape(-1, 1)
t_ = t_.values.reshape(-1, 1)
u_ = u.values.reshape(-1, 1)
print(x_.shape, t_.shape,u_.shape)


N_u = 200
rand_idx = np.random.choice(len(u), N_u, replace=False) #选择索引这行代码从 u_ 的索引中随机选择 N_u 个索引，并将结果存储在 rand_idx 中。这可能是为了从数据中选择部分样本用于训练或其他用途
# 所以这部分可以变为分为0.8 和 0.2的比例。
x = torch.tensor(x_[rand_idx], dtype=torch.float32).to(device)
t = torch.tensor(t_[rand_idx], dtype=torch.float32).to(device)
xt = torch.cat((x, t), dim=1)
print("xt",xt.shape)
u = torch.tensor(u_[rand_idx], dtype=torch.float32).to(device)
print(x.shape,t.shape)
print(u.shape,xt.shape)

# print(u)
# print(xt)



class PINN:
    def __init__(self, u):
        self.u = u
        self.lambda_1 = torch.tensor([0.1], requires_grad=True).to(device)
        self.lambda_2 = torch.tensor([0.3], requires_grad=True).to(device)
        self.lambda_3 = torch.tensor([0.1], requires_grad=True).to(device)
        self.lambda_4 = torch.tensor([0.3], requires_grad=True).to(device)
        self.lambda_5 = torch.tensor([0.3], requires_grad=True).to(device)
        self.lambda_6 = torch.tensor([0.1], requires_grad=True).to(device)

        self.lambda_1 = torch.nn.Parameter(self.lambda_1)
        self.lambda_2 = torch.nn.Parameter(self.lambda_2)
        self.lambda_3 = torch.nn.Parameter(self.lambda_2)
        self.lambda_4 = torch.nn.Parameter(self.lambda_2)
        self.lambda_5 = torch.nn.Parameter(self.lambda_2)
        self.lambda_6 = torch.nn.Parameter(self.lambda_2)

        self.net = DNN(dim_in=2, dim_out=1, n_layer=7, n_node=20, ub=ub, lb=lb,).to(device)
        self.net.register_parameter("lambda_1", self.lambda_1)
        self.net.register_parameter("lambda_2", self.lambda_2)
        self.net.register_parameter("lambda_2", self.lambda_3)
        self.net.register_parameter("lambda_2", self.lambda_4)
        self.net.register_parameter("lambda_2", self.lambda_5)
        self.net.register_parameter("lambda_2", self.lambda_6)
        # self.net.register_parameter("lambda_2", self.lambda_2)
        self.optimizer = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",
        )
        self.iter = 0

    def f(self, xt):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        lambda_3 = self.lambda_3
        lambda_4 = self.lambda_4
        lambda_5 = self.lambda_5
        lambda_6 = self.lambda_6
        # lambda_1 = self.lambda_1

        xt = xt.clone()
        xt.requires_grad = True

        u = self.net(xt)
        #
        # u_xt = grad(u.sum(), xt, create_graph=True)[0]
        # u_x = u_xt[:, 0:1]
        # u_t = u_xt[:, 1:2]
        #
        # u_xx = grad(u_x.sum(), xt, create_graph=True)[0][:, 0:1]

        # f = u_t + lambda_1 * u * u_x - lambda_2 * u_xx
        f = lambda_1 * dayl  + lambda_2 * prcp + lambda_3 * srad +lambda_4 *tmax+lambda_5 *tmin +lambda_6 *vp + x
        # Apply output bounds
        f = torch.clamp(f, 0, 365)
        return f


    def closure(self):
        self.optimizer.zero_grad()
        u_pred = self.net(xt)
        f_pred = self.f(xt)
        mse_u = torch.mean(torch.square(u_pred - self.u))
        mse_f = torch.mean(torch.square(f_pred))
        loss = mse_u + mse_f
        loss.backward()
        self.iter += 1
        print(
            f"\r{self.iter} loss : {loss.item():.3e} l1 : {self.lambda_1.item():.5f}, l2 : {torch.exp(self.lambda_2).item():.5f}",
            end="",
        )
        if self.iter % 500 == 0:
            print("")
        return loss

def calcError(pinn):
    u_pred = pinn.net(torch.hstack((x, t)))
    u_pred = u_pred.detach().cpu().numpy()
    u_ = u.detach().cpu().numpy()

    print("u_",u_)
    print("u_pred",u_pred)


    error_u = np.linalg.norm(u_ - u_pred, 2) / np.linalg.norm(u_, 2)
    lambda1 = pinn.lambda_1.detach().cpu().item()
    lambda2 = np.exp(pinn.lambda_2.detach().cpu().item())
    error_lambda1 = np.abs(lambda1 - 1.0) * 100
    error_lambda2 = np.abs(lambda2 - 0.01 / np.pi) * 100
    print(
        f"\nError u  : {error_u:.5e}",
        f"\nError l1 : {error_lambda1:.5f}%",
        f"\nError l2 : {error_lambda2:.5f}%",
    )
    return (error_u, error_lambda1, error_lambda2)

if __name__ == "__main__":
    pinn = PINN(u)
    pinn.optimizer.step(pinn.closure)
    torch.save(pinn.net.state_dict(), r"D:\02_github\03_PINNs\PINNs_tensorflow\weight_clean3.pt")
    pinn.net.load_state_dict(torch.load(r"D:\02_github\03_PINNs\PINNs_tensorflow\weight_clean3.pt"))
    calcError(pinn)

