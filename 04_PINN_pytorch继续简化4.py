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

data = loadmat(r"D:\02_github\03_PINNs\PINNs-torch\Burgers\burgers_shock.mat")
x = data["x"]
t = data["t"]

# s = data["usol"]
# print(s.shape)
u = data["usol"].T

print(x.shape,t.shape)
print(u.shape)

ub = np.array([x.max(), t.max()])
lb = np.array([x.min(), t.min()])
print(ub,lb)
# print(x.max(), t.max())

# x_, t_ = np.meshgrid(x, t)
df1 = pd.DataFrame(x)
df2 = pd.DataFrame(t)
xt = pd.concat([df1,df2], axis=1).reset_index(drop=True)
print(xt)

x_, t_ = np.asarray(x),np.asarray(t)
# print(x_.shape, t_.shape)

x_ = x_.reshape(-1, 1)
t_ = t_.reshape(-1, 1)
u_ = u.reshape(-1, 1)
print(x_.shape, t_.shape,u_.shape)

# N_u = 2000
# rand_idx = np.random.choice(len(u_), N_u, replace=False) #选择索引这行代码从 u_ 的索引中随机选择 N_u 个索引，并将结果存储在 rand_idx 中。这可能是为了从数据中选择部分样本用于训练或其他用途
# # 所以这部分可以变为分为0.8 和 0.2的比例。
# x = torch.tensor(x_[rand_idx], dtype=torch.float32).to(device)
# t = torch.tensor(t_[rand_idx], dtype=torch.float32).to(device)
# xt = torch.cat((x, t), dim=1)
# u = torch.tensor(u_[rand_idx], dtype=torch.float32).to(device)
print(x.shape,t.shape)
print(u.shape,xt.shape)

print(u)
print(xt)



class PINN:
    def __init__(self, u):
        self.u = u
        self.lambda_1 = torch.tensor([0.2], requires_grad=True).to(device)
        self.lambda_2 = torch.tensor([0.3], requires_grad=True).to(device)
        self.lambda_1 = torch.nn.Parameter(self.lambda_1)
        self.lambda_2 = torch.nn.Parameter(self.lambda_2)
        self.net = DNN(dim_in=2, dim_out=1, n_layer=7, n_node=20, ub=ub, lb=lb,).to(device)
        self.net.register_parameter("lambda_1", self.lambda_1)
        self.net.register_parameter("lambda_2", self.lambda_2)
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

    # def f(self, xt):
    #     xt = xt.clone()            #创建了输入 xt 的一个副本，以确保不会改变原始输入。
    #     xt.requires_grad = True    #对 xt 进行梯度计算。
    #     u = self.net(xt)       #对输入的x和t进行网络模拟预测为u.
    #     f = grad(u.sum(), xt, create_graph=True)[0]     #这行代码计算了 u 的和 u.sum() 对 xt 的梯度。grad() 函数是 PyTorch 提供的用于计算梯度的工具，其中 create_graph=True 表示我们希望为计算的梯度创建一个计算图，以便进行高阶梯度计算。[0] 表示我们仅获取梯度值而不是其他返回值。
    #     f = torch.clamp(f, 0, 365)      #使用torch.clamp函数对变量f进行限制，确保其取值在0到365之间
    #     return f

    def f(self, xt):
        lambda_1 = self.lambda_1
        lambda_2 = torch.exp(self.lambda_2)
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
        f = lambda_1 * u  + lambda_2 * u
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

