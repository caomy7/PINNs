import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from scipy.io import loadmat
import pandas as pd
from sklearn.model_selection import train_test_split
from network import DNN
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1234)
np.random.seed(1234)

data = pd.DataFrame(pd.read_csv(r"D:\02_github\01_RF_sites\121_SOSdaymet568_OK.csv"))
u = data['SOS']
x = data.iloc[:, -46:]
t = data.iloc[:, [2, 3, 4, 5, 6, 7]]

dayl = data.iloc[:, 2]
prcp = data.iloc[:, 3]
srad = data.iloc[:, 4]
tmax = data.iloc[:, 5]
tmin = data.iloc[:, 6]
vp = data.iloc[:, 7]
dayl = torch.tensor(dayl.values).to(device)
prcp = torch.tensor(prcp.values).to(device)
srad = torch.tensor(srad.values).to(device)
tmax = torch.tensor(tmax.values).to(device)
tmin = torch.tensor(tmin.values).to(device)
vp = torch.tensor(vp.values).to(device)

ub = np.array([np.array(x).max(), np.array(t).max()])
lb = np.array([np.array(x).min(), np.array(t).min()])
x_, t_ = x.values.reshape(-1, 1), t.values.reshape(-1, 1)
u_ = u.values.reshape(-1, 1)

scaler = StandardScaler()
t = scaler.fit_transform(t)
x = scaler.fit_transform(x)
u = scaler.fit_transform(u.values.reshape(-1, 1)).flatten()


# N_u = 200
# print(len(u_), N_u)
x = torch.Tensor(x)
t = torch.Tensor(t)
u = torch.Tensor(u)
# rand_idx = np.random.choice(len(u_), N_u, replace=False)
# x = torch.tensor(x_[rand_idx], dtype=torch.float32).to(device)
# t = torch.tensor(t_[rand_idx], dtype=torch.float32).to(device)
xt = torch.cat((x, t), dim=1)
# u = torch.tensor(u_[rand_idx], dtype=torch.float32).to(device)
# print(x.shape,t.shape)
# print(u.shape,xt.shape)

# 划分训练集和测试集
xt, x_test, u, y_test = train_test_split(xt, u, test_size=0.2, random_state=2)

# 转换数据为Tensor
xt = torch.Tensor(xt)
u = torch.Tensor(u)
x_test_tensor = torch.Tensor(x_test)
y_test_tensor = torch.Tensor(y_test)
print(x.shape,u.shape,t.shape)
print(x_.shape,u.shape,t_.shape)
# 定义模型
input_size = xt.shape[1] #52
print(input_size)   #52



class DNN(nn.Module):
    def __init__(self, dim_in, dim_out, n_layer, n_node, ub, lb):
        super(DNN, self).__init__()
        self.ub = torch.nn.Parameter(torch.tensor(ub, dtype=torch.float32))
        self.lb = torch.nn.Parameter(torch.tensor(lb, dtype=torch.float32))
        self.n_layer = n_layer
        self.n_node = n_node
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(dim_in, n_node))
        for _ in range(n_layer - 1):
            self.layers.append(nn.Linear(n_node, n_node))
        self.layers.append(nn.Linear(n_node, dim_out))


    def forward(self, x):
        print(x.shape,lb.shape,ub.shape)  #x:454x52, lb为2
        x = x.reshape[-1,1]
        z = (x_ - self.lb) / (self.ub - self.lb)   # Rescale input to [-1, 1]
        for i in range(len(self.layers) - 1):
            z = torch.sin(self.layers[i](z))
        z = self.layers[-1](z)
        return z


class PINN:
    def __init__(self, u):
        self.u = u
        self.lambda_1 = torch.tensor([0.1], requires_grad=True).to(device)
        self.lambda_2 = torch.tensor([0.3], requires_grad=True).to(device)
        self.lambda_3 = torch.tensor([0.3], requires_grad=True).to(device)
        self.lambda_4 = torch.tensor([0.1], requires_grad=True).to(device)
        self.lambda_5 = torch.tensor([0.3], requires_grad=True).to(device)
        self.lambda_6 = torch.tensor([0.1], requires_grad=True).to(device)

        self.lambda_1 = torch.nn.Parameter(self.lambda_1)
        self.lambda_2 = torch.nn.Parameter(self.lambda_2)
        self.lambda_3 = torch.nn.Parameter(self.lambda_3)
        self.lambda_4 = torch.nn.Parameter(self.lambda_4)
        self.lambda_5 = torch.nn.Parameter(self.lambda_5)
        self.lambda_6 = torch.nn.Parameter(self.lambda_6)

        self.net = DNN(dim_in=input_size, dim_out=1, n_layer=7, n_node=20, ub=ub, lb=lb).to(device)
        self.net.register_parameter("lambda_1", self.lambda_1)
        self.net.register_parameter("lambda_2", self.lambda_2)
        self.net.register_parameter("lambda_3", self.lambda_3)
        self.net.register_parameter("lambda_4", self.lambda_4)
        self.net.register_parameter("lambda_5", self.lambda_5)
        self.net.register_parameter("lambda_6", self.lambda_6)

        self.optimizer = torch.optim.LBFGS(
            self.net.parameters(),
            # lr=1.0,
            lr=0.01,
            max_iter=5000,
            max_eval=5000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",  # line_search_fn="strong_wolfe",
        )
        self.iter = 0

    def f(self, xt):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        lambda_3 = self.lambda_3
        lambda_4 = self.lambda_4
        lambda_5 = self.lambda_5
        lambda_6 = self.lambda_6

        xt = xt.clone()
        xt.requires_grad = True
        u = self.net(xt)

        print("x:",x)
        lambda_2>lambda_3

        f = x+ lambda_1 * dayl + lambda_2 * prcp + lambda_3 * srad + lambda_4 * tmax + lambda_5 * tmin + lambda_6 * vp
        f = torch.clamp(f, 0, 365)
        return f

    def closure(self):
        self.optimizer.zero_grad()
        u_pred = self.net(xt)
        # f_pred = self.f(xt)
        # print(f_pred.shape)   #[200, 568]
        # print(u_pred.shape)  #[200, 1]

        mse_u = torch.mean(torch.square(u_pred - self.u))
        # mse_f = torch.mean(torch.square(f_pred))
        # loss = mse_u + mse_f
        loss = mse_u
        loss.backward()
        self.iter += 1
        print(
            f"\r{self.iter} loss: {loss.item():.3e}, l1: {self.lambda_1.item():.5f}, l2: {torch.exp(self.lambda_2).item():.5f}",
            end=""
        )
        if self.iter % 500 == 0:
            print("")
        return loss
def calcError(pinn):
    u_pred = pinn.net(torch.hstack((x, t)))
    u_pred = u_pred.detach().cpu().numpy()
    u_ = u.detach().cpu().numpy()

    error_u = np.linalg.norm(u_ - u_pred, 2) / np.linalg.norm(u_, 2)
    lambda1 = pinn.lambda_1.detach().cpu().item()
    lambda2 = np.exp(pinn.lambda_2.detach().cpu().item())
    error_lambda1 = np.abs(lambda1 - 1.0) * 100
    error_lambda2 = np.abs(lambda2 - 0.01 / np.pi) * 100
    print(
        f"\nError u : {error_u:.5e}",
        f"\nError l1 : {error_lambda1:.5f}%",
        f"\nError l2 : {error_lambda2:.5f}%",
    )
    return (error_u, error_lambda1, error_lambda2)
    # print("u_pred", u_pred)


num_epochs = 200
batch_size = 32

train_dataset = TensorDataset(xt, u)
train_loader = DataLoader(train_dataset, batch_size=batch_size)

if __name__ == "__main__":
    for epoch in range(num_epochs):
        for batch_inputs, batch_labels in train_loader:
            pinn = PINN(u)
            pinn.optimizer.step(pinn.closure)
            calcError(pinn)


# if __name__ == "__main__":
#     pinn = PINN(u)
#     pinn.optimizer.step(pinn.closure)
#     torch.save(pinn.net.state_dict(), r"D:\02_github\03_PINNs\PINNs_tensorflow\weight_clean3.pt")
#     pinn.net.load_state_dict(torch.load(r"D:\02_github\03_PINNs\PINNs_tensorflow\weight_clean3.pt"))
#     calcError(pinn)

# print("u_", u_)
# print("u_pred", u_pred)

# print(pinn)


#
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
sns.set_style("darkgrid")

with torch.no_grad():
    u_pred = pinn.net(xt.unsqueeze(1)).cpu().numpy()

u = u.cpu().numpy()
print(u_.shape,u.shape,u_pred.shape)
print(u,u_pred)
# print(r2_score(u, u_pred))
# print(mean_squared_error(u, u_pred))

# 显示散点图
plt.scatter(u, u_pred)
plt.xlabel('Field-observed SOS')
plt.ylabel('Predict SOS')
plt.title('Scatter plot of True vs Predicted')
plt.show()

# plt.xlim(50,220,'R-squared='+str(r2_score(u, u_pred))[:4],fontdict={'size':16})
# plt.xlim(50,200,'RMSE='+str(mean_squared_error(u, u_pred)**0.5)[:5],fontdict={'size':16})
#
#
# # ax1.set_xlim(0,365)
# # ax1.set_ylim(0,365)
# plt.set_ylabel('Predict SOS')
# plt.set_xlabel('Field-observed SOS')


# fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(16,8))
# sns.regplot(x=y_train,y=y_predict_train,ax=ax1)
#
# ax1.text(50,220,'R-squared='+str(r2_score(y_train,y_predict_train))[:4],fontdict={'size':16})
# ax1.text(50,200,'RMSE='+str(mean_squared_error(y_train,y_predict_train)**0.5)[:5],fontdict={'size':16})
# # ax1.set_xlim(0,365)
# # ax1.set_ylim(0,365)
# ax1.set_ylabel('Predict SOS')
# ax1.set_xlabel('Field-observed SOS')
#
# sns.regplot(x=y_test,y=y_predict_test,ax=ax2)
# ax2.text(50,180,'R-squared='+str(r2_score(y_test,y_predict_test))[:4],fontdict={'size':16})
# ax2.text(50,160,'RMSE='+str(mean_squared_error(y_test,y_predict_test)**0.5)[:5],fontdict={'size':16})
# # ax2.set_xlim(0,365)
# # ax2.set_ylim(0,365)
#
# ax2.set_ylabel('Predict SOS')
# ax2.set_xlabel('Field-observed SOS')
#
# # plt.savefig(r'D:\02_github\02_ANN\01_result\03_ANN_PhenoCam_SOS.png', dpi=300)
# plt.savefig(r'D:\02_github\02_ANN\01_result\05_lstm_PhenoCam_climate_SOS.png', dpi=300)
#
# plt.show()
