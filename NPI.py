import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



class ANN_MLP(nn.Module):

    "Use MLP as a surrogate brain"

    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim):
        super().__init__()
        self.func = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, output_dim),
        ).to(device)

    def forward(self, x):
        return self.func(x)



class ANN_CNN(nn.Module):

    "Use CNN as a surrogate brain"

    def __init__(self, in_channels, hidden_channels, out_channels, data_length):
        super().__init__()
        self.in_channels = in_channels
        self.data_length = data_length
        self.CNN = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size = 2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, out_channels, kernel_size = 2), 
        ).to(device)
        self.Linear = nn.Linear(out_channels * (data_length - 2), in_channels).to(device)

    def forward(self, x):
        x = x.view(-1, self.data_length, self.in_channels)
        x = x.permute(0, 2, 1)
        pred = self.CNN(x)
        pred = torch.squeeze(pred)
        pred = self.Linear(pred)
        return pred



class ANN_RNN(nn.Module):

    "Use RNN as a surrogate brain"
    
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim, data_length):
        super().__init__()
        self.input_dim = input_dim
        self.data_length = data_length
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        ).to(device)
        self.rnn = nn.RNN(input_size = 1, hidden_size = latent_dim, batch_first = True).to(device)
        self.output = nn.Linear(latent_dim, output_dim).to(device)

    def forward(self, x):
        x = x.view(-1, self.data_length, self.input_dim)
        encodes = self.enc(x)
        ht, _ = self.rnn(torch.zeros((x.shape[0], 1, 1)).to(device), torch.permute(encodes[:, self.data_length-1:, :], (1, 0, 2)).contiguous())
        return self.output(ht)[:,0,:]



class ANN_VAR(nn.Module):

    "Use VAR as a surrogate brain"

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.func = nn.Linear(input_dim, output_dim).to(device)
    
    def forward(self, x):
        return self.func(x)



def multi2one(time_series, steps):

    "Split the data into several input-output pairs"

    n_area = time_series.shape[1]
    n_step = time_series.shape[0]
    input_X = np.zeros((n_step - steps, n_area * steps))
    target_Y = np.zeros((n_step - steps, n_area))
    for i in range(n_step - steps):
        input_X[i] = time_series[i:steps+i].flatten()
        target_Y[i] = time_series[steps+i].flatten()
    return np.array(input_X), np.array(target_Y)



def train_NN(model, input_X, target_Y, batch_size = 50, train_set_proportion = 0.8, num_epochs = 100, lr = 1e-3, l2 = 0):

    "Use empirical data to tune the model"

    train_inputs = torch.tensor(input_X[:int(train_set_proportion * input_X.shape[0])], dtype = torch.float).to(device)
    train_targets = torch.tensor(target_Y[:int(train_set_proportion * target_Y.shape[0])], dtype = torch.float).to(device)
    test_inputs = torch.tensor(input_X[int(train_set_proportion * input_X.shape[0]):], dtype = torch.float).to(device)
    test_targets = torch.tensor(target_Y[int(train_set_proportion * target_Y.shape[0]):], dtype = torch.float).to(device)
    train_dataset = data.TensorDataset(train_inputs, train_targets)
    test_dataset = data.TensorDataset(test_inputs, test_targets)
    train_iter = data.DataLoader(train_dataset, batch_size, shuffle = True)
    test_iter = data.DataLoader(test_dataset, batch_size, shuffle = False)

    loss = nn.MSELoss()
    trainer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = l2)

    train_epoch_loss = []; test_epoch_loss = []
    for _ in range(num_epochs):
        model.train()
        for X, y in train_iter:
            y_hat = model(X)
            l = loss(y_hat, y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        model.eval()
        with torch.no_grad():
            total_loss = 0; total_num = 0
            for X, y in train_iter:
                y_hat = model(X)
                l = loss(y_hat, y)
                total_loss += l * y.shape[0]
                total_num += y.shape[0]
            train_epoch_loss.append(float(total_loss / total_num))
            total_loss = 0; total_num = 0
            for X, y in test_iter:
                y_hat = model(X)
                l = loss(y_hat, y)
                total_loss += l * y.shape[0]
                total_num += y.shape[0]
            test_epoch_loss.append(float(total_loss / total_num))
    return model, train_epoch_loss, test_epoch_loss



def corrcoef(signals):

    "Calculate FC of the empirical data"

    return torch.corrcoef(torch.tensor(signals.T, dtype = torch.float).to(device)).detach().cpu().numpy()



def model_FC(model, node_num, steps):

    "Simulate data with random noise and calculate model-FC"

    NN_sim = []
    for _ in range(steps): NN_sim.append(np.zeros(node_num))
    for _ in range(1200):
        noise = 0.1 * np.random.randn(steps * node_num)
        model_input = np.array(NN_sim[-steps:]).flatten() + noise
        if isinstance(model, ANN_RNN): NN_sim.append(model(torch.tensor(model_input, dtype = torch.float).to(device)).detach().cpu().numpy()[0])
        else: NN_sim.append(model(torch.tensor(model_input, dtype = torch.float).to(device)).detach().cpu().numpy())
    NN_sim = np.array(NN_sim)
    return corrcoef(NN_sim)



def model_EC(model, input_X, target_Y, pert_strength):

    "Infer EC by perturbing the surrogate brain"

    node_num = target_Y.shape[1]
    steps = int(input_X.shape[1] / node_num)
    NPI_EC = np.zeros((node_num, node_num))
    for node in range(node_num):
        unperturbed_output = model(torch.tensor(input_X, dtype = torch.float).to(device)).detach().cpu().numpy()
        perturbation = np.zeros((steps, node_num)); perturbation[-1, node] = pert_strength
        perturbed_output = model(torch.tensor(input_X + perturbation.flatten(), dtype = torch.float).to(device)).detach().cpu().numpy()
        NPI_EC[node] = np.mean(perturbed_output - unperturbed_output, axis = 0)
    return NPI_EC



def model_Jacobian(model, input_X, steps):

    "Calculate the Jacobian matrix of the model"

    node_num = int(input_X.shape[1] / steps)
    jacobian = np.zeros((node_num, node_num))
    model.train()
    for i in range(input_X.shape[0]): 
        if isinstance(model, ANN_RNN): jacobian += torch.autograd.functional.jacobian(model, torch.tensor(input_X[i], dtype = torch.float).to(device)).cpu().detach().numpy()[0, :, -node_num:]
        else: jacobian += torch.autograd.functional.jacobian(model, torch.tensor(input_X[i], dtype = torch.float).to(device)).cpu().detach().numpy()[:, -node_num:]
    model.eval()
    jacobian_EC = jacobian.T / input_X.shape[0]
    return jacobian_EC