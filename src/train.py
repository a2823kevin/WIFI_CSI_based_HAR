from random import shuffle
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch import optim
from utils import *
from models.LSTM import *
from models.TCN import *
from models.utils import *



def train_LSTM(device, dataset, data_length):

    input_size = 912
    learning_rate = 1e-3
    batch_size = 50
    num_epochs = 100

    test_dataset = dataset[len(dataset)*8//10:]
    train_dataset = dataset[0:len(dataset)*8//10]
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    model = LSTM(device, input_size, 7, 4, data_length)
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_min = 9999

    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            data = data.to(device=device)
            targets = targets.to(device=device)
            # forward
            scores = model(data)
            loss = criterion(scores, targets)
            print(loss)

            # backward
            optimizer.zero_grad()
            loss.backward()
            
            # gradient descent update step/adam step
            optimizer.step()
        print(f"epoch {epoch}:")
        #print(f"accuracy on training set: {check_accuracy(device, train_loader, model):2f}")
        loss = check_accuracy(device, test_loader, model)
        #print(f"accuracy on test set: {loss}")
        if (loss<loss_min):
            torch.save(model.state_dict(), "assets/trained_model/csi_lstm")
            loss_min = loss

def train_TCN(device, dataset, data_length):
    input_size = 456
    learning_rate = 1e-4
    batch_size = 50
    num_epochs = 100

    test_dataset = dataset[len(dataset)*8//10:]
    train_dataset = dataset[0:len(dataset)*8//10]
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    model = temporal_convolution_network(device, input_size, 2, data_length, [int(912-905*0.2*i) for i in range(1, 6)])
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_min = 9999

    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            data = data.to(device=device)
            targets = targets.to(device=device)
            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()
            
            # gradient descent update step/adam step
            optimizer.step()
        print(f"epoch {epoch}:")
        print(f"accuracy on training set: {check_accuracy(device, train_loader, model):2f}")
        loss = check_accuracy(device, test_loader, model)
        print(f"accuracy on test set: {loss}")
        if (loss<loss_min):
            torch.save(model.state_dict(), "assets/trained_model/csi_lstm")
            loss_min = loss

if __name__=="__main__":
    with open("src/settings.json", "r") as fin:
        settings = json.load(fin)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lst = []
    for file in os.listdir("assets/preprocessed_datasets"):
        if (file.endswith(".csv")):
            lst.append(generate_CSI_dataset(f"assets/preprocessed_datasets/{file}", settings, 25, "tcn"))
            break
    ds = merge_datasets(lst)

    train_TCN(device, ds, 25)