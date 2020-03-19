import torch
from utils.utils import show_training_progress

class Trainer():
    def __init__(self, model, optimizer=None, loss=None, metric=None, train_data=None, validation_data=None, epochs=None, device=None, lr_scheduler=None):
        self.model = model
        self.optimizer = optimizer 
        self.loss = loss
        self.metric = metric
        self.train_data = train_data
        self.validation_data = validation_data
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.device = device

    def train(self):
        soft = nn.Softmax(dim = 1)
        for epoch in range(self.epochs):
            print("Epoch {}" .format(epoch))
            self.model.train()
            total_loss = 0
            y_true = None
            y_pred = None
            for i, (x, y) in enumerate(self.train_data):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_out = self.model(x)
                loss_value = self.loss(y_out, y.view(-1))
                loss_value.backward()
                self.optimizer.step()
                total_loss = total_loss + loss_value.item()
                if y_true == None:
                    y_true = y
                else:
                    y_true = torch.cat([y_true, y], dim = 0)
                y_out = soft(y_out)
                if y_pred == None:
                    y_pred = y_out
                else:
                    y_pred = torch.cat([y_pred, y_out], dim = 0)
                auc_value = self.metric(y_true, y_pred)
                show_training_progress(auc_value, i, len(self.train_data), True)
            if self.lr_scheduler != None:
                self.lr_scheduler.step(total_loss)
            print("")
            self.model.eval()
            total_loss = 0 
            y_true = None
            y_pred = None
            with torch.no_grad():
                for i, (x, y) in enumerate(self.validation_data):
                    x, y = x.to(self.device), y.to(self.device)
                    y_out = self.model(x)
                    loss_value = self.loss(y_out, y.view(-1))
                    total_loss = total_loss + loss_value.item()
                    if y_true == None:
                        y_true = y
                    else:
                        y_true = torch.cat([y_true, y], dim = 0)
                    y_out = soft(y_out)
                    if y_pred == None:
                        y_pred = y_out
                    else:
                        y_pred = torch.cat([y_pred, y_out], dim = 0)
                    auc_value = self.metric(y_true, y_pred)
                    show_training_progress(auc_value, i, len(self.validation_data), False)
            print("")

        def save(self, train_mode=False):
            torch.save(self.model.state_dict(), "model.pth")

        def load(self, train_mode=False):
            self.model.load_state_dict(torch.load("model.pth"))

if __name__ == "__main__":
    pass