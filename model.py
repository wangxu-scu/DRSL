import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageDNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ImageDNN, self).__init__()
        self.Sequential = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        nn.BatchNorm1d(hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.BatchNorm1d(hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, output_dim),
                                        nn.BatchNorm1d(output_dim),
                                        nn.ReLU()
                                        )

    def forward(self, x):
        # x = F.leaky_relu(self.fc1(x), 0.05)
        # x = F.leaky_relu(self.fc2(x), 0.05)
        x = self.Sequential(x)
        return x

class TextDNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TextDNN, self).__init__()
        self.Sequential = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        nn.BatchNorm1d(hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.BatchNorm1d(hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, output_dim),
                                        nn.BatchNorm1d(output_dim),
                                        nn.ReLU()
                                        )

    def forward(self, x):
        # x = F.leaky_relu(self.fc1(x), 0.05)
        # x = F.leaky_relu(self.fc2(x), 0.05)
        x = self.Sequential(x)
        return x

class RelationDNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RelationDNN, self).__init__()
        self.Sequential = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        nn.BatchNorm1d(hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.BatchNorm1d(hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, output_dim)
                                        )

    def forward(self, x):
        x = self.Sequential(x)
        # x = F.sigmoid(x)
        return x


class Model(nn.Module):
    def __init__(
            self,
            input_dim_I,
            input_dim_T,
            hidden_dim_I,
            hidden_dim_T,
            hidden_dim_R,
            output_dim_I,
            output_dim_T,
            output_dim_R):
        super(Model, self).__init__()

        self.ImageDNN = ImageDNN(input_dim_I, hidden_dim_I, output_dim_I)
        self.TextDNN = TextDNN(input_dim_T, hidden_dim_T, output_dim_T)
        self.RelationDNN = RelationDNN(output_dim_I + output_dim_T, hidden_dim_R, output_dim_R)
        # self.RelationDNN = RelationDNN(input_dim_I + input_dim_T, hidden_dim_R, output_dim_R)

    def forward(self, img, text, return_relation_score=True):
        # Image Pathway
        y_I = self.ImageDNN(img)
        # y_I = img

        # Text Pathway
        y_T = self.TextDNN(text)
        # y_T = text

        if return_relation_score is False:
            return y_I, y_T
        relation_score = self.cal_relation_score(y_I, y_T)

        return relation_score

    def cal_relation_score(self, y_I, y_T):
        ni = y_I.size(0)
        di = y_I.size(1)
        nt = y_T.size(0)
        dt = y_T.size(1)
        y_I = y_I.unsqueeze(1).expand(ni, nt, di)
        y_I = y_I.reshape(-1, di)


        y_T = y_T.unsqueeze(0).expand(ni, nt, dt)
        y_T = y_T.reshape(-1, dt)

        y = torch.cat((y_I, y_T), 1)
        # y = y_I * y_T

        relation_score = self.RelationDNN(y)
        return  relation_score