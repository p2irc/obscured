import torch
import torch.nn as nn


def get_conv_output_size(w, padding, kernel_size, stride):
    return int(((w - kernel_size + (2 * padding)) / stride) + 1)


class MatchingModel(nn.Module):
    def __init__(self, sample_length, num_hidden_estimator, d=1):
        super(MatchingModel, self).__init__()

        self.sample_length = sample_length
        conv1_output_size = get_conv_output_size(sample_length, 9, 18, 1)

        self.extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=18, stride=1, padding=9),
            nn.ReLU(),
            nn.MaxPool1d(4, stride=4),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(int(conv1_output_size / 4) * 8, 32),
            nn.ReLU(),
            nn.Dropout(p=0.05),
            nn.Linear(32, d)
        )

        self.estimator = nn.Sequential(
            nn.Linear(d, num_hidden_estimator, bias=False),
            nn.BatchNorm1d(num_hidden_estimator),
            nn.ReLU(),
            nn.Linear(num_hidden_estimator, 1)
        )

    def forward(self, seqs_a, seqs_b, pheno_a):
        matches = (seqs_a == seqs_b).to(torch.float32).clone()
        feats = self.extractor(torch.unsqueeze(matches, dim=1))
        estimator_input = torch.cat((pheno_a[:, None], feats), dim=1)

        y = self.estimator(estimator_input)
        return torch.squeeze(y)
