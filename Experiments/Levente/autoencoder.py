import torch
import torch.nn as nn
import h5py
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, l1_coeff=1e-5):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.decoder = nn.Linear(hidden_size, input_size)
        self.l1_coeff = l1_coeff

    def forward(self, x):
        pre_encoder_bias = x - self.decoder.bias
        encoded = self.encoder(pre_encoder_bias)
        decoded = self.decoder(encoded) + self.decoder.bias
        return encoded, decoded

    def custom_loss(self, input_data, encoded, decoded):
        l2_loss = F.mse_loss(input_data, decoded, reduction='mean')
        l1_loss = torch.mean(torch.abs(encoded))
        total_loss = l2_loss + self.l1_coeff * l1_loss
        return total_loss


def main():
    f = h5py.File("/Users/leventefoldesi/Downloads/acts_B_12-Wi_1024_cifar10_test.h5", "r")
    for key in f.keys():
        print(key) 
        print(type(f[key])) # get the object type: usually group or dataset   
    group = f[key]
    acitvations = []
    for i in range(len(group)):
        acitvations.append(group[i])

    input_size =  1024 
    hidden_size = input_size*3
    sparse_autoencoder = SparseAutoencoder(input_size, hidden_size=hidden_size, l1_coeff=1e-5)

    activations_tensor = torch.tensor(acitvations, dtype=torch.float32)

    # Reshape activations to be a 2D tensor
    activations_tensor = activations_tensor.view(-1, 1024)
    print("Activations Tensor Shape:", activations_tensor.shape)

    # Define a DataLoader
    batch_size = 32
    dataset = TensorDataset(activations_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define an optimizer
    optimizer = optim.Adam(sparse_autoencoder.parameters())

    # Learning rate scheduling
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Number of epochs to train for
    epochs = 10

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_data in dataloader:
            input_data = batch_data[0]
            optimizer.zero_grad()

            # Forward pass
            encoded, decoded = sparse_autoencoder(input_data)

            # Calculate loss using the custom loss function
            loss = sparse_autoencoder.custom_loss(input_data, encoded, decoded)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Learning rate scheduling step
        scheduler.step()

        average_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {average_loss:.4f}, Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')


if __name__ == "__main__":
    main()
