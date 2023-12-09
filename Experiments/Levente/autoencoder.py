import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size=512):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.decoder = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def custom_loss(self, input_data, encoded, decoded):
        l2_loss = nn.MSELoss()(input_data, decoded)
        l1_loss = torch.mean(torch.abs(encoded))
        total_loss = l2_loss + l1_loss
        return total_loss


def main():

    input_size =  100  # Replace with actual input size
    sparse_autoencoder = SparseAutoencoder(input_size)

    activations =  10 # Replace with actual activations
    activations_tensor = torch.tensor(activations, dtype=torch.float32) 

    net = SparseAutoencoder(100)
    print("network",net)

    with torch.no_grad():
        encoded_activations, decoded_activations = sparse_autoencoder(activations_tensor)

    input_data = activations_tensor 
    loss = sparse_autoencoder.custom_loss(input_data, encoded_activations, decoded_activations)

    print("Encoded Activations:", encoded_activations)
    print("Decoded Activations:", decoded_activations)
    print("Reconstruction Loss:", loss.item())

if __name__ == "__main__":
    main()
