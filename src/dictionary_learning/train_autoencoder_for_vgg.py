import torch
import torch.nn as nn
import h5py
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

# the SparseAutoencoder is based on this: https://colab.research.google.com/drive/1u8larhpxy8w4mMsJiSBddNOzFGj7_RTn?usp=sharing
# that colab was done by the monosemanticity paper, so we could use that for feature interpretation later on as well
#original author: Levente Foldesi
#adpated by Francesco to train on VGG activations
#date: 17.12.2023

cfg = {
    "seed": 49,
    "buffer_mult": 384,
    "l1_coeff": 3e-4,
    "dict_mult": 3, #expansion factor of the latent representation
    "d_activation": 4096,
    "enc_dtype":"fp32",
}

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
class SparseAutoencoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_hidden = cfg["d_activation"] * cfg["dict_mult"]
        d_mlp = cfg["d_activation"]
        l1_coeff = cfg["l1_coeff"]
        dtype = DTYPES[cfg["enc_dtype"]]
        torch.manual_seed(cfg["seed"])
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_mlp, d_hidden, dtype=dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_hidden, d_mlp, dtype=dtype)))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_mlp, dtype=dtype))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff

        #self.to("cuda") # uncomment if you want to use GPU

    def forward(self, x):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
        l1_loss = self.l1_coeff * (acts.float().abs().sum())
        loss = l2_loss + l1_loss
        return loss, x_reconstruct, acts, l2_loss, l1_loss

    @torch.no_grad()
    def remove_parallel_component_of_grads(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj

def main():
    f = h5py.File("/home/fmassari/scaling_mlps/acts_VGG13_bn_cifar10_test.h5", "r")
    for key in f.keys():
        print(key) 
        print(type(f[key])) # get the object type   
    group = f[key]
    acitvations = []
    for i in range(len(group)):
        acitvations.append(group[i])

    sparse_autoencoder = SparseAutoencoder(cfg)
    #sparse_autoencoder.load_state_dict(torch.load('SAE_100_epochs_CIFAR10_test_vgg_bn13.pt'))

    activations_tensor = torch.tensor(acitvations, dtype=torch.float32)

    # Reshape activations to be a 2D tensor
    activations_tensor = activations_tensor.view(-1, cfg["d_activation"])
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
    epochs = 100

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_data in dataloader:
            optimizer.zero_grad()

            loss, _, _, _, _ = sparse_autoencoder(batch_data[0])      

            # Backward pass and optimize
            loss.backward()
            sparse_autoencoder.remove_parallel_component_of_grads() # not 100% sure what this does
            optimizer.step()

            total_loss += loss.item()
            

        # Learning rate scheduling step
        scheduler.step()

        average_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {average_loss:.4f}, Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
    
    torch.save(sparse_autoencoder.state_dict(), 'SAE_100_epochs_CIFAR10_test_vgg_bn13.pt')
    # evaulate the model
    with torch.no_grad():
        loss, decoded, encoded, _, _ = sparse_autoencoder(activations_tensor)
        mse = F.mse_loss(activations_tensor, decoded, reduction='mean')
        print(f'Mean Squared Error: {mse.item()}')

    # trying to do some feature interpretation, no idea if that in the right direction    
    feature_densities = (encoded != 0).float().mean(dim=0)
    plt.hist(feature_densities.numpy(), bins=50, log=True)
    plt.xlabel('Feature Density')
    plt.ylabel('Frequency')
    plt.title('Histogram of Feature Densities')
    plt.show()


if __name__ == "__main__":
    main()