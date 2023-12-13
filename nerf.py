import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime


class NerfModel(nn.Module):
    def __init__(self, embedding_dim_pos=10, embedding_dim_direction=4, hidden_dim=128):
        super(NerfModel, self).__init__()

        self.block1 = nn.Sequential(
            nn.Linear(embedding_dim_pos * 6 + 3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Linear(embedding_dim_pos * 6 + hidden_dim + 3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim + 1),
        )

        self.block3 = nn.Sequential(
            nn.Linear(embedding_dim_direction * 6 + hidden_dim + 3, hidden_dim // 2), nn.ReLU(),
        )

        self.block4 = nn.Sequential(
            nn.Linear(hidden_dim // 2, 3), nn.Sigmoid()
        )

        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction
        self.relu = nn.ReLU()

    @staticmethod
    def positional_encoding(x, L):
        '''
        Original NeRF positional encoding, creates extra values from x at different frequencies

        Args:
            x (torch.Tensor): input tensor
            L (int): number of frequencies to create

        Returns:
            tensor of shape (x.shape[0], 3 * L + 3)
        '''
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)
        
    def forward(self, o, d):
        emb_x = self.positional_encoding(o, self.embedding_dim_pos)
        emb_d = self.positional_encoding(d, self.embedding_dim_direction)
        h = self.block1(emb_x)
        tmp = self.block2(torch.cat([emb_x, h], dim=1))
        h, sigma = tmp[:, :-1], self.relu(tmp[:, -1])
        h = self.block3(torch.cat([emb_d, h], dim=1))
        c = self.block4(h)
        return c, sigma


class PiecewiseConstantPDF:
    def __init__(self, bin_boundaries, probabilities):
        """
        Args:
            bin_boundaries (torch.Tensor): boundaries of the bins
            probabilities (torch.Tensor): probabilities a sample falls into a bin
        """
        self.bin_boundaries = bin_boundaries
        self.probabilities = probabilities
        self.lower_bounds = bin_boundaries[:, :-1]
        self.upper_bounds = bin_boundaries[:, 1:]
        # Do not sample from the last bin because it is unbounded
        self.probabilities[:, -1] = 0
        # Normalize probabilities to sum to 1
        epsilon = 1e-6 # Small added value to avoid division by zero
        self.probabilities = self.probabilities / self.probabilities.sum(dim=1, keepdim=True) + epsilon
        self.cdf = torch.cumsum(self.probabilities, dim=1)

    def sample(self, num_samples=128):
        """
        Randomly sample from the Piecewise Constant probability distribution using the inverse CDF method

        Args:
            num_samples (int): number of samples to randomly draw

        Returns:
            samples (torch.Tensor): samples from the distribution
        """

        # Generate Uniform random samples
        uniform_samples = torch.rand((self.probabilities.shape[0], num_samples), device=self.probabilities.device).unsqueeze(2)
        # Use inverse transform sampling to map Uniform samples to Piecewise Constant samples
        sample_bin_indices = torch.searchsorted(self.cdf.squeeze(2), uniform_samples.squeeze(2)).unsqueeze(2)
        # The searchsorted function may return an index that is out of bounds, so clamp it
        sample_bin_indices = torch.clamp(sample_bin_indices, 0, self.cdf.shape[1] - 1)
        # Randomly sample values from within corresponding bins using linear interpolation
        random_interpolation = torch.rand((self.probabilities.shape[0], num_samples), device=self.probabilities.device)

        # Select the lower and upper bounds of the bins corresponding to the sampled indices
        sample_lower_bounds = self.lower_bounds.gather(1, sample_bin_indices.squeeze(2))
        sample_upper_bounds = self.upper_bounds.gather(1, sample_bin_indices.squeeze(2))
        # Randomly interpolate a new t sample for each sampled bin
        samples = torch.lerp(sample_lower_bounds, sample_upper_bounds, random_interpolation)

        return samples
    

def compute_accumulated_transmittance(alphas):
    """
    Compute the accumulated transmittance from the alpha values.
    Transmittance is the probability the ray travels from the near plane to the current sample without being occluded.

    Args:
        alphas (torch.Tensor): alpha values

    Returns:
        (torch.Tensor): accumulated transmittance values
    """
    accumulated_transmittance = torch.cumprod(alphas, dim=1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=1)


def generate_binned_t_values(num_rays, num_bins, upper_bound, lower_bound):
    """
    Generate t values for sampling the NeRF rays by taking one random sample from bins of equal distance

    Args:
        num_rays (int): number of rays
        num_bins (int): number of bins to sample from
        upper_bound (float): upper bound of the t values
        lower_bound (float): lower bound of the t values

    Returns:
        (torch.tensor) t_values: t values for ray sampling
    """

    num_boundaries = num_bins + 1
    bin_boundaries = torch.linspace(lower_bound, upper_bound, num_boundaries, device=device).expand(num_rays, num_boundaries)
    random_selections = torch.rand(num_rays, num_bins, device=device)
    t_values = torch.lerp(bin_boundaries[:, :-1], bin_boundaries[:, 1:], random_selections)
    return t_values


def render_coarse(coarse_nerf_model, ray_origins, ray_directions, t_near=0, t_far=0.5, num_bins=64):
    """
    Render rays colours using the coarse NeRF model.
    Also return sample distances and weights for fine network stratified samples.

    Args:
        nerf_model (NerfModel): Coarse NeRF model
        ray_origins (torch.Tensor): ray origins
        ray_directions (torch.Tensor): ray directions
        t_near (float): near plane
        t_far (float): far plane
        num_bins (int): number of bins to use for the Piecewise Constant PDF

    Returns:
        (torch.Tensor): Ray colours
        (torch.Tensor): Ray sample distances
        (torch.Tensor): Ray sample weights
    """

    device = ray_origins.device

    t_coarse = generate_binned_t_values(ray_origins.shape[0], num_bins, t_far, t_near)

    # Distances between the samples t, concatenated with a large value to represent the ray extending to infinity
    deltas_coarse = t_coarse[:, 1:] - t_coarse[:, :-1]
    deltas_coarse = torch.cat((deltas_coarse, torch.full((deltas_coarse.shape[0], 1), t_far, device=device)), dim=1)

    # x_coarse is shape (num_rays, num_samples, 3)
    # x_coarse[i, j] is the point along the ray ray_origins[i] + t_coarse[i, j] * ray_directions[i]
    x_coarse = ray_origins.clone()
    x_coarse = x_coarse.unsqueeze(1) + (t_coarse.unsqueeze(2) * ray_directions.unsqueeze(1))

    # Sample the coarse NeRF model for colours and sigmas (densities)
    ray_directions_coarse = ray_directions.clone()
    # Expand ray_directions
    ray_directions_coarse = ray_directions_coarse.expand(num_bins, ray_directions_coarse.shape[0], 3)
    colours_coarse, sigmas_coarse = coarse_nerf_model(x_coarse.reshape(-1, 3), ray_directions_coarse.reshape(-1, 3))
    colours_coarse = colours_coarse.reshape(x_coarse.shape)
    sigmas_coarse = sigmas_coarse.reshape(x_coarse.shape[:-1])

    # Compute the alpha values
    alphas_coarse = 1 - torch.exp(-sigmas_coarse * deltas_coarse)

    # Compute the weighting of each sample based on the alpha values and transmittance
    # weights_coarse[i, j] is the weight of the jth sample of the ith ray
    weights_coarse = compute_accumulated_transmittance(1 - alphas_coarse).unsqueeze(2) * alphas_coarse.unsqueeze(2)
    weights_coarse_sum = weights_coarse.sum(1).sum(1)
    # Compute the pixel colour from the weighted sum of sample colours
    c_coarse = torch.sum(weights_coarse * colours_coarse, dim=1)
    # Adjust final colour for remaining transparency
    c_coarse_adjusted = c_coarse + 1 - weights_coarse_sum.unsqueeze(1)
    
    return c_coarse_adjusted, t_coarse, weights_coarse


def render_fine(fine_nerf_model, ray_origins, ray_directions, t_coarse, weights_coarse, num_samples_fine=128, t_near=0, t_far=0.5):
    """
    Render rays colours using the fine NeRF model.
    If in training mode, use the coarse samples and generate new stratified samples.
    If in evaluation mode, generate all samples for the fine network.

    Args:
        nerf_model (NerfModel): fine NeRF model
        ray_origins (torch.Tensor): ray origins
        ray_directions (torch.Tensor): ray directions
        t_coarse (torch.Tensor): coarse sample distances
        weights_coarse (torch.Tensor): coarse sample weights        
        fine_samples (int): number of fine samples to use
        t_near (float): near plane for ray sampling
        t_far (float): far plane for ray sampling


    Returns:
        c_fine_adjusted (torch.Tensor): Ray colours adjusted for transparency
    """

    piecewise_constant_pdf = PiecewiseConstantPDF(bin_boundaries = torch.linspace(t_near, t_far, weights_coarse.shape[1] + 1, device=ray_origins.device).expand(weights_coarse.shape[0], weights_coarse.shape[1] + 1),
                                                    probabilities = weights_coarse)
    additional_samples = piecewise_constant_pdf.sample(num_samples_fine)
    t_fine = torch.cat((t_coarse, additional_samples), dim=1)
    t_fine, _ = torch.sort(t_fine, dim=1)

    # Distances between the samples t, concatenated with a large value to represent the ray extending to infinity
    deltas_fine = t_fine[:, 1:] - t_fine[:, :-1]
    deltas_fine = torch.cat((deltas_fine, torch.full([deltas_fine.shape[0], 1], t_far, device=ray_origins.device)), dim=1)

    # x_fine is shape (num_rays, num_samples, 3)
    # x_fine[i, j] is the point along the ray ray_origins[i] + t_fine[i, j] * ray_directions[i]
    x_fine = ray_origins.clone()
    x_fine = x_fine.unsqueeze(1) + (t_fine.unsqueeze(2) * ray_directions.unsqueeze(1))

    # Sample the fine NeRF model for colours and sigmas (densities)
    ray_directions_fine = ray_directions.clone()
    ray_directions_fine = ray_directions_fine.expand(num_samples_fine + t_coarse.shape[1], ray_directions.shape[0], 3)
    colours_fine, sigmas_fine = fine_nerf_model(x_fine.reshape(-1, 3), ray_directions_fine.reshape(-1, 3))
    colours_fine = colours_fine.reshape(x_fine.shape)
    sigmas_fine = sigmas_fine.reshape(x_fine.shape[:-1])

    # Compute the alpha values
    alphas_fine = 1 - torch.exp(-sigmas_fine * deltas_fine)

    # Compute the weighting of each sample based on the alpha values and transmittance
    # weights_coarse[i, j] is the weight of the jth sample of the ith ray
    weights_fine = compute_accumulated_transmittance(1 - alphas_fine).unsqueeze(2) * alphas_fine.unsqueeze(2)
    weights_fine_sum = weights_fine.sum(1).sum(1)
    # Compute the pixel colour from the weighted sum of sample colours
    c_fine = torch.sum(weights_fine * colours_fine, dim=1)
    # Adjust final colour for remaining transparency
    c_fine_adjusted = c_fine + 1 - weights_fine_sum.unsqueeze(1)

    return c_fine_adjusted


def train(model_coarse, model_fine, optimizer_coarse, optimizer_fine, scheduler_coarse, scheduler_fine, data_loader, device='cpu', t_near=0, t_far=1, num_epochs=16, num_samples_coarse=64, num_samples_fine=128, render_height=400, render_width=400):
    """
    Train both coarse and fine network models.
    Saves model checkpoints and rendered novel view images at each training epoch.
    
    Args:
        model_coarse (NerfModel): The coarse NeRF model
        model_fine (NerfModel): The fine NeRF model
        optimizer_coarse (torch.optim): The optimizer for the coarse NeRF model
        optimizer_fine (torch.optim): The optimizer for the fine NeRF model
        scheduler_coarse (torch.optim.lr_scheduler): The learning rate scheduler for the coarse NeRF model optimizer
        scheduler_fine (torch.optim.lr_scheduler): The learning rate scheduler for the fine NeRF model optimizer
        data_loader (torch.utils.data.DataLoader): The PyTorch training data loader
        device (str): The hardware device on which to run training
        t_near (int): The near plane boundary for ray sampling
        t_far (int): The far plane boundary for ray sampling
        num_epochs (int): Number of training epochs
        num_samples_coarse (int): The number of ray samples for training the coarse network
        num_samples_fine (int): The number of ray samples for training the fine network
        render_height (int): Rendered image height
        render_width (int): Rendered image width

    """
    start_time = datetime.now().strftime("%Y%m%d%H%M%S")
    training_loss_coarse=[]
    training_loss_fine=[]
    for e in tqdm(range(num_epochs)):
        for batch in data_loader:
            """ Data is shape [num_rays, data]
                One ray is of form :3 = ray origins x, y, z; 3:6 = ray directions; 6: = pixel colour values """
            ray_origins = batch[:, :3].to(device)
            ray_directions = batch[:, 3:6].to(device)
            gt_px_values = batch[:, 6:].to(device)

            pred_px_coarse, t_coarse, weights = render_coarse(model_coarse, ray_origins, ray_directions, t_near=t_near, t_far=t_far, num_bins=num_samples_coarse)
            coarse_loss = ((gt_px_values - pred_px_coarse) ** 2).sum()

            coarse_samples_copy, weights_copy, coarse_loss_copy = t_coarse.clone().detach(), weights.clone().detach(), coarse_loss.clone().detach()

            pred_px_fine = render_fine(model_fine, ray_origins, ray_directions, coarse_samples_copy, weights_copy, num_samples_fine=128, t_near=t_near, t_far=t_far)
            fine_loss = ((gt_px_values - pred_px_fine) ** 2).sum() + coarse_loss_copy

            optimizer_coarse.zero_grad()
            coarse_loss.backward()
            optimizer_coarse.step()
            training_loss_coarse.append(coarse_loss.item())
            writer_coarse.add_scalar('Training Loss Coarse', coarse_loss.item(), len(training_loss_coarse))

            optimizer_fine.zero_grad()
            fine_loss.backward()
            optimizer_fine.step()
            training_loss_fine.append(fine_loss.item())
            writer_fine.add_scalar('Training Loss Fine', fine_loss.item(), len(training_loss_fine))

        scheduler_coarse.step()
        scheduler_fine.step()


        for img_index in range(200):
            test(t_near, t_far, testing_dataset, num_samples_coarse=num_samples_coarse, num_samples_fine=num_samples_fine, img_index=img_index, render_height=render_height, render_width=render_width, epoch=e, time=start_time)
        
        torch.save(model_coarse.state_dict(), "checkpoints/coarse_network_weights/coarse_nerf_model_"+str(training_loss_coarse[-1])+".pt")
        torch.save(model_fine.state_dict(), "checkpoints/fine_network_weights/fine_nerf_model_"+str(training_loss_fine[-1])+".pt")

@torch.no_grad()
def test(t_near, t_far, dataset, num_samples_coarse=64, num_samples_fine=128, chunk_size=5, img_index=0, render_height=400, render_width=400, epoch=0, time=""):
    ray_origins = dataset[img_index * render_height * render_width: (img_index + 1) * render_height * render_width, :3]
    ray_directions = dataset[img_index * render_height * render_width: (img_index + 1) * render_height * render_width, 3:6]

    data = []
    for i in range(int(np.ceil(render_height / chunk_size))):
        ray_origins_ = ray_origins[i * render_width * chunk_size: (i + 1) * render_width * chunk_size].to(device)
        ray_directions_ = ray_directions[i * render_width * chunk_size: (i + 1) * render_width * chunk_size].to(device)
        # Get sample distances and weights from the coarse network
        _, t_coarse_, weights_coarse_ = render_coarse(coarse_nerf_model=model_coarse, ray_origins=ray_origins_, ray_directions=ray_directions_, t_near=t_near, t_far=t_far, num_bins=num_samples_coarse)
        # Render pixel colour values from the fine network only
        regenerated_px_values = render_fine(model_fine, ray_origins_, ray_directions_, t_coarse_, weights_coarse_, num_samples_fine, t_near=t_near, t_far=t_far)
        data.append(regenerated_px_values)

    img = torch.cat(data).data.cpu().numpy().reshape(render_height, render_width, 3)

    experiment_dir = f'novel_views/experiment_{time}'
    os.makedirs(experiment_dir, exist_ok=True)

    epoch_folder = f'epoch_{epoch}'
    folder_path = os.path.join(experiment_dir, epoch_folder)
    os.makedirs(folder_path, exist_ok=True)

    img_path = os.path.join(folder_path, f'img{img_index}.png')
    plt.figure()
    plt.imshow(img)
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device.')

writer_coarse = SummaryWriter('logs/logs_coarse')
writer_fine = SummaryWriter('logs/logs_fine')

training_dataset = torch.from_numpy(np.load('data/training_data.pkl', allow_pickle=True))
testing_dataset = torch.from_numpy(np.load('data/testing_data.pkl', allow_pickle=True))
                                    
model_coarse = NerfModel().to(device)
optimizer_coarse = torch.optim.Adam(model_coarse.parameters(), lr=5e-4)

model_fine = NerfModel().to(device)
optimizer_fine = torch.optim.Adam(model_fine.parameters(), lr=5e-4)

num_epochs = 16
end_lr = 5e-5
gamma = (end_lr / 5e-4) ** (1 / num_epochs)
print("Exponential Scheduler Gamma: ", gamma)

scheduler_coarse = torch.optim.lr_scheduler.ExponentialLR(optimizer_coarse, gamma=gamma)
scheduler_fine = torch.optim.lr_scheduler.ExponentialLR(optimizer_fine, gamma=gamma)

data_loader = DataLoader(training_dataset, batch_size=4096, shuffle=True)
train(model_coarse, model_fine, optimizer_coarse, optimizer_fine, scheduler_coarse, scheduler_fine, data_loader, num_epochs=16, device=device, t_near=2, t_far=6, num_samples_coarse=64, num_samples_fine=128, render_height=400, render_width=400)