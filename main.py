import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import os
from models import *
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("./checkpoints", exist_ok=True)

batch_size = 1024
learning_rate = 0.0002
epochs = 25

def main():
    # MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root="./data", train=True, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers = 8,
        pin_memory=True
    )


    d = Discriminator().to(device)
    g = Generator().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer_d = optim.Adam(d.parameters(), lr=learning_rate/10, betas=(0.5, 0.999))
    optimizer_g = optim.Adam(g.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=25, gamma=0.60)
    scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=25, gamma=0.60)
    fixed_noise = torch.randn(1, 100).to(device)
    fixed_label = torch.tensor([3]).to(device)
    generated_images = []

    for data, target_n in train_loader:
        print(data.shape)
        print(target_n.shape)
        break
    for epoch in range(epochs):
        d.train()
        g.train()
        d_total_loss = 0
        g_total_loss = 0
        for data, target_n in tqdm(train_loader):
            data = data.to(device); target_n = target_n.to(device)
            B = data.size(0)

            optimizer_d.zero_grad()

            fake_noise = torch.randn(B, 100).to(device)
            fake_labels = torch.randint(0, 10, (B,)).to(device)
            fake_images = g(fake_noise, fake_labels).detach()
            d_fake_output = d(fake_images, fake_labels)
            d_fake_loss = criterion(d_fake_output, torch.zeros_like(d_fake_output)+0.1)

            d_real_output = d(data, target_n)
            d_real_loss = criterion(d_real_output, torch.ones_like(d_real_output)*0.9)

            d_loss = (d_real_loss + d_fake_loss)/2
            d_loss.backward()
            optimizer_d.step()
            for i in d.parameters():
                i.requires_grad = False
            for _ in range(3):

                optimizer_g.zero_grad()

                fake_noise = torch.randn(B, 100).to(device)
                fake_labels = torch.randint(0, 10, (B,)).to(device)
                fake_images = g(fake_noise, fake_labels)
                g_s_realism = d(fake_images, fake_labels)
                g_faking_loss = criterion(g_s_realism, torch.ones_like(g_s_realism)*0.9)

                g_faking_loss.backward()
                optimizer_g.step()
            for i in d.parameters():
                i.requires_grad = True
            d_total_loss += d_loss.item()
            g_total_loss += g_faking_loss.item()
        g.eval()
        with torch.no_grad():
            fake_img = g(fixed_noise, fixed_label).cpu()
            generated_images.append(fake_img)


        print(f" Epoch {epoch+1}/{epochs} G loss: {g_total_loss/len(train_loader)} D loss: {d_total_loss/len(train_loader)}")
        print(f"----G lr: {scheduler_g.get_last_lr()[0]} D lr: {scheduler_d.get_last_lr()[0]}")

        torch.save(g.state_dict(), f"checkpoints/g_{epoch+1}")
    fig, axes = plt.subplots(1, epochs, figsize=(epochs * 2, 2))

    for i, img in enumerate(generated_images):
        axes[i].imshow(img.squeeze(), cmap="gray")
        axes[i].set_title(f"E{i+1}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()