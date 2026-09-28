from models import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt

def main():
    g = Generator().to(device)
    g.load_state_dict(torch.load("checkpoints/g_25"))
    generated_images = []
    g.eval()
    with torch.no_grad():
        fake_noises = torch.randn(10, 100).to(device)
        labels = torch.arange(0, 10).to(device)
        generated_images = g(fake_noises, labels)
    fig, axes = plt.subplots(1, 10, figsize=(10 * 2, 2))
    for i, img in enumerate(generated_images):
        axes[i].imshow(img.cpu().squeeze(), cmap="gray")
        axes[i].set_title(f"{i}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()