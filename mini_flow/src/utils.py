import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt
def show_field(field, epoch, samples=100, grid_size=20, steps=100, fps=10, max_coord=3):

    frames = []
    # prepare position data
    pos_x_t = torch.randn((100,2), device="cuda")

    # grid coord data
    x = torch.linspace(-max_coord,max_coord,grid_size)
    y = torch.linspace(-max_coord,max_coord,grid_size)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

    # compute for all timesteps
    for step in range(steps+20):
        # prepare data for field visual
        vel_x_t = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1).to("cuda")

        # timestep tensors
        time = min(1.0, step/steps)
        vel_t = torch.ones((vel_x_t.shape[0], 1), device="cuda") * time
        pos_t = torch.ones((samples, 1), device="cuda") * time

        # inference
        field.eval()
        with torch.no_grad():
            # field visual
            input = torch.cat([vel_x_t, vel_t], dim=-1)
            vels = field(input).cpu().numpy()

            # sample transformation
            input = torch.cat([pos_x_t, pos_t], dim=-1)
            sample_vel_t = field(input)

    
        # plotting frame
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.quiver(grid_x.numpy(), grid_y.numpy(), vels[:, 0], vels[:, 1], color='blue')
        ax.scatter(pos_x_t[:, 0].cpu(), pos_x_t[:, 1].cpu(), s=5, alpha=0.5, c='green')
        ax.set_title(f"Velocity Field t={time}")
        ax.set_xlim(-max_coord, max_coord)
        ax.set_ylim(-max_coord, max_coord)
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frames.append(image)
        plt.close(fig)

        # "oiler" (lol) step 
        pos_x_t = pos_x_t + (1/steps) * sample_vel_t
    
    # save
    imageio.mimsave(f"./output/flow_matching_moon{epoch}.gif", frames, fps=fps)
    