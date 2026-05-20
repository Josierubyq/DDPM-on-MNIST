# DDPM on MNIST

This repository contains the code for a minimal MNIST DDPM experiment and a conditional DDPM extension.

The project follows the homework requirement of implementing an end-to-end diffusion model on MNIST, including the forward noising process, reverse denoising trajectory, generated samples, and training loss curves.

## Contents

- `mnist_ddpm_conditional_with_unconditional_module.ipynb`  
  Main notebook containing:
  - MNIST loading and preprocessing
  - Dataset preview and sanity checks
  - Linear beta schedule
  - Forward noising process
  - Unconditional DDPM baseline
  - Conditional DDPM extension with digit labels
  - Reverse denoising sampling
  - Generated sample grids
  - Training loss visualizations

- `requirements.txt`  
  Python dependencies.

- `outputs/`  
  Saved figures and model checkpoints.

## Method

MNIST images are normalized to `[-1, 1]`. The diffusion process uses `T = 200` timesteps with a linear beta schedule.

The forward noising process is:

```text
x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) epsilon
```

where `epsilon ~ N(0, I)`.

The unconditional DDPM baseline predicts:

```text
epsilon_theta(x_t, t)
```

The conditional DDPM extension predicts:

```text
epsilon_theta(x_t, t, y)
```

where `y` is the MNIST digit label from 0 to 9.

Both models are trained with MSE noise prediction loss.

## How to Run

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Open the notebook:

```bash
jupyter notebook mnist_ddpm_conditional_with_unconditional_module.ipynb
```

Then run the notebook cells in order.

## Main Outputs

The notebook saves the following figures in `outputs/`:

- `mnist_dataset_preview.png`
- `forward_noising_grid.png`
- `reverse_denoising_trajectory.png`
- `unconditional_generated_digits_compact_notitle.png`
- `conditional_generated_digits_compact_notitle.png`
- `loss_curves_side_by_side.png`

It also saves model checkpoints:

- `unconditional_ddpm_mnist.pth`
- `conditional_ddpm_mnist.pth`

## Notes

The unconditional model corresponds to the original minimal MNIST DDPM objective. The conditional model extends it by adding label conditioning, which allows controlled generation of digits from 0 to 9.
