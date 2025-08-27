import numpy as np
import torch
import cv2

# Create on 15:40/10/22
class PhysicalAugmentation:
    def __init__(self, noise_std=0.01, color_shift=0.05, depth_factor=0.005, occlusion_size = 0.1):
        """
        Initialize the augmentation parameters.
        :param noise_std: Standard deviation for Gaussian noise.
        :param color_shift: Maximum shift for RGB values to simulate camera noise.
        :param light_intensity_range: Range of intensity for simulating lighting changes.
        """
        self.noise_std = noise_std
        self.color_shift = color_shift
        self.depth_factor = depth_factor
        self.occlusion_size = occlusion_size


    # def add_gaussian_noise(self, image):
    #     """
    #     Add Gaussian noise to simulate lens blur.
    #     :param image: Input RGB image as a numpy array.
    #     :return: Noisy image.
    #     """
    #     noise = np.random.normal(0, self.noise_std, image.shape).astype(np.float32)
    #     noisy_image = np.clip(image + noise, 0, 1)
    #     return noisy_image
    # def add_gaussian_noise(self, image):
    #     """ Add Gaussian noise to simulate lens blur. """
    #     device = image.device
    #     noise = torch.randn_like(image, device=device) * self.noise_std
    #     return torch.clamp(image + noise, 0, 1)

    # def perturb_color(self, image):
    #     """
    #     Apply small shifts to the color channels to simulate camera sensor noise.
    #     :param image: Input RGB image as a numpy array.
    #     :return: Color perturbed image.
    #     """
    #     shift_values = np.random.uniform(-self.color_shift, self.color_shift, (1, 1, 3)).astype(np.float32)
    #     color_shifted_image = np.clip(image + shift_values, 0, 1)
    #     return color_shifted_image
    # def perturb_color(self, image):
    #     """ Apply small shifts to the color channels to simulate camera sensor noise. """
    #     device = image.device
    #     shift_values = (torch.rand(1, 1, 3, device=device) - 0.5) * 2 * self.color_shift  # [-color_shift, color_shift]
    #     return torch.clamp(image + shift_values, 0, 1)

    # def simulate_lighting(self, image):
    #     """
    #     Simulate lighting effects by altering pixel brightness locally.
    #     :param image: Input RGB image as a numpy array.
    #     :return: Image with simulated lighting effects.
    #     """
    #     # Randomly adjust light intensity in a local region
    #     height, width, _ = image.shape
    #     x1, y1 = np.random.randint(0, width // 2), np.random.randint(0, height // 2)
    #     x2, y2 = np.random.randint(width // 2, width), np.random.randint(height // 2, height)

    #     # Random light intensity
    #     light_intensity = np.random.uniform(self.light_intensity_range[0], self.light_intensity_range[1])

    #     # Create mask for the region and apply light intensity
    #     mask = np.zeros_like(image, dtype=np.float32)
    #     mask[y1:y2, x1:x2] = 1.0

    #     lighting_image = np.clip(image * (1 - mask + mask * light_intensity), 0, 1)
    #     return lighting_image

    # def simulate_lighting(self, image):
    #     """ Simulate lighting effects by altering pixel brightness locally. """
    #     device = image.device
    #     height, width, _ = image.shape

    #     x1, y1 = torch.randint(0, width // 2, (1,), device=device), torch.randint(0, height // 2, (1,), device=device)
    #     x2, y2 = torch.randint(width // 2, width, (1,), device=device), torch.randint(height // 2, height, (1,), device=device)

    #     light_intensity = torch.empty(1, device=device).uniform_(*self.light_intensity_range)
        
    #     mask = torch.zeros_like(image, device=device)
    #     mask[y1:y2, x1:x2, :] = 1.0  

    #     return torch.clamp(image * (1 - mask + mask * light_intensity), 0, 1)

    # def add_occlusion(self, image):
    #     """ Apply a random occlusion in the center area of the image. """
    #     height, width, _ = image.shape

    #     # Calculate the occlusion size
    #     occlusion_h = int(height * self.occlusion_size)
    #     occlusion_w = int(width * self.occlusion_size)

    #     # Random choise occlusion center
    #     center_x = np.random.randint(width // 3, 2 * width // 3)
    #     center_y = np.random.randint(height // 3, 2 * height // 3)

    #     # Calculate the occlusion border
    #     x1 = max(0, center_x - occlusion_w // 2)
    #     y1 = max(0, center_y - occlusion_h // 2)
    #     x2 = min(width, center_x + occlusion_w // 2)
    #     y2 = min(height, center_y + occlusion_h // 2)

    #     # Choose the occlusion color
    #     occlusion_type = np.random.choice(["black", "gray", "noise"])
    #     if occlusion_type == "black":
    #         image[y1:y2, x1:x2] = 0
    #     elif occlusion_type == "gray":
    #         image[y1:y2, x1:x2] = 0.5
    #     elif occlusion_type == "noise":
    #         noise = np.random.uniform(0, 1, (y2 - y1, x2 - x1, 3)).astype(np.float32)
    #         image[y1:y2, x1:x2] = noise

    #     return image
    def add_occlusion(self, image):
        """ Apply a random occlusion in the center area of the image. """
        device = image.device
        height, width, _ = image.shape

        # Calculate the occlusion size
        occlusion_h = int(height * self.occlusion_size)
        occlusion_w = int(width * self.occlusion_size)

        # Random choise occlusion center
        center_x = torch.randint(width // 3, 2 * width // 3, (1,), device=device).item()
        center_y = torch.randint(height // 3, 2 * height // 3, (1,), device=device).item()

        # Calculate the occlusion border
        x1 = max(0, center_x - occlusion_w // 2)
        y1 = max(0, center_y - occlusion_h // 2)
        x2 = min(width, center_x + occlusion_w // 2)
        y2 = min(height, center_y + occlusion_h // 2)

        # Choose the occlusion color
        occlusion_type = torch.randint(0, 3, (1,), device=device).item()
        if occlusion_type == 0:  
            image[y1:y2, x1:x2, :] = 0
        elif occlusion_type == 1:  
            image[y1:y2, x1:x2, :] = 0.5
        elif occlusion_type == 2:  
            noise = torch.rand((y2 - y1, x2 - x1, 3), device=device)
            image[y1:y2, x1:x2, :] = noise

        return image


    # def augment(self, image):
    #     """
    #     Apply all augmentations (Gaussian noise, color perturbation, lighting simulation, occlusion).
    #     :param image: Input RGB image as a numpy array.
    #     :return: Augmented RGB image.
    #     """
    #     image = image.clone()
    #     image = self.add_gaussian_noise(image)
    #     image = self.perturb_color(image)
    #     image = self.simulate_lighting(image)
    #     image = self.add_occlusion(image)
    #     return image
    def augment(self, image, depth_map=None):
        """
        Apply physical augmentation including imaging degradation, photometric variation,
        shadow projection, and occlusion.
        """
        device = image.device
        image = image.clone()
        H, W, C = image.shape

        if depth_map is None:
            # print(image.shape)
            depth_map = torch.full((H, W), fill_value=3.0, device=device)
            depth_map = depth_map.unsqueeze(-1)

        # Imaging Degradation (Gaussian Noise with Depth Dependency)
        noise_std = self.noise_std + self.depth_factor * depth_map  # σ(d) = σ0 + k * d
        noise = torch.randn_like(image, device=device) * noise_std
        image = torch.clamp(image + noise, 0, 1)

        # Photometric Variation (Affine Color Shift)
        alpha = torch.rand(1, 1, 3, device=device) * 0.2 + 0.9  # α_c ∼ U(0.9, 1.1)
        beta = (torch.rand(1, 1, 3, device=device) - 0.5) * 2 * self.color_shift  # β_c ∼ U(-0.05, 0.05)
        image = torch.clamp(image * alpha + beta, 0, 1)

        # Shadow Projection (Soft Shadowing)
        gamma = 10  # Controls shadow sharpness
        shadow_mask = 1 / (1 + torch.exp(-gamma * (depth_map - depth_map.median())))
        image = image * shadow_mask  # Apply soft shadowing

        # print(image.shape)
        # Partial Occlusion (Random Rectangular Mask)
        image = self.add_occlusion(image)

        return image


# Example usage:
if __name__ == "__main__":
    # Load an example image
    # image_path = 'input_image.png'  # Replace with your image path
    image_path = '345.png'
    image = cv2.imread(image_path)  
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    # print(image.shape)

    image_tensor = torch.tensor(image, dtype=torch.float32, device="cuda")
    # Apply augmentations
    augmentor = PhysicalAugmentation()
    augmented_tensor = augmentor.augment(image_tensor)

    # Convert back to uint8 and save the result
    augmented_image = augmented_tensor.cpu().numpy()
    augmented_image = (augmented_image * 255).astype(np.uint8)
    augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite("augmented_image.png", augmented_image)