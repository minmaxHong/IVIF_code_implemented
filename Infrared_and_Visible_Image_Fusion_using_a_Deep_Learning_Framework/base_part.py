import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F

class BasePart:
    def __init__(self, visible_image: np.ndarray, infrared_image: np.ndarray, device: torch.device):
        super(BasePart, self).__init__()
        self.device = device  

        self.vertical_conv = np.array([[1], [-1]], dtype=np.float32)  
        self.horizontal_conv = np.array([[1, -1]], dtype=np.float32)  
        
        self.visible_image = torch.tensor(visible_image.astype(np.float32), requires_grad=False).to(self.device)
        self.infrared_image = torch.tensor(infrared_image.astype(np.float32), requires_grad=False).to(self.device)
        
        h, w = self.visible_image.shape
        self.visible_init_I_k_b = np.random.randint(0, 255, (h, w)).astype(np.float32)
        self.infrared_init_I_k_b = np.random.randint(0, 255, (h, w)).astype(np.float32)
        
        self.visible_I_k_b = nn.Parameter(torch.tensor(self.visible_init_I_k_b, requires_grad=True).to(self.device))
        self.infrared_I_k_b = nn.Parameter(torch.tensor(self.infrared_init_I_k_b, requires_grad=True).to(self.device))

        self.I_1_b = None # I_1^b(x,y)
        self.I_2_b = None # I_2^b(x,y)

    def get_edge_images(self):
        ''' Get edge images using edge filters
        Args:
            None
            
        Returns:
            vertical_visible_edge_image: vertical edge image of visible
            horizontal_visible_edge_image: horizontal edge image of visible
            
            vertical_infrared_edge_image: vertical edge image of infrared
            horizontal_infrared_edge_image: horizontal edge image of infrared
        '''
        # visible        
        vertical_visible_edge_image = F.conv2d(self.visible_image.unsqueeze(0).unsqueeze(0), 
                                        torch.tensor(self.vertical_conv).unsqueeze(0).unsqueeze(0).to(self.device), 
        
                                        padding=(1, 0)).squeeze(0).squeeze(0).cpu().numpy()
        horizontal_visible_edge_image = F.conv2d(self.visible_image.unsqueeze(0).unsqueeze(0), 
                                          torch.tensor(self.horizontal_conv).unsqueeze(0).unsqueeze(0).to(self.device), 
                                          padding=(0, 1)).squeeze(0).squeeze(0).cpu().numpy()
        
        # infrared
        vertical_infrared_edge_image = F.conv2d(self.infrared_image.unsqueeze(0).unsqueeze(0), 
                                        torch.tensor(self.vertical_conv).unsqueeze(0).unsqueeze(0).to(self.device), 
                                        padding=(1, 0)).squeeze(0).squeeze(0).cpu().numpy()
        
        horizontal_infrared_edge_image = F.conv2d(self.visible_image.unsqueeze(0).unsqueeze(0), 
                                          torch.tensor(self.horizontal_conv).unsqueeze(0).unsqueeze(0).to(self.device), 
                                          padding=(0, 1)).squeeze(0).squeeze(0).cpu().numpy()
        
        
        
        return vertical_visible_edge_image, horizontal_visible_edge_image, vertical_infrared_edge_image, horizontal_infrared_edge_image

    def get_frobenius_norm(self, image: torch.Tensor):
        ''' Calculate the Frobenius norm 

        Return:
            torch.sum(image ** 2): Frobenius norm
        '''
        
        return torch.sum(image ** 2)

    def get_optimization_I_k_b(self, hyperparameter: int = 5, iterations_per_epoch: int = 30000):
        ''' Optimize I_k_b using Adam optimizer 
        Args:
            hyperparameter: lambda value
            iterations_per_epoch: train iterations
        
        Returns:
            self.I_1_b: get optimized visible image in order to extract I_k_d_1 
            self.I_2_b: optimized infrared image in order to I_k_d_2
        '''

        visible_optimizer = torch.optim.Adam([self.visible_I_k_b], lr=0.01)
        infrared_optimizer = torch.optim.Adam([self.infrared_I_k_b], lr=0.01)
        
        # visible
        for i in range(iterations_per_epoch):
            visible_optimizer.zero_grad()

            data_term = self.get_frobenius_norm(self.visible_image - self.visible_I_k_b)  # ||I_k - I_k_b||_F^2
            conv_g_x = F.conv2d(self.visible_I_k_b.unsqueeze(0).unsqueeze(0), 
                                torch.tensor(self.vertical_conv).unsqueeze(0).unsqueeze(0).to(self.device), 
                                padding=(1, 0))  
            
            conv_g_y = F.conv2d(self.visible_I_k_b.unsqueeze(0).unsqueeze(0), 
                                torch.tensor(self.horizontal_conv).unsqueeze(0).unsqueeze(0).to(self.device), 
                                padding=(0, 1))  

            regularization_term = self.get_frobenius_norm(conv_g_x) + self.get_frobenius_norm(conv_g_y)
            loss = data_term + hyperparameter * regularization_term

            loss.backward()
            visible_optimizer.step()

            print(f"Visible Iteration [{i+1}], Loss: {loss.item():.4f}")

        # infrared
        for i in range(iterations_per_epoch):
            infrared_optimizer.zero_grad()

            data_term = self.get_frobenius_norm(self.infrared_image - self.infrared_I_k_b)  # ||I_k - I_k_b||_F^2
            conv_g_x = F.conv2d(self.infrared_I_k_b.unsqueeze(0).unsqueeze(0), 
                                torch.tensor(self.vertical_conv).unsqueeze(0).unsqueeze(0).to(self.device), 
                                padding=(1, 0))  
            
            conv_g_y = F.conv2d(self.infrared_I_k_b.unsqueeze(0).unsqueeze(0), 
                                torch.tensor(self.horizontal_conv).unsqueeze(0).unsqueeze(0).to(self.device), 
                                padding=(0, 1))  

            regularization_term = self.get_frobenius_norm(conv_g_x) + self.get_frobenius_norm(conv_g_y)
            loss = data_term + hyperparameter * regularization_term

            loss.backward()
            infrared_optimizer.step()

            print(f"Infrared Iteration [{i+1}], Loss: {loss.item():.4f}")
        
        self.I_1_b = self.visible_I_k_b.detach().cpu().numpy()
        self.I_2_b = self.infrared_I_k_b.detach().cpu().numpy()
        
        return self.I_1_b, self.I_2_b

    def get_fusion_base_parts(self, hyperparameter: int=0.5):
        '''F_b(x,y) = a_1I_1^b(x,y) + a_2I_2^b(x,y)
        
        Arg:
            hyperparameter: 0.5
        
        Return:
            F_b: Fusion base part
        '''
        F_b = hyperparameter * self.I_1_b + hyperparameter * self.I_2_b
        
        return F_b