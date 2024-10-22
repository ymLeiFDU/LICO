import numpy as np
import torch
import os
import glob
import cv2
import pickle
import random
import skimage
import torchvision.transforms as transforms
# from torch.nn.functional import InterpolationMode

from torch.utils.data import DataLoader
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from random import shuffle
import torch.nn.functional as F
from PIL import Image

class SamCifarDataset(Dataset):
	def __init__(self, sampled_data, sampled_labels, transform):

		self.sampled_data = sampled_data
		self.sampled_labels = sampled_labels
		self.transform = transform

	def __getitem__(self, index):

		data = self.sampled_data[index % self.sampled_data.shape[0]]
		data = self.transform(Image.fromarray(np.uint8(data)))
		# data = torch.from_numpy(data).permute(2, 0, 1).type(torch.FloatTensor)
		noise = torch.randn_like(data)*0.1 + 0.
		data += noise

		label = self.sampled_labels[index % self.sampled_labels.shape[0]]
		label = torch.from_numpy(np.array(label)).type(torch.LongTensor)

		# label = F.one_hot(label, num_classes = 10)
		# return data, label.type(torch.FloatTensor)
		return data, label

	def __len__(self):

		return len(self.sampled_data)









