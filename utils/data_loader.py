# data loader
from __future__ import print_function, division
import glob
import torch
from skimage import io, transform, color, img_as_bool
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageDraw

#==========================dataset load==========================
class RescaleT(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size
		
	def __repr__(self):
        	return self.__class__.__name__ + '(output_size={0})'.format(self.output_size)

	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'],sample['label']
		h, w = image.shape[:2]

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		# img = transform.resize(image,(new_h,new_w),mode='constant')
		# lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

		img = transform.resize(image,(self.output_size,self.output_size),mode='constant')
		lbl = transform.resize(label,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True)
		if lbl.max() <= 1:
			lbl=lbl*255

		return {'imidx':imidx, 'image':img,'label':lbl}


class Rescale(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size
		
	def __repr__(self):
        	return self.__class__.__name__ + '(output_size={0})'.format(self.output_size)		
	
	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'],sample['label']

		if random.random() >= 0.5:
			image = image[::-1]
			label = label[::-1]

		h, w = image.shape[:2]

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		img = transform.resize(image,(new_h,new_w),mode='constant')
		lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

		return {'imidx':imidx, 'image':img,'label':lbl}


class RandomCrop(object):

	def __init__(self, appliance):
		assert isinstance(appliance, (str))
		self.appliance = appliance
        
	def __repr__(self):
        	return self.__class__.__name__ + '(appliance={0})'.format(self.appliance)
	
	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'], sample['label']
		rate_of_appliance = 1
		if self.appliance != "always":
			rate_of_appliance = random.random()
		if rate_of_appliance >= 0.5:
			output_size=(int(image.shape[0]/2),int(image.shape[1]/2))
			h, w = image.shape[:2]
			new_h, new_w = output_size

			top = np.random.randint(0, h - new_h)
			left = np.random.randint(0, w - new_w)

			img = image[top: top + new_h, left: left + new_w]
			lbl = label[top: top + new_h, left: left + new_w]
            
			image = transform.resize(img,(image.shape[0],image.shape[1]),mode='constant')
			lbl = img_as_bool(lbl)
			label = transform.resize(lbl,(label.shape[0],label.shape[1]),mode='constant', order=0, preserve_range=True)*255
		if image.max() > 1:
			image=image/255
		if label.max() <= 1:
			label=label*255
        
		return {'imidx':imidx,'image':image, 'label':label}
		

class Rotate(object):

	def __init__(self, appliance, degrees):
		assert isinstance(appliance, (str))
		self.appliance = appliance
		assert isinstance(degrees, (int, tuple))
		if isinstance(degrees, int):
			self.degrees = (-degrees, degrees)
		else:
			assert len(degrees) == 2
			self.degrees = degrees
        
	def __repr__(self):
        	return self.__class__.__name__ + '(appliance={0}, degrees={1})'.format(self.appliance, self.degrees)
	
	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'], sample['label']
		rate_of_appliance = 1
		if self.appliance != "always":
			rate_of_appliance = random.random()
		if rate_of_appliance >= 0.5:
			angle = random.uniform(self.degrees[0],self.degrees[1])
			image = transform.rotate(image, angle)
			label = img_as_bool(label)
			label = transform.rotate(label, angle, preserve_range=True)*255
		if image.max() > 1:
			image=image/255
		if label.max() <= 1:
			label=label*255

		return {'imidx':imidx,'image':image, 'label':label}


class VerticalFlip(object):

	def __init__(self, appliance):
		assert isinstance(appliance, (str))
		self.appliance = appliance
		
	def __repr__(self):
        	return self.__class__.__name__ + '(appliance={0})'.format(self.appliance)
	
	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'], sample['label']
		rate_of_appliance = 1
		if self.appliance != "always":
			rate_of_appliance = random.random()
		if rate_of_appliance >= 0.5:
			image = np.flipud(image)
			label = np.flipud(label)
		if image.max() > 1:
			image=image/255
		if label.max() <= 1:
			label=label*255
		return {'imidx':imidx,'image':image, 'label':label}


class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):

		imidx, image, label = sample['imidx'], sample['image'], sample['label']

		tmpImg = np.zeros((image.shape[0],image.shape[1],3))
		tmpLbl = np.zeros(label.shape)

		image = image/np.max(image)
		if(np.max(label)<1e-6):
			label = label
		else:
			label = label/np.max(label)

		if image.shape[2]==1:
			tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
		else:
			tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
			tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

		tmpLbl[:,:,0] = label[:,:,0]

		# change the r,g,b to b,r,g from [0,255] to [0,1]
		#transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
		tmpImg = tmpImg.transpose((2, 0, 1))
		tmpLbl = label.transpose((2, 0, 1))

		return {'imidx':torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}


class ToTensorLab(object):
	"""Convert ndarrays in sample to Tensors."""
	def __init__(self,flag=0):
		self.flag = flag

	def __call__(self, sample):

		imidx, image, label =sample['imidx'], sample['image'], sample['label']

		tmpLbl = np.zeros(label.shape)

		if(np.max(label)<1e-6):
			label = label
		else:
			label = label/np.max(label)

		# change the color space
		if self.flag == 2: # with rgb and Lab colors
			tmpImg = np.zeros((image.shape[0],image.shape[1],6))
			tmpImgt = np.zeros((image.shape[0],image.shape[1],3))
			if image.shape[2]==1:
				tmpImgt[:,:,0] = image[:,:,0]
				tmpImgt[:,:,1] = image[:,:,0]
				tmpImgt[:,:,2] = image[:,:,0]
			else:
				tmpImgt = image
			tmpImgtl = color.rgb2lab(tmpImgt)

			# nomalize image to range [0,1]
			tmpImg[:,:,0] = (tmpImgt[:,:,0]-np.min(tmpImgt[:,:,0]))/(np.max(tmpImgt[:,:,0])-np.min(tmpImgt[:,:,0]))
			tmpImg[:,:,1] = (tmpImgt[:,:,1]-np.min(tmpImgt[:,:,1]))/(np.max(tmpImgt[:,:,1])-np.min(tmpImgt[:,:,1]))
			tmpImg[:,:,2] = (tmpImgt[:,:,2]-np.min(tmpImgt[:,:,2]))/(np.max(tmpImgt[:,:,2])-np.min(tmpImgt[:,:,2]))
			tmpImg[:,:,3] = (tmpImgtl[:,:,0]-np.min(tmpImgtl[:,:,0]))/(np.max(tmpImgtl[:,:,0])-np.min(tmpImgtl[:,:,0]))
			tmpImg[:,:,4] = (tmpImgtl[:,:,1]-np.min(tmpImgtl[:,:,1]))/(np.max(tmpImgtl[:,:,1])-np.min(tmpImgtl[:,:,1]))
			tmpImg[:,:,5] = (tmpImgtl[:,:,2]-np.min(tmpImgtl[:,:,2]))/(np.max(tmpImgtl[:,:,2])-np.min(tmpImgtl[:,:,2]))

			# tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
			tmpImg[:,:,3] = (tmpImg[:,:,3]-np.mean(tmpImg[:,:,3]))/np.std(tmpImg[:,:,3])
			tmpImg[:,:,4] = (tmpImg[:,:,4]-np.mean(tmpImg[:,:,4]))/np.std(tmpImg[:,:,4])
			tmpImg[:,:,5] = (tmpImg[:,:,5]-np.mean(tmpImg[:,:,5]))/np.std(tmpImg[:,:,5])

		elif self.flag == 1: #with Lab color
			tmpImg = np.zeros((image.shape[0],image.shape[1],3))

			if image.shape[2]==1:
				tmpImg[:,:,0] = image[:,:,0]
				tmpImg[:,:,1] = image[:,:,0]
				tmpImg[:,:,2] = image[:,:,0]
			else:
				tmpImg = image

			tmpImg = color.rgb2lab(tmpImg)

			# tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.min(tmpImg[:,:,0]))/(np.max(tmpImg[:,:,0])-np.min(tmpImg[:,:,0]))
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.min(tmpImg[:,:,1]))/(np.max(tmpImg[:,:,1])-np.min(tmpImg[:,:,1]))
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.min(tmpImg[:,:,2]))/(np.max(tmpImg[:,:,2])-np.min(tmpImg[:,:,2]))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])

		else: # with rgb color
			tmpImg = np.zeros((image.shape[0],image.shape[1],3))
			image = image/np.max(image)
			if image.shape[2]==1:
				tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
			else:
				tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
				tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

		tmpLbl[:,:,0] = label[:,:,0]

		# change the r,g,b to b,r,g from [0,255] to [0,1]
		#transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
		tmpImg = tmpImg.transpose((2, 0, 1))
		tmpLbl = label.transpose((2, 0, 1))

		return {'imidx':torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}


class ChangeBackground(object):
	def __init__(self,appliance):
		assert isinstance(appliance, (str))
		self.appliance = appliance
		
	def __repr__(self):
        	return self.__class__.__name__ + '(appliance={0})'.format(self.appliance)
	
	def __call__(self, sample):
		imidx, image, label = sample['imidx'], sample['image'],sample['label']
		rate_of_appliance = 1
		if self.appliance != "always":
			rate_of_appliance = random.random()
		if rate_of_appliance >= 0.5:
			new_img = self.generate_random_gradient(image.shape[1], image.shape[0])
			thrs = label.max()/2
			idx = (label>=thrs).all(axis=2)
			new_img_c = new_img.copy()
			if image.max() > 1:
				new_img_c[idx] = image[idx]/255
			else:
				new_img_c[idx] = image[idx]
			image=new_img_c
		if image.max() > 1:
			image=image/255
		if label.max() <= 1:
			label=label*255
		
		return {'imidx':imidx,'image':image, 'label':label}
		
	def generate_random_gradient(self, image_width, image_height):
		img = Image.new(mode='RGB', size=(image_width, image_height))
		draw = ImageDraw.Draw(img)

		r2,g2,b2 = random.choice([(random.randint(0,255), random.randint(0,255), random.randint(0,255)),(random.randint(235,255), random.randint(235,255), random.randint(235,255)), (247,243,223),(238,242,239),(234, 234, 234),(180, 170, 168),(74, 74, 70),(255,255,255)])
		r,g,b = r2+(255-r2)*3/4,g2+(255-g2)*3/4,b2+(255-b2)*3/4
		
		dr = (r2 - r)/image_height
		dg = (g2 - g)/image_height
		db = (b2 - b)/image_height  
		#grey (247,243, 223)
		#grey fotostudio rgb(238, 242, 239)
		#grey compare site rgb(234, 234, 234)
		#cream rgb(180, 170, 168)
		#black compare site rgb(74, 74, 70)
		#white
		
		for i in range(max(image_height,image_width)):
			r,g,b = r+dr, g+dg, b+db
			draw.line((0,i,max(image_height,image_width),i), fill=(int(r),int(g),int(b)))
		return np.asarray(img)/255


class CombineImages(object):

	def __init__(self, appliance, img_name_list, label_name_list):
		assert isinstance(appliance, (str))
		self.appliance = appliance
		assert isinstance(img_name_list, (list))
		assert isinstance(label_name_list, (list))
		self.img_name_list = img_name_list
		self.label_name_list=label_name_list
        
	def __repr__(self):
        	return self.__class__.__name__ + '(appliance={0})'.format(self.appliance)
	
	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'], sample['label']
		rate_of_appliance = 1
		if self.appliance != "always":
			rate_of_appliance = random.random()
		if rate_of_appliance >= 0.5:
			if imidx+1 < len(self.img_name_list):
				image2 = io.imread(self.img_name_list[imidx[0]+1])

				if(0==len(self.label_name_list)):
					label_3 = np.zeros(image.shape)
				else:
					label_3 = io.imread(self.label_name_list[imidx[0]+1])
			else:
				image2 = io.imread(self.img_name_list[imidx[0]-1])

				if(0==len(self.label_name_list)):
					label_3 = np.zeros(image.shape)
				else:
					label_3 = io.imread(self.label_name_list[imidx[0]-1])
			label2 = np.zeros(label_3.shape[0:2])
			if(3==len(label_3.shape)):
				label2 = label_3[:,:,0]
			elif(2==len(label_3.shape)):
				label2 = label_3

			if(3==len(image2.shape) and 2==len(label2.shape)):
				label2 = label2[:,:,np.newaxis]
			elif(2==len(image2.shape) and 2==len(label2.shape)):
				image2 = image2[:,:,np.newaxis]
				label2 = label2[:,:,np.newaxis]
			if image2.shape != image.shape:
				image2 = transform.resize(image2,(image.shape[0],image.shape[1]),mode='constant')
				label2 = img_as_bool(label2)
				label2 = transform.resize(label2,(label.shape[0],label.shape[1]),mode='constant', order=0, preserve_range=True)*255
			if image.max() > 1:
				image=image/255
			if label.max() <= 1:
				label=label*255
			if image2.max() > 1:
				image2 = image2/255
			if label2.max() <=1:
				label2 = label2*255
			image = np.hstack((image,image2))
			label = np.hstack((label,label2))
		if image.max() > 1:
			image=image/255
		if label.max() <= 1:
			label=label*255

		return {'imidx':imidx,'image':image, 'label':label}


class SalObjDataset(Dataset):
	def __init__(self,img_name_list,lbl_name_list,transform=None):
		# self.root_dir = root_dir
		# self.image_name_list = glob.glob(image_dir+'*.png')
		# self.label_name_list = glob.glob(label_dir+'*.png')
		self.image_name_list = img_name_list
		self.label_name_list = lbl_name_list
		self.transform = transform

	def __len__(self):
		return len(self.image_name_list)

	def __getitem__(self,idx):

		# image = Image.open(self.image_name_list[idx])#io.imread(self.image_name_list[idx])
		# label = Image.open(self.label_name_list[idx])#io.imread(self.label_name_list[idx])

		image = io.imread(self.image_name_list[idx])
		imname = self.image_name_list[idx]
		imidx = np.array([idx])

		if(0==len(self.label_name_list)):
			label_3 = np.zeros(image.shape)
		else:
			label_3 = io.imread(self.label_name_list[idx])

		label = np.zeros(label_3.shape[0:2])
		if(3==len(label_3.shape)):
			label = label_3[:,:,0]
		elif(2==len(label_3.shape)):
			label = label_3

		if(3==len(image.shape) and 2==len(label.shape)):
			label = label[:,:,np.newaxis]
		elif(2==len(image.shape) and 2==len(label.shape)):
			image = image[:,:,np.newaxis]
			label = label[:,:,np.newaxis]

		sample = {'imidx':imidx, 'image':image, 'label':label}

		if self.transform:
			sample = self.transform(sample)

		return sample
