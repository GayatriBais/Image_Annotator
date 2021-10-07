import numpy as np
import json

import cv2
from PIL import Image
#import skimage
#from skimage.draw import disk

import matplotlib
from matplotlib import pyplot as plt

import re
import os

import ipywidgets as widgets
from ipywidgets import *
import IPython.display as Disp



#######################################################################################################################
##Functions to load image and image annotations
#######################################################################################################################
def get_Image(img_path):
    
    """
    Loads Image given file path
    Return: Image, Image Name
    """
    
    #Extract Image name from path 
    img_name = re.split(r"[/|\\]", img_path)[-1]
    #print(img_name)
    
    # Load Image
    try:
        # CV2 read image in BGR by default
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # Change to RGB for viewing with Matplot
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print(image.shape)
        # print(image)
        #plt.imshow(image)
    except:
        print("No such file exists")
        
    return image, img_name

def load_image_info(ann_path):
    
    """
    Input: Path to annotaion file that marks plate in each Image
    Output: A dictionary with key = image name and 
    value = (filename, circle x-coordinate, circle y-coordinate, radius)
    """
    
    # Load JSON file
    with open(ann_path) as jfile:
        data = json.load(jfile)
        
    # Create a dicitonary that maps Image Name to Image Meta
    img_name_2_meta = dict()

    for item in data:
        #print(item)
        
        filename = data[item]['filename']
        #print("filename: ", filename)
        cx = data[item]['regions'][0]['shape_attributes']['cx']
        #print("cx: ", cx)
        cy = data[item]['regions'][0]['shape_attributes']['cy']
        #print("cy: ", cy)
        r = data[item]['regions'][0]['shape_attributes']['r']
        #print("r: ", r)
        
        #Save Image annotation meta to dictionary
        img_name_2_meta[filename] = (filename, cx, cy, r)
    
    return img_name_2_meta



#######################################################################################################################
##Functions to load Threshold image
#######################################################################################################################
def get_thresh_image(image_name):
    img_title = image_name[:-4]
    mask_path = img_title+"_thresh_mask.png"
    mask_path = os.path.join(os.getcwd(), mask_path)
    #print(mask_path)
    
    thresh_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    return thresh_mask

#######################################################################################################################
##Functions to load Zero image
#######################################################################################################################
def get_zero_mask(image_name, image_shape):
    
    img_title = image_name[:-4]
    img_title = img_title+"_segment_mask.png"
    final_mask_path = os.path.join(os.getcwd(), img_title)
    #print(final_mask_path)


    if os.path.isfile(final_mask_path):
        zero_mask = cv2.imread(final_mask_path, cv2.IMREAD_GRAYSCALE)
    else:
        #write if condition to check if in disc
        zero_mask = np.zeros( (image_shape[0], image_shape[1]), dtype = 'uint8')

    return zero_mask

#######################################################################################################################
# Function to draw binary circle mask
#######################################################################################################################
def make_circle_mask(image, cx, cy, r):
    
    """
    Returns a binary mask of the plate/disc annotation
    """
    cx = int(cx)
    cy = int(cy)
    r = int(r)
    
    # Draw Disk
    #rr, cc = disk((cy, cx), r, shape = image.shape)
    
    circle_mask = np.zeros( (image.shape[0], image.shape[1]), dtype = 'uint8')
    #circle_mask[rr, cc] = 1

    circle_mask = cv2.circle(circle_mask, (cx, cy), r, 1, -1)
    
    return circle_mask 


#######################################################################################################################
##Function to plot images
#######################################################################################################################
def imagePlotter(color_image, binary_mask):
    
    thresh_fig, thresh_ax =  plt.subplots(1, 2, figsize=(15, 15))
    
    #First subplot
    thresh_ax[0].imshow(color_image)
    thresh_ax[0].axis('off')
    thresh_ax[0].set_title("Color Image")
    
    
    #Second subplot
    thresh_ax[1].imshow(binary_mask, cmap = 'gray')
    thresh_ax[1].axis('off')
    thresh_ax[1].set_title("Binary Mask")
    
    plt.show()


#######################################################################################################################
## Class For Manual Thresholding Operations
#######################################################################################################################
class Manual_Thresholder():
    
    def __init__(self, img_path, img_name_2_meta):
        
        # Load Image
        self.image, self.image_name = get_Image(img_path)
        # Load Image Meta
        self.image_meta = img_name_2_meta[self.image_name]
        
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # image_meta = (filename, cx, cy, r)
        self.circle_mask = make_circle_mask(self.image, self.image_meta[1], self.image_meta[2], self.image_meta[3])
        
        
        
        # Button used to Save Image
        self.save_button = widgets.Button(value=False, description='Save Image',
                                                disabled=False, button_style='success', 
                                                tooltip='Description', icon='floppy-o')
        
        #Link button to callback fuction
        self.save_button.on_click(self.button_click)
        
                                           
        # Slider used to Adjust Threshold                                      )
        self.MANUAL_THRES_SLIDER = widgets.FloatSlider(value=120.0, min=0, max=255.0, step=1.0, description='THRESH:',
                                                disabled=False, continuous_update=False, orientation='horizontal',
                                                readout=True, readout_format='.1f',
                                               )

        
        # Display Save Button
        Disp.display(self.save_button)
        
        # Run Widget Interaction
        widgets.interact(self.run_slider, thres = self.MANUAL_THRES_SLIDER)
        
        
    
    def run_slider(self, thres = 120):
        (self.thresh, self.thresh_image) = cv2.threshold(self.gray_image, thres, 255, cv2.THRESH_BINARY_INV)
        self.thresh_image *= self.circle_mask
        #Plot Changed Images
        imagePlotter(self.image, self.thresh_image)
        
    def button_click(self, _):
        #Create path for thresholded mask
        mask_path = self.image_meta[0][:-4]
        mask_path = mask_path+"_thresh_mask.png"
        mask_path = os.path.join(os.getcwd(), mask_path)
       
        #Write file to disc
        cv2.imwrite(mask_path, self.thresh_image)

        
#######################################################################################################################
## Class For Interactive BBox Operations
#######################################################################################################################

class bbox_select():
    
    def __init__(self, img_path):
        
        # Load the color image passed in 
        self.color_image, self.image_name = get_Image(img_path)
        
        # Load the threshold mask, load as RGB to do color annotation on it
        thresh_img = get_thresh_image(self.image_name)
        self.thres_mask_rgb = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2RGB)
        # Load the threshold mask, normalize for easy computation
        self.thres_mask = (thresh_img/255).astype(np.uint8)
        
        # Load the zero mask, normalize for easy computation
        self.zero_mask = (get_zero_mask(self.image_name, self.color_image.shape)/255).astype(np.uint8)
         
        
        # Temporary Mask
        self.temp_mask = np.zeros_like(self.zero_mask, dtype = np.uint8)
        
        # Array to store polygon coordinates
        self.selected_points = []
        
        # Create new figure
        self.bbox_figure = plt.figure(2, constrained_layout=True, figsize=(9, 9))
        
        # Create event handler for clicks on matplotlib canvas
        self.mouse_event = self.bbox_figure.canvas.mpl_connect('button_press_event', self.onclick)
        
        # Create a grid spec for figure
        gs = self.bbox_figure.add_gridspec(4, 4)
        
        # Create first subplot
        self.bbox_figure_ax1 = self.bbox_figure.add_subplot(gs[0:2, 0:2])
        self.bbox_figure_ax1.set_title('1. Color Image')
        self.bbox_figure_ax1.imshow(self.color_image.copy())
        
        
        # Create second subplot
        self.bbox_figure_ax2 = self.bbox_figure.add_subplot(gs[2:4, 0:2])
        self.bbox_figure_ax2.set_title('3. Preview Mask')
        self.bbox_figure_ax2.imshow(self.temp_mask.copy(), cmap = 'gray')
        
        # Create third subplot
        self.bbox_figure_ax3 = self.bbox_figure.add_subplot(gs[2:4, 2:4])
        self.bbox_figure_ax3.set_title('4. Segmentation Mask')
        self.bbox_figure_ax3.imshow(self.zero_mask.copy(), cmap = 'gray')
        
        
        # Create fourth subplot
        self.bbox_figure_ax4 = self.bbox_figure.add_subplot(gs[0:2, 2:4])
        self.bbox_figure_ax4.set_title('2. Threshold Mask')
        self.thresh_view = self.bbox_figure_ax4.imshow(self.thres_mask_rgb.copy())
        
        
        
        # Create a refresh button to display on matplotlib canvas
        refresh_button = widgets.Button(description='Refresh',
                                        disabled=False,
                                        button_style = 'danger', # 'success', 'info', 'warning', 'danger' or ''
                                        tooltip='Click me',
                                        icon='refresh' # (FontAwesome names without the `fa-` prefix)
                                       )
        # Display Refresh Button
        Disp.display(refresh_button)
        refresh_button.on_click(self.refresh)
        
        
        # Create a preview button to display on matplotlib canvas
        preview_button = widgets.Button(description='Preview',
                                        disabled=False,
                                        button_style = 'info', # 'success', 'info', 'warning', 'danger' or ''
                                        tooltip='Click me',
                                        icon='eye' # (FontAwesome names without the `fa-` prefix)
                                       )
        # Display Refresh Button
        Disp.display(preview_button)
        preview_button.on_click(self.preview)
        
        
        # Create a save button to display on matplotlib canvas
        save_button = widgets.Button(description='Save',
                                        disabled=False,
                                        button_style = 'success', # 'success', 'info', 'warning', 'danger' or ''
                                        tooltip='Click me',
                                        icon='check-circle-o' # (FontAwesome names without the `fa-` prefix)
                                       )
        # Display Refresh Button
        Disp.display(save_button)
        save_button.on_click(self.save_state)
        
  
    
    
    def poly_img(self, img, pts):
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (255, 0, 0), 2)
        return img

    
    def onclick(self, event):
    #display(str(event))
        self.selected_points.append([event.xdata,event.ydata])
        if len(self.selected_points)>1:
            self.bbox_figure
            self.thresh_view.set_data(self.poly_img(self.thres_mask_rgb.copy(), self.selected_points))
    
        
    def refresh(self, _):

        # Reset Selected points
        self.selected_points = []
        #Reset Plot View
        self.thresh_view = self.bbox_figure_ax4.imshow(self.thres_mask_rgb.copy())
  
        #Reset temp mask
        self.temp_mask = np.zeros_like(self.zero_mask, dtype = np.uint8)
        #Plot Empty mask
        self.bbox_figure_ax2.imshow(self.temp_mask, cmap = 'gray')
     
    def preview(self, _):
        
        # Convert Coordiantes of the polygon to numpy array
        self.np_selected_points = np.array([self.selected_points],'int')
        
        #Fill polygon region with ones
        self.fill_mask = cv2.fillPoly(np.zeros(self.zero_mask.shape, np.uint8), self.np_selected_points, [1])
        # Apply fill mask 
        self.temp_mask = np.multiply(self.thres_mask, self.fill_mask)

        #Plot updated mask
        self.bbox_figure_ax2.imshow(self.temp_mask.copy(), cmap = 'gray')
        
    def save_state(self, _):
        
        # Convert Coordiantes of the polygon to numpy array
        self.np_selected_points = np.array([self.selected_points],'int')
        
        #Fill polygon region with ones
        self.fill_mask = cv2.fillPoly(np.zeros(self.zero_mask.shape, np.uint8), self.np_selected_points, [1])
        # Apply fill mask 
        self.fill_mask = np.multiply(self.thres_mask, self.fill_mask)
        
        #Update zero-mask
        self.zero_mask = self.zero_mask + self.fill_mask
        #Make sure no vlaue goes over 1
        self.zero_mask = np.where(self.zero_mask > 0,  1, 0)
        
        #Plot updated mask
        self.bbox_figure_ax3.imshow(self.zero_mask, cmap = 'gray')
        
        #Write image to disc
        final_mask_path = self.image_name[:-4]
        final_mask_path = final_mask_path+"_segment_mask.png"
        final_mask_path = os.path.join(os.getcwd(), final_mask_path)

        cv2.imwrite(final_mask_path, (self.zero_mask*255).astype(np.uint8))
        