import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as anim
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import nibabel as nib
import nilearn as nl
import nilearn.plotting as nlplt
import numpy as np
import nrrd
import h5py
import nilearn.plotting as nlplt
from skimage.transform import resize
from skimage.util import montage
from numba import jit, cuda
import imageio

from IPython.display import clear_output
from IPython.display import YouTubeVideo
from IPython.display import Image as show_gif

import warnings

warnings.simplefilter("ignore")

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class ImageToGIF:
    """Create GIF without saving image files."""

    def __init__(self,
                 size=(600, 400),
                 xy_text=(80, 10),
                 dpi=100,
                 cmap='CMRmap'):

        self.fig = plt.figure()
        self.fig.set_size_inches(size[0] / dpi, size[1] / dpi)
        self.xy_text = xy_text
        self.cmap = cmap

        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.images = []

    def add(self, *args, label, with_mask=True):

        image = args[0]
        mask = args[-1]
        plt.set_cmap(self.cmap)
        plt_img = self.ax.imshow(image, animated=True)
        if with_mask:
            plt_mask = self.ax.imshow(np.ma.masked_where(mask == False, mask),
                                      alpha=0.7,
                                      animated=True)

        plt_text = self.ax.text(*self.xy_text, label, color='red')
        to_plot = [plt_img, plt_mask, plt_text
                   ] if with_mask else [plt_img, plt_text]
        self.images.append(to_plot)
        plt.close()

    def save(self, filename, fps):
        animation = anim.ArtistAnimation(self.fig, self.images)
        animation.save(filename, writer='pillow', fps=fps)


class Image3dToGIF3d:
    """
    Displaying 3D images in 3d axes.
    Parameters:
        img_dim: shape of cube for resizing.
        figsize: figure size for plotting in inches.
    """

    def __init__(
            self,
            img_dim: tuple = (55, 55, 55),
            figsize: tuple = (15, 10),
    ):
        """Initialization."""
        self.img_dim = img_dim
        print(img_dim)
        self.figsize = figsize

    def _explode(self, data: np.ndarray):
        """
        Takes: array and return an array twice as large in each dimension,
        with an extra space between each voxel.
        """
        shape_arr = np.array(data.shape)
        size = shape_arr[:3] * 2 - 1
        exploded = np.zeros(np.concatenate([size, shape_arr[3:]]),
                            dtype=data.dtype)
        exploded[::2, ::2, ::2] = data
        return exploded

    def _expand_coordinates(self, indices: np.ndarray):
        x, y, z = indices
        x[1::2, :, :] += 1
        y[:, 1::2, :] += 1
        z[:, :, 1::2] += 1
        return x, y, z

    def _normalize(self, arr: np.ndarray):
        """Normilize image value between 0 and 1."""
        return arr / arr.max()

    def _scale_by(self, arr: np.ndarray, factor: int):
        """
        Scale 3d Image to factor.
        Parameters:
            arr: 3d image for scalling.
            factor: factor for scalling.
        """
        mean = np.mean(arr)
        return (arr - mean) * factor + mean

    def get_transformed_data(self, data: np.ndarray):
        """Data transformation: normalization, scaling, resizing."""
        norm_data = np.clip(self._normalize(data) - 0.1, 0, 1)**0.4
        scaled_data = np.clip(self._scale_by(norm_data, 2) - 0.1, 0, 1)
        resized_data = resize(scaled_data, self.img_dim, mode='constant')
        return resized_data

    
    def plot_cube(self,
                  cube,
                  title: str = '',
                  init_angle: int = 0,
                  make_gif: bool = False,
                  path_to_save: str = 'filename.gif'):
        """
        Plot 3d data.
        Parameters:
            cube: 3d data
            title: title for figure.
            init_angle: angle for image plot (from 0-360).
            make_gif: if True create gif from every 5th frames from 3d image plot.
            path_to_save: path to save GIF file.
            """
        cube = self._normalize(cube)

        facecolors = cm.gist_stern(cube)
        facecolors[:, :, :, -1] = cube
        facecolors = self._explode(facecolors)

        filled = facecolors[:, :, :, -1] != 0
        x, y, z = self._expand_coordinates(
            np.indices(np.array(filled.shape) + 1))

        with plt.style.context("dark_background"):

            fig = plt.figure(figsize=self.figsize)
            ax = plt.axes(projection='3d')

            ax.view_init(30, init_angle)
            ax.set_xlim(right=self.img_dim[0] * 2)
            ax.set_ylim(top=self.img_dim[1] * 2)
            ax.set_zlim(top=self.img_dim[2] * 2)
            ax.set_title(title, fontsize=18, y=1.05)

            ax.voxels(x, y, z, filled, facecolors=facecolors, shade=False)

            if make_gif:
                images = []
                for angle in tqdm(range(0, 360, 5)):
                    ax.view_init(30, angle)
                    fname = str(angle) + '.png'

                    plt.savefig(fname,
                                dpi=120,
                                format='png',
                                bbox_inches='tight')
                    images.append(imageio.imread(fname))
                    #os.remove(fname)
                imageio.mimsave(path_to_save, images)
                plt.close()

            else:
                plt.show()

class CorrectedPrceess:

    def __init__(self, orign_path_img, correcte_img):

        self.orign_img = orign_path_img
        self.corrected_img = correcte_img

    def virtualize_bias(self, save_path_img, type_virtualizer):
        #take the last part of path as sting to allocate in the title of image
        before_img = nl.image.load_img(self.orign_img)
        correction_img = nl.image.load_img(self.corrected_img)
        name_img = self.orign_img.split('/')[-1]
        fig, axes = plt.subplots(nrows=2, figsize=(15, 20))

        #the function to plot the corrected with oring img
        #     type_virtualizer{
        #     option 1 = Anat ,
        #     option 2 = epi ,
        #     option 3 = img ,
        #     option 4 = roi
        #  }
        if type_virtualizer == 'anat':

            nlplt.plot_anat(before_img,
                            title=name_img + 'oring plot_anat',
                            axes=axes[0])
            nlplt.plot_anat(correction_img,
                            title=name_img + 'corrcted plot_anat',
                            axes=axes[1])
            fig.savefig(save_path_img + name_img + 'comparing_corrected.png')

            #plt.show()
        # Plot cuts of an EPI image (by default 3 cuts: Frontal, Axial, and Lateral)
        elif type_virtualizer == 'epi':

            nlplt.plot_epi(before_img,
                           title=name_img + 'oring plot_epi',
                           axes=axes[0])
            nlplt.plot_epi(correction_img,
                           title=name_img + 'corrcted plot_epi',
                           axes=axes[1])
            fig.savefig(save_path_img + name_img +
                        'comparing_corrected_with_oring.png')
        # Plot cuts of an ROI/mask image (by default 3 cuts: Frontal, Axial, and Lateral)
        elif type_virtualizer == 'img':

            nlplt.plot_img(before_img,
                           title=name_img + 'oring plot_img',
                           axes=axes[0])
            nlplt.plot_img(correction_img,
                           title=name_img + 'corrcted plot_img',
                           axes=axes[1])
            fig.savefig(save_path_img + name_img +
                        'comparing_corrected_with_oring.png')
        else:
              print("make sure you chose the corrected type plot")

        return plt.show()
