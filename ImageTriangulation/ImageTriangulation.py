import cv2, functools, math, random, sys, time, warnings
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import operator as op
import pandas as pd

from matplotlib.patches import Polygon
from PIL import Image
from queue import PriorityQueue
from scipy.spatial import Delaunay
from skimage.color import rgb2gray, rgb2lab
from skimage.filters import sobel
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.restoration import denoise_tv_bregman
from skimage.util import img_as_ubyte

PI = 3.141592653589

DEBUG = False

class Point:
    def __init__(self, x, y, priority=0.0):
        self.x = x
        self.y = y
        self.priority = priority

    def __str__(self):
        return "[%d, %d]" % (self.x, self.y)

    def __repr__(self):
        return "[%d, %d]" % (self.x, self.y)

    def __iter__(self):
        for i in [self.x, self.y]:
            yield i

    def __lt__(self, other):
        return self.priority < other.priority
    
    def generateRandomPointAround(self, minDist):
        r1 = random.random()
        r2 = random.random()
        
        radius = minDist * (1. + r1)
        angle = 2 * PI * r2

        newPointX = self.x + int(radius * math.cos(angle))
        newPointY = self.y + int(radius * math.sin(angle))

        return Point(newPointX, newPointY)

    def distance(self, point):
        return math.sqrt((point.x - self.x)**2 + (point.y - self.y)**2)

class Grid:
    def __init__(self, minDist, imageWidth, imageHeight):
        self.cellSize = minDist / math.sqrt(2)
        self.minDist = minDist
        self.width = math.ceil(imageWidth / self.cellSize)
        self.height = math.ceil(imageHeight / self.cellSize)
        self.grid = [None] * self.width * self.height

    def getGridCoordinates(self, point):
        return Point(int(point.x / self.cellSize), int(point.y / self.cellSize))

    '''
    This is a dumb/quick way of calculating grid array index based on whether the point coordinates
    are in the image space or the grid space. If gridSpace = False, the point coordinates are first
    converted to gridspace from imagespace before returning the calculated grid index. 
    '''
    def getIndex(self, x, y, gridSpace = False):
        if not gridSpace:
            x, y = list(self.getGridCoordinates(Point(x, y)))
        return y * self.width + x
            

    def insert(self, point):
        #gridCoords = self.getGridCoordinates(point)
        #self.grid[gridCoords.y * self.width + gridCoords.x] = point
        self.grid[self.getIndex(point.x, point.y)] = point

    def checkInNeighborhood(self, point):
        gridCoords = self.getGridCoordinates(point)
        radius = 2
        for y in range(gridCoords.y - radius, gridCoords.y + radius):
            for x in range(gridCoords.x - radius, gridCoords.x + radius):
                index = self.getIndex(x, y, True)
                if index in range(0, len(self.grid)):
                    neighbor = self.grid[index]
                    if neighbor and point.distance(neighbor) < self.minDist:
                        return True;
        return False
    
    def reset(self):
        self.grid = [None] * len(self.grid)

class BlueNoiseGenerator:
    def __init__(self, pointCount, image, grid, minDist, newPointsGenerationCount = 50):
        self.pointCount = pointCount
        self.newPointsToGenerate = newPointsGenerationCount
        self.image = image
        self.imageHeight, self.imageWidth = image.shape[:2]
        self.grid = grid
        self.minDist = grid.minDist
        self.sampledList = []
        self.processingList = []
        self.imageWeight = None
        self.edgeThreshold = minDist # could be a function of the minDist, but this works for now

    def generateBoundary(self):
        # generate points around the edge (this is optional, but the final image looks a bit more complete)
        upperWidth = 0.0
        lowerWidth = 0.0
        leftHeight = 0.0
        rightHeight = 0.0
        twiceMinDist = 2 * self.minDist
        aspectRatio = float(self.imageWidth) / self.imageHeight if self.imageWidth > self.imageHeight \
            else float(self.imageHeight) / self.imageWidth
        while lowerWidth < self.imageWidth and upperWidth < self.imageWidth \
            and leftHeight < self.imageHeight and rightHeight < self.imageHeight:
            self.sampledList.append([upperWidth, 0])
            self.sampledList.append([lowerWidth, self.imageHeight - 1])
            self.sampledList.append([0, leftHeight])
            self.sampledList.append([self.imageWidth - 1, rightHeight])
            if self.imageWidth > self.imageHeight:
                upperDelta = random.uniform(self.minDist * aspectRatio, twiceMinDist * aspectRatio)
                lowerDelta = random.uniform(self.minDist * aspectRatio, twiceMinDist * aspectRatio)
            else:
                upperDelta = random.uniform(self.minDist, twiceMinDist)
                lowerDelta = random.uniform(self.minDist, twiceMinDist)
            if self.imageHeight > self.imageWidth:
                leftDelta = random.uniform(self.minDist * aspectRatio, twiceMinDist * aspectRatio)
                rightDelta = random.uniform(self.minDist * aspectRatio, twiceMinDist * aspectRatio)
            else:
                leftDelta = random.uniform(self.minDist, twiceMinDist)
                rightDelta = random.uniform(self.minDist, twiceMinDist)
            upperWidth += upperDelta
            lowerWidth += lowerDelta
            leftHeight += leftDelta
            rightHeight += rightDelta
        self.sampledList.append([self.imageWidth - 1, 0])
        self.sampledList.append([self.imageWidth - 1, self.imageHeight - 1])

    def generateWeighted(self):

        # preprocessing
        if not self.imageWeight:
            image_denoise = denoise_tv_bregman(self.image, 20.0)
            image_gray = rgb2gray(image_denoise)
            image_lab = rgb2lab(image_denoise)

            image_entropy = 2**(entropy(img_as_ubyte(image_gray), disk(20)))
            image_entropy /= np.max(image_entropy)

            color = [sobel(image_lab[:, :, channel])**2 for channel in range(1, 3)]
            image_sobel = functools.reduce(op.add, color) ** (1/2) / 5

            self.imageWeight = (0.3*image_entropy + 0.7*image_sobel)
            self.imageWeight /= np.mean(self.imageWeight)
        
        # blue noise generation
        startingPoint = Point(random.randint(0, self.imageWidth), random.randint(0, self.imageHeight))
        
        self.sampledList.append(list(startingPoint))
        self.processingList.append(startingPoint)
        self.grid.insert(startingPoint)
        
        pQueue = PriorityQueue()

        while len(self.processingList) > 0 and len(self.sampledList) < self.pointCount:
            randomPoint = self.processingList.pop(random.randint(0, len(self.processingList) - 1))
            for i in range(0, self.newPointsToGenerate):
                newPoint = randomPoint.generateRandomPointAround(self.minDist)
                # early boundary check 
                if newPoint.x > self.edgeThreshold and newPoint.x < self.imageWidth - self.edgeThreshold \
                    and newPoint.y > self.edgeThreshold and newPoint.y < self.imageHeight - self.edgeThreshold:
                    newPoint.priority = -self.imageWeight[newPoint.y][newPoint.x]
                    pQueue.put(newPoint)

            found = False
            while not pQueue.empty() and not found:
                newPoint = pQueue.get()
                if not self.grid.checkInNeighborhood(newPoint):
                    self.processingList.append(newPoint)
                    self.sampledList.append(list(newPoint))
                    self.grid.insert(newPoint)
                    found = True

        if DEBUG:
            im_info = Image.open(sys.argv[1]).info
            if 'dpi' in im_info:
                my_dpi = im_info['dpi'][0]
            else:
                my_dpi = 60
            figSize = self.imageWidth / my_dpi, self.imageHeight / my_dpi
            fig = plt.figure(figsize=figSize, dpi=my_dpi)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.plot(*zip(*self.sampledList), color='black',marker=',',lw=0, linestyle="")
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(('%d_points_weighted_blue_noise.png' % len(self.sampledList)), bbox_inches=extent.expanded(0.9, 0.9), pad_inches=0, dpi=my_dpi)

            ax.clear()
            ax.plot(*zip(*self.sampledList), color='yellow',marker=',',lw=0, linestyle="")
            ax.imshow(self.imageWeight, cmap='gray')
            fig.savefig(('%d_points_weighted_blue_noise_filter.png' % len(self.sampledList)), bbox_inches=extent.expanded(0.9, 0.9), pad_inches=0, dpi=my_dpi)

            img = cv2.imread(('%d_points_weighted_blue_noise.png' % len(self.sampledList)),0)
            img2 = img
            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = np.log(np.abs(fshift))

            ax.clear()
            ax.imshow(magnitude_spectrum, cmap='gray')
            fig.savefig(('%d_points_weighted_blue_noise_dft.png' % len(self.sampledList)), bbox_inches=extent.expanded(0.9, 0.9), pad_inches=0, dpi=my_dpi)
        
        self.generateBoundary()

        return self.sampledList

    def generate(self):

        startingPoint = Point(random.randint(0, self.imageWidth), random.randint(0, self.imageHeight))
        self.sampledList.append(list(startingPoint))
        self.processingList.append(startingPoint)
        self.grid.insert(startingPoint)

        edgeThreshold = 3

        while len(self.processingList) > 0 and len(self.sampledList) < self.pointCount:
            randomPoint = self.processingList.pop(random.randint(0, len(self.processingList) - 1))
            for i in range(0, self.newPointsToGenerate):
                newPoint = randomPoint.generateRandomPointAround(self.minDist)
                if newPoint.x > self.edgeThreshold and newPoint.x < self.imageWidth - self.edgeThreshold \
                and newPoint.y > self.edgeThreshold and newPoint.y < self.imageHeight - self.edgeThreshold \
                and not self.grid.checkInNeighborhood(newPoint):
                    self.processingList.append(newPoint)
                    self.sampledList.append(list(newPoint))
                    self.grid.insert(newPoint)
                    continue

        if DEBUG:
            im_info = Image.open(sys.argv[1]).info
            if 'dpi' in im_info:
                my_dpi = im_info['dpi'][0]
            else:
                my_dpi = 60
            figSize = self.imageWidth / my_dpi, self.imageHeight / my_dpi
            fig = plt.figure(figsize=figSize, dpi=my_dpi)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.plot(*zip(*self.sampledList), color='black',marker=',',lw=0, linestyle="")
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(('%d_points_blue_noise.png' % len(self.sampledList)), bbox_inches=extent.expanded(0.9, 0.9), pad_inches=0, dpi=my_dpi)

            img = cv2.imread(('%d_points_blue_noise.png' % len(self.sampledList)),0)
            img2 = img
            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = np.log(np.abs(fshift))

            ax.clear()
            ax.imshow(magnitude_spectrum, cmap='gray')
            fig.savefig(('%d_points_blue_noise_dft.png' % len(self.sampledList)), bbox_inches=extent.expanded(0.9, 0.9), pad_inches=0, dpi=my_dpi)

        self.generateBoundary()

        return self.sampledList

def calculateTriColors(image, triangulation, aggregateFunc=np.mean):
    imageHeight, imageWidth = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(imageWidth), np.arange(imageHeight))
    pixelCoords = np.c_[xx.ravel(), yy.ravel()]
    
    pixeltoTriangle = triangulation.find_simplex(pixelCoords)

    df = pd.DataFrame({
        "triangle": pixeltoTriangle,
        "r": image.reshape(-1, 3)[:, 0],
        "g": image.reshape(-1, 3)[:, 1],
        "b": image.reshape(-1, 3)[:, 2]
    })

    triangleCount = triangulation.simplices.shape[0]

    triangleColors = (df.groupby("triangle")[["r", "g", "b"]].aggregate(aggregateFunc).reindex(range(triangleCount), fill_value=0))

    return triangleColors.values / 256
    
if __name__ == '__main__':
    
    if (len(sys.argv) < 3):
        print ("usage: python3 ImageTriangulation.py <input_file_path> <min_dist_between_samples>")
    
    warnings.filterwarnings("ignore")
    currentMilliseconds = lambda: int(round(time.time() * 1000))

    print ("Reading Image...")

    image = plt.imread(sys.argv[1])
    imageHeight, imageWidth = image.shape[:2]
    minDist = int(sys.argv[2])
    grid = Grid(minDist, imageWidth, imageHeight)

    imageCount = 1

    for i in range(imageCount):

            print("Generating blue noise...")
        
            start = currentMilliseconds()
            grid.reset()
            numPoints = int(imageWidth * imageHeight / (grid.cellSize ** 2))
            #blueNoise = BlueNoiseGenerator(numPoints, image, grid, minDist).generate()
            blueNoise = BlueNoiseGenerator(numPoints, image, grid, minDist).generateWeighted()
            end = currentMilliseconds()
        
            print("Blue noise generated in %f seconds with %d points" % ((end - start)/1000.0, len(blueNoise)))
            
            tris = Delaunay(blueNoise)
            print ("%d triangles generated from %d points." % (tris.simplices.shape[0], len(blueNoise)))

            print ("Calculating triangle colors...")
            start = currentMilliseconds()
            triColors = calculateTriColors(image, tris, np.median)
            end = currentMilliseconds()

            print ("Colorized triangles in %f seconds" % ((end - start)/1000.0))

            print ("Saving image...")  
            
            im_info = Image.open(sys.argv[1]).info
            if 'dpi' in im_info:
                my_dpi = im_info['dpi'][0]
            else:
                my_dpi = 60

            print ('dpi: %d' % my_dpi)
            
            figSize = imageWidth / my_dpi, imageHeight / my_dpi
            fig = plt.figure(figsize=figSize, dpi=my_dpi)
            ax = fig.add_axes([0., 0., 1., 1.])
            ax.invert_yaxis()
            for triangle, eColor, fColor in zip(tris.simplices, triColors, triColors):
                p = Polygon([tris.points[i] for i in triangle], closed=True, facecolor=fColor, edgecolor=eColor)
                ax.add_patch(p)
            ax.axis('tight')
            ax.set_axis_off()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            outfile_name = "%s_%s_%s.png" % (sys.argv[1].split('.')[0], str(tris.simplices.shape[0]), str(i))
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(outfile_name, bbox_inches=extent.expanded(0.9, 0.9), pad_inches=0, dpi=my_dpi)
            