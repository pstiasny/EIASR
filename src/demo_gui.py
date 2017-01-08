import sys
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import *
from scipy.misc import imread, imsave

from canny import gradient, thin_nonmaximum, thin_hysteresis
from hough import hough_learn, hough_detect

class Image():

	def __init__(self, img_path):
		self.path = img_path
		self.gray = imread(img_path, flatten=True, mode='L')
		# if self.img.shape[2] < 3:
		# 	self.gray = self.img
		# else:
		# 	self.gray = np.dot(self.img[...,:3], [0.299, 0.587, 0.114])
		self.computed = False

	@property
	def name(self):
		if not hasattr(self,"_name"):
			self._name = self.path.split('/')[-1]
		return self._name

	@property
	def gradient(self):
		if hasattr(self, "_gradient"):
			return self._gradient

	@property
	def thinned_nmax(self):
		if hasattr(self,"_thinned_nmax"):
			return self._thinned_nmax

	@property
	def thinned_hyst(self):
		if hasattr(self,"_thinned_hyst"):
			return self._thinned_hyst
	
	def visualImages(self):
		images = [self.gray]
		if self.computed:
			images.extend([self.gradient.magnitudes, self.thinned_nmax.magnitudes, self.thinned_hyst.magnitudes])
		if hasattr(self, "ght_res"):
			images.append(self.ght_res)
		return [imgToQImg(im) for im in images]

	def compute(self):

		if not self.computed:
			self._gradient = gradient(self.gray)
			self._thinned_nmax =  thin_nonmaximum(self.gradient)
			self._thinned_hyst =  thin_hysteresis(self.thinned_nmax)
			self.computed = True


class HoughShapeDetector():

	def __init__(self, template):
		self.template = template
		self.train()

	def train(self):
		self.template.compute()
		self.rtable = hough_learn(self.template.thinned_hyst)

	def detect(self, img):
		img.compute()
		result = hough_detect(self.rtable, img.thinned_hyst)
		img.ght_res = np.sum(result.accumulator, axis=(0, 1))
		

def imgToQImg(img):

	if img is None:
		return
	
	if len(img.shape) == 3:
		height, width, channel = img.shape
		bytesPerLine = 3 * width
		return QImage(img, width, height, bytesPerLine, QImage.Format_RGB888)
	else:
		height, width = img.shape
		bytesPerLine =  width
		mn = np.min(img)
		mx = np.max(img)
		img = np.uint8((img - mn)*255/(mx - mn))
		return QImage(img, width, height, bytesPerLine, QImage.Format_Indexed8)


class Canny(QWidget):
    
	def __init__(self):

		super(Canny, self).__init__()
		self.images = []
		self.initUI()

	def initUI(self):               
        
		self.setGeometry(0, 0, 780, 420)
		self.setWindowTitle('Jagermeister detector 3000')   
		self.setupImgViews()
		self.setupButtons()
		self.setupTable()
		self.show()

	def setupImgViews(self):
		
		self.ig = ImageGallery(parent=self)
		self.ig.setGeometry(410, 230, 370, 190)

		self.main_img_view = QLabel(self)
		self.main_img_view.setStyleSheet('background-color: gray')
		self.main_img_view.setGeometry(10, 10, 400, 400)


	def setupButtons(self):

		load_imgs_but = QPushButton('Load imgs', self)
		load_imgs_but.clicked.connect(self.loadImages)
		load_imgs_but.setMaximumWidth(85)
		# load_imgs_but.resize(load_imgs_but.sizeHint())
		load_imgs_but.move(595, 330)

		load_templ_but = QPushButton('Load template', self)
		load_templ_but.clicked.connect(self.loadTemplate)
		load_templ_but.setMaximumWidth(110)
		load_templ_but.move(670, 330)

		exit_but = QPushButton('Quit', self)
		exit_but.clicked.connect(QCoreApplication.instance().quit)
		exit_but.move(595, 380)

	def setupTable(self):

		self.table = QTableWidget(self)
		self.table.resize(350, 220)
		self.table.setColumnCount(2)
		self.table.move(420, 10)
		self.table.setHorizontalHeaderItem(0, QTableWidgetItem("Image"))
		self.table.setHorizontalHeaderItem(1, QTableWidgetItem("Result"))
		self.table.setColumnWidth(0, 300)
		self.table.setColumnWidth(1, 40)
		self.table.cellDoubleClicked.connect(self.double_clicked_cell)
		self.table.cellClicked.connect(self.clicked_cell)
		self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
		self.table.setSelectionBehavior(QAbstractItemView.SelectRows)

	
	def updateTable(self):
		self.table.setRowCount(len(self.images))
		for idx, img in enumerate(self.images):
			self.table.setItem(idx,0, QTableWidgetItem(img.name))

	def refreshImgViews(self):
		images = self.activeImg.visualImages()
		self.ig.populate(images)
		self.ig.setStyleSheet('background-color: gray')
		self.ig.show()

	def setImgForMainImgView(self, qimg):
		main_pixm = QPixmap(qimg)
		main_pixm = main_pixm.scaled(self.main_img_view.size(), Qt.KeepAspectRatio)
		self.main_img_view.setPixmap(main_pixm)
		
	def double_clicked_cell(self, row, column):

		self.images[row].compute()
		if hasattr(self, "detector"):
			self.detector.detect(self.images[row])

		self.refreshImgViews()
		

	def clicked_cell(self, row, column):

		self.activeImg = self.images[row]
		self.setImgForMainImgView(imgToQImg(self.activeImg.gray))
		self.refreshImgViews()


	def loadImages(self):
		
		fileDialog = QFileDialog(self)
		fileDialog.setFileMode(QFileDialog.ExistingFiles)
		fileDialog.setNameFilters(['Images (*.jpg *.png)'])
		fileDialog.show()

		if fileDialog.exec_():
			existing_imgs = [img.path for img in self.images]
			filenames =  fileDialog.selectedFiles() 
			images = [Image(str(f_n)) for f_n in filenames if f_n not in existing_imgs]
			self.images.extend(images)
			self.updateTable()

	def loadTemplate(self):
		fileDialog = QFileDialog(self)
		fileDialog.setFileMode(QFileDialog.ExistingFiles)
		fileDialog.setNameFilters(['Images (*.jpg *.png)'])
		fileDialog.show()

		if fileDialog.exec_():
			filename =  fileDialog.selectedFiles()[0]
			template = Image(str(filename))
			self.detector = HoughShapeDetector(template)


	def detectTemplateInImage(self, img):
		self.template.compute()
		

	



	
class ImageGallery(QWidget):
    
	def __init__(self, imagesPerRow=4,itemSize=80, parent=None):
		super(QWidget, self).__init__(parent)
                self.imagesPerRow = imagesPerRow
                self.itemSize = itemSize
		self.grid = QGridLayout()
		self.setLayout(self.grid)
		self.createLabels()
		
		self.mousePressEvent = self.changeActiveThumb


        def createLabels(self):
                row = col = 0
                for i in range(6):
                        label = ImageLabel(self)
                        self.grid.addWidget(label, row, col)
                        col +=1
			if col % self.imagesPerRow == 0:
				row += 1
				col = 0
                        
	def populate(self, pics, flags=Qt.KeepAspectRatio):
		row = col = 0
		self.pics = pics
		self.clearAllLabels()
		for idx, pic in enumerate(pics):
			label = self.grid.itemAt(idx).widget()
			pixmap = QPixmap(pic)
			pixmap = pixmap.scaled(self.itemSize,self.itemSize, flags)
			label.setPixmap(pixmap)
			col +=1
			if col % self.imagesPerRow == 0:
				row += 1
				col = 0
		self.deactivateAll()
		self.grid.itemAt(0).widget().active = True

	def changeActiveThumb(self, event):

		clickedLab = self.childAt(event.pos())
		clickedIdx = self.grid.indexOf(clickedLab)
		if clickedLab is None or clickedIdx >= len(self.pics):
			return
		if not clickedLab.active:
			clickedLab.active = True
			self.activeLabIdx = clickedIdx
			items = [self.grid.itemAt(i).widget() for i in range(self.grid.count())]
			self.deactivateAll()
			clickedLab.active = True
			
		self.parentWidget().setImgForMainImgView(self.pics[clickedIdx])
                
	def deactivateAll(self):
                items = [self.grid.itemAt(i).widget() for i in range(self.grid.count())]
		for i in items:
			i.active = False
                                   
	def clearAllLabels(self):
                items = [self.grid.itemAt(i).widget() for i in range(self.grid.count())]
                for lab in items:
			lab.clear()

class ImageLabel(QLabel):
	"""docstring for ImageLabel"""
	def __init__(self, parent=None):
		super(QLabel, self).__init__(parent)
		self._active = False

	@property
	def active(self):
		return self._active

	@active.setter
	def active(self, value):
		self._active = value
		if value:
			self.setStyleSheet('border: 2px solid blue')
		else:
			self.setStyleSheet('border:None')


		
def main():
    
    app = QApplication(sys.argv)
    ex = Canny()
    sys.exit(app.exec_())
	

if __name__ == '__main__':
	main()
