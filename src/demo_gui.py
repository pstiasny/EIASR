import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4 import QtGui, QtCore, uic
import cv2

from scipy.misc import imread, imsave
from canny import gradient, thin_nonmaximum
import numpy as np
from functools import partial 

class Image():

	def __init__(self, img_path):
		self.path = img_path
		self.img = cv2.imread(img_path)
		self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
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
	def magnitudes(self):
		if hasattr(self,"_magnitudes"):
			return self._magnitudes
			
	@property
	def angles(self):
		if hasattr(self,"_angles"):
			return self._angles

	@property
	def thinned(self):
		if hasattr(self,"_thinned"):
			return self._thinned
		
	def compute(self):
		if not self.computed:
			self._gradient = gradient(self.gray)
			self._magnitudes =  self.gradient.magnitudes
			self._angles =  self.gradient.angles
			self._thinned =  thin_nonmaximum(self.gradient)
			self.computed = True


def cvImgToQtImg(cvImg):
	
	if len(cvImg.shape) == 3:
		cvImg = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)
		height, width, channel = cvImg.shape
		bytesPerLine = 3 * width
		return QImage(cvImg.data, width, height, bytesPerLine, QImage.Format_RGB888)
	else:
		
		normalizedImg = np.zeros(cvImg.shape)
		normalizedImg = cv2.normalize(cvImg,  normalizedImg, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

		colorTable = [qRgb(c,c,c) for c in range(256)]
		height, width = normalizedImg.shape
		bytesPerLine =  width
		qimg = QImage(normalizedImg.data, width, height, bytesPerLine, QImage.Format_Indexed8)
		qimg.setColorTable(colorTable)
		return qimg


class Canny(QtGui.QWidget):
    
	def __init__(self):

		super(Canny, self).__init__()
		self.images = []
		self.initUI()
        
	def initUI(self):               
        
		self.setGeometry(0, 0, 780, 420)
		self.setWindowTitle('Jegermeister detector 3000')   
		self.setupImgViews()
		self.setupButtons()
		self.setupTable()
		self.show()

	def setupImgViews(self):

		
		self.ig = ImageGallery(parent=self)
		self.ig.setGeometry(410, 230, 370, 190)

		self.main_img_view = QtGui.QLabel(self)
		self.main_img_view.setStyleSheet('background-color: gray')
		self.main_img_view.setGeometry(10, 10, 400, 400)


	def setupButtons(self):

		exit_but = QtGui.QPushButton('Quit', self)
		exit_but.clicked.connect(QtCore.QCoreApplication.instance().quit)
		exit_but.resize(exit_but.sizeHint())
		exit_but.move(615, 360)


		load_but = QtGui.QPushButton('Load images', self)
		load_but.clicked.connect(self.loadImagesClicked)
		load_but.resize(load_but.sizeHint())
		load_but.move(600, 330)

	def setupTable(self):

		self.table = QTableWidget(self)
		self.table.resize(350, 220)
		self.table.setColumnCount(2)
		self.table.move(420, 10)
		self.table.setHorizontalHeaderItem(0, QtGui.QTableWidgetItem("Image"))
		self.table.setHorizontalHeaderItem(1, QtGui.QTableWidgetItem("Result"))
		self.table.setColumnWidth(0, 300)
		self.table.setColumnWidth(1, 40)
		self.table.cellDoubleClicked.connect(self.double_clicked_cell)
		self.table.cellClicked.connect(self.clicked_cell)
		self.table.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
		self.table.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)

	
	def updateTable(self):
		self.table.setRowCount(len(self.images))
		for idx, img in enumerate(self.images):
			self.table.setItem(idx,0, QTableWidgetItem(img.name))

	def refreshImgViews(self, img):
		
		pics = [img.img, img.gray, img.angles, img.magnitudes, img.thinned]
		pics = [cvImgToQtImg(p) for p in pics]
		self.ig.populate(pics, QSize(80,80))
		self.ig.setStyleSheet('background-color: gray')
		self.ig.show()
		
	def double_clicked_cell(self, row, column):

		self.images[row].compute()
		self.refreshImgViews(self.images[row])
		

	def clicked_cell(self, row, column):

		img = self.images[row]
		main_pixm = QtGui.QPixmap(cvImgToQtImg(img.img))
		main_pixm = main_pixm.scaled(self.main_img_view.size(), Qt.KeepAspectRatio)
		self.main_img_view.setPixmap(main_pixm)
		self.setImgForMainImgView


	def loadImagesClicked(self):
		
		fileDialog = QtGui.QFileDialog(self)
		fileDialog.setFileMode(QFileDialog.ExistingFiles)
		fileDialog.setFilter("Images (*.jpg *.png)")
		fileDialog.show()

		if fileDialog.exec_():
			existing_imgs = [img.path for img in self.images]
			filenames =  fileDialog.selectedFiles() 
			images = [Image(str(f_n)) for f_n in filenames if f_n not in existing_imgs]
			self.images.extend(images)
			self.updateTable()


	def setImgForMainImgView(self, qimg):
		main_pixm = QtGui.QPixmap(qimg)
		main_pixm = main_pixm.scaled(self.main_img_view.size(), Qt.KeepAspectRatio)
		self.main_img_view.setPixmap(main_pixm)



	
class ImageGallery(QWidget):
    
	def __init__(self, parent=None):
		super(QWidget, self).__init__(parent)
		self.grid = QGridLayout()
		self.setLayout(self.grid)	
		self.mousePressEvent = self.changeActiveThumb


    
	def populate(self, pics, size, imagesPerRow=4, flags=Qt.KeepAspectRatio):
		row = col = 0
		self.pics = pics
		for pic in pics:
			label = ImageLabel(self)
			pixmap = QPixmap(pic)
			pixmap = pixmap.scaled(size, flags)
			label.setPixmap(pixmap)
			self.grid.addWidget(label, row, col)

			col +=1
			if col % imagesPerRow == 0:
				row += 1
				col = 0
		self.grid.itemAt(0).widget().active = True

	def changeActiveThumb(self, event):

		clickedLab = self.childAt(event.pos())
		clickedIdx = self.grid.indexOf(clickedLab)
		if not clickedLab.active:
			clickedLab.active = True
			self.activeLabIdx = clickedIdx
			items = [self.grid.itemAt(i).widget() for i in range(self.grid.count())]
			for i in items:
				if not i == clickedLab:
					i.active = False
		self.parentWidget().setImgForMainImgView(self.pics[clickedIdx])
			


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
    
    app = QtGui.QApplication(sys.argv)
    ex = Canny()
    sys.exit(app.exec_())
	

if __name__ == '__main__':
	main()