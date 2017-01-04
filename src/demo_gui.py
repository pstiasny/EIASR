import sys
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import *
from scipy.misc import imread, imsave

from canny import gradient, thin_nonmaximum

class Image():

	def __init__(self, img_path):
		self.path = img_path
		self.img = imread(img_path)
		self.gray = np.dot(self.img[...,:3], [0.299, 0.587, 0.114])
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

		exit_but = QPushButton('Quit', self)
		exit_but.clicked.connect(QCoreApplication.instance().quit)
		exit_but.resize(exit_but.sizeHint())
		exit_but.move(615, 360)

		load_but = QPushButton('Load images', self)
		load_but.clicked.connect(self.loadImagesClicked)
		load_but.resize(load_but.sizeHint())
		load_but.move(600, 330)

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
		img = self.activeImg
		pics = [img.img, img.gray, img.angles, img.magnitudes, img.thinned]
		pics = [p for p in pics if p != None]
		pics = [imgToQImg(p) for p in pics]
		self.ig.populate(pics)
		self.ig.setStyleSheet('background-color: gray')
		self.ig.show()
		
	def double_clicked_cell(self, row, column):

		self.images[row].compute()
		self.refreshImgViews()
		

	def clicked_cell(self, row, column):

		self.activeImg = self.images[row]
		self.setImgForMainImgView(imgToQImg(self.activeImg.img))
		self.refreshImgViews()


	def loadImagesClicked(self):
		
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


	def setImgForMainImgView(self, qimg):
		main_pixm = QPixmap(qimg)
		main_pixm = main_pixm.scaled(self.main_img_view.size(), Qt.KeepAspectRatio)
		self.main_img_view.setPixmap(main_pixm)



	
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
