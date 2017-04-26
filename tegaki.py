# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
from scipy import ndimage
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import tensorflow as tf

class pline(QWidget):
	def __init__(self, parent = None):
		QWidget.__init__(self, parent)
		self.px = None
		self.py = None
		self.points = []
		self.psets = []
		self.points_saved = []
	def mousePressEvent(self, event):
		self.points.append(event.pos())
		self.update()
	def mouseMoveEvent(self, event):
		self.points.append(event.pos())
		self.update()
	def mouseReleaseEvent(self, event):
		self.pressed = False
		self.psets.append(self.points)
		self.points_saved.extend(self.points)
		self.points = []
		self.update()
	def paintEvent(self, event):
		painter = QPainter(self)
		painter.setPen(Qt.NoPen)
		painter.setBrush(Qt.white)
		painter.drawRect(self.rect())
		painter.setPen(Qt.blue)
		for points in self.psets:
			painter.drawPolyline(*points)
		if self.points:
			painter.drawPolyline(*self.points)
	def clear(self):
		self.points = []
		self.psets = []
		self.points_saved = []
		self.repaint()

class MainWindow(QWidget):
	def __init__(self, parent = None):
		super(MainWindow, self).__init__(parent)
		Button0 = QPushButton("recog")
		Button0.clicked.connect(self.recog)
		self.pain = pline()
		self.setGeometry(200, 200, 200, 200)
		layout = QVBoxLayout()
		layout.addWidget(self.pain)
		layout.addWidget(Button0)
		self.setLayout(layout)
		self.image = np.zeros((paint_width, paint_height, 3), np.uint8)
		self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
		self.mnist_image = np.zeros((mnist_width, mnist_height, 3), np.uint8)
		self.mnist_image = cv2.cvtColor(self.mnist_image, cv2.COLOR_RGB2GRAY)
		self.setWindowTitle('MNIST')
	def recog(self):
		for point in self.pain.points_saved:
			self.image[point.x(), point.y()] = 255
		self.image = cv2.flip(self.image, 0)
		self.image = ndimage.rotate(self.image, 270)
		image_dilation = cv2.dilate(self.image, kernel, iterations = 1)
		cv2.imwrite("dilation.bmp", image_dilation)
		self.mnist_image = cv2.resize(image_dilation, mnist_size)
		cv2.imwrite("mnist.bmp", self.mnist_image)
		ret, thre_image = cv2.threshold(self.mnist_image, 5, 255, cv2.THRESH_BINARY_INV)
		cv2.imwrite("thre_image.bmp", thre_image)
		cv2.imshow('image', thre_image)
		image0 = 1.0 - np.asarray(thre_image, dtype = "float32") / 255
		image0 = image0.reshape((1, 784))
		graph = tf.Graph()
		with graph.as_default():
			with open('trained_graph.pb', 'rb') as f:
				graph_def = tf.GraphDef()
				graph_def.ParseFromString(f.read())
				tf.import_graph_def(graph_def, name = '')
			with tf.Session() as sess:
				result = sess.run('ans:0', feed_dict = {
					'x:0': image0
				})
		msg = QMessageBox()
		msg.setText("ans: " + str(result))
		msg.setWindowTitle("recog")
		msg.setStandardButtons(QMessageBox.Ok)
		retval = msg.exec_()
		self.pain.clear()
		self.image = np.zeros((paint_width, paint_height, 3), np.uint8)
		self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
		self.mnist_image = np.zeros((mnist_width, mnist_height, 3), np.uint8)
		self.mnist_image = cv2.cvtColor(self.mnist_image, cv2.COLOR_RGB2GRAY)

paint_width = 180
paint_height = 180
mnist_width = 28
mnist_height = 28
mnist_size = (mnist_width, mnist_height)
kernel = np.ones((16, 16), np.uint8)

if __name__ == '__main__':
	app = QApplication(sys.argv)
	main_window = MainWindow()
	main_window.show()
	sys.exit(app.exec_())



