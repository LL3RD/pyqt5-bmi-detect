from PyQt5 import QtCore, QtGui, QtWidgets, Qt
import datetime
import sys
import qtawesome
from Image_Processor import *  # 检测BMI的文件，不用管
from Visual import *  # 检测BMI的文件，不用管
import cv2


class Thread_show_Pred(QtCore.QThread):  # 多线程
    # show_signal = QtCore.pyqtSignal(QtGui.QImage,float)
    def __init__(self):
        super(Thread_show_Pred, self).__init__()  # 初始化，初始化了P是用来检测BMI的，stop_flag 是用来检测是否需要停止这个进程
        self.stop_flag = True  # 直接停是停不了的，会在后台一直跑
        self.Image = None
        self.P = Image_Processor()

    show_signal = QtCore.pyqtSignal(QtGui.QImage, float)  # 回传函数，参数是需要回传的值的类型
    # show_signal = QtCore.pyqtSignal(QtGui.QPixmap)
    stop_signal = QtCore.pyqtSignal()  # 回传stop信号

    def stop(self):  # 停止线程，不能单单只令stop_flag=0, terminate()是强制终止进程函数
        self.stop_flag = False
        torch.cuda.empty_cache()
        self.terminate()

    def setImage(self, image):
        self.Image = image

    def run(self):  # Run函数
        while (self.stop_flag):
            # time.sleep(0.03)
            show = self.Image
            oldtime = datetime.datetime.now()  # 这个是计算检测一幅图的时间，不用管

            # mask = cv2.cvtColor(show, cv2.COLOR_RGB2GRAY)
            # showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
            '''
            key,mask = self.P._detected(show)
            mask = mask[:, :, None]
            show = np.concatenate((mask, mask, mask), axis=2)
            plt.imshow(show)
            plt.axis('off')
            plt.savefig('image/demo.jpg')
            showImage = QtGui.QPixmap('image/demo.jpg')
            '''

            key, mask, BMI = self.P.Process(show)
            # mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            # print(type(mask))
            if not isinstance(mask, np.ndarray):
                mask = np.zeros((480,640,3))
                BMI  = 0
            else:
                mask = (mask * 255).astype(np.uint8)

            cv2.imwrite('demo.jpg', mask)
            showImage = QtGui.QImage(mask.data, mask.shape[1], mask.shape[0], mask.shape[1],
                                     QtGui.QImage.Format_Indexed8)
            newtime = datetime.datetime.now()
            # print((newtime - oldtime).microseconds)

            self.show_signal.emit(showImage, BMI)  # showImage，BMI是检测的图片和检测出来的值 ， emit就是对应上面回传函数，参数一定要是对应的，不然会报错

        self.stop_signal.emit()  # 回传stop信号


class MainUi(QtWidgets.QMainWindow):  # 主界面
    def __init__(self):
        super().__init__()

        self.timer_camera = QtCore.QTimer()

        self.cap = cv2.VideoCapture()  # 视频流
        self.CAM_NUM = 0  # 哪个摄像头

        self.init_ui()
        self.slot_init()
        self.beauty_ui()
        self.beauty_cursor()

        self.thread = Thread_show_Pred()  # 在这里声明一个线程类


    def init_ui(self):  # 初始化UI，不用管
        self.resize(800, 600)
        self.main_widget = QtWidgets.QWidget()
        self.main_layout = QtWidgets.QGridLayout()
        self.main_widget.setLayout(self.main_layout)

        self.left_widget = QtWidgets.QWidget()
        self.left_widget.setObjectName('left_widget')
        self.left_layout = QtWidgets.QGridLayout()  # 创建左侧部件的网格布局层
        self.left_widget.setLayout(self.left_layout)  # 设置左侧部件布局为网格

        self.right_widget = QtWidgets.QWidget()  # 创建右侧部件
        self.right_widget.setObjectName('right_widget')
        self.right_layout = QtWidgets.QGridLayout()
        self.right_widget.setLayout(self.right_layout)  # 设置右侧部件布局为网格

        self.main_layout.addWidget(self.left_widget, 0, 0, 12, 2)  # 左侧部件在第0行第0列，占8行3列
        self.main_layout.addWidget(self.right_widget, 0, 2, 12, 10)  # 右侧部件在第0行第3列，占8行9列

        self.setCentralWidget(self.main_widget)  # 设置窗口主部件

        self.left_button_1 = QtWidgets.QPushButton(qtawesome.icon('fa.camera', color='white'), '打开相机')
        self.left_button_2 = QtWidgets.QPushButton(qtawesome.icon('fa.male', color='white'), '预测BMI')
        self.left_button_3 = QtWidgets.QPushButton(qtawesome.icon('fa.power-off', color='white'), '退出')

        self.left_layout.addWidget(self.left_button_1, 1, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_2, 2, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_3, 3, 0, 1, 3)

        self.right_label_show_camera = QtWidgets.QLabel()
        self.right_label_show_camera.setFixedSize(641, 481)
        self.right_label_show_detect = QtWidgets.QLabel()
        self.right_label_show_detect.setFixedSize(641, 481)

        # self.right_show_BMI = QtWidgets.QWidget()
        # self.right_show_BMI_layout = QtWidgets.QHBoxLayout()
        # self.right_show_BMI.setLayout(self.right_show_BMI_layout)

        self.right_show_BMI = QtWidgets.QWidget()
        self.right_show_BMI_layout = QtWidgets.QGridLayout()
        self.right_show_BMI.setLayout(self.right_show_BMI_layout)

        self.right_BMI_PRED = QtWidgets.QLabel()
        self.right_BMI_PRED.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignCenter)
        self.right_BMI_PRED.setText("<font size=20 face='Times' color='gray'>BMI Pred :</font>")
        self.right_BMI_TRUE = QtWidgets.QLabel()
        self.right_BMI_TRUE.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignCenter)
        self.right_BMI_TRUE.setText("<font size=20 face='Times' color='gray'>BMI Truth :</font>")
        self.right_BMI_ERROR = QtWidgets.QLabel()
        self.right_BMI_ERROR.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignCenter)
        self.right_BMI_ERROR.setText("<font size=20 face='Times' color='gray'>BMI Error :</font>")

        self.right_LCD = QtWidgets.QLCDNumber()
        self.right_LCD.setStyleSheet("border:none;")
        self.right_LCD.setFixedSize(100, 100)
        self.right_LCD.setSegmentStyle(QtWidgets.QLCDNumber.Filled)
        self.right_LCD.setDigitCount(5)
        self.right_LCD.display(00.00)

        self.right_LCD_TRUE = QtWidgets.QLCDNumber()
        self.right_LCD_TRUE.setStyleSheet("border:none;")
        self.right_LCD_TRUE.setFixedSize(100, 100)
        self.right_LCD_TRUE.setSegmentStyle(QtWidgets.QLCDNumber.Filled)
        self.right_LCD_TRUE.setDigitCount(5)
        self.right_LCD_TRUE.display(00.00)

        self.right_LCD_ERROR = QtWidgets.QLCDNumber()
        self.right_LCD_ERROR.setFixedSize(100,100)
        self.right_LCD_ERROR.setStyleSheet("border:none;")
        self.right_LCD_ERROR.setSegmentStyle(QtWidgets.QLCDNumber.Filled)
        self.right_LCD_ERROR.setDigitCount(5)
        self.right_LCD_ERROR.display(00.00)

        # self.right_show_BMI_layout.addWidget(self.right_BMI_PRED,1,0)
        # self.right_show_BMI_layout.addWidget(self.right_LCD,1,1)
        # self.right_show_BMI_layout.addWidget(self.right_BMI_TRUE,2,0)
        # self.right_show_BMI_layout.addWidget(self.right_LCD_TRUE, 2, 1)
        # self.right_show_BMI_layout.addWidget(self.right_BMI_ERROR,3,0)
        # self.right_show_BMI_layout.addWidget(self.right_LCD_ERROR, 3, 1)
        # self.right_show_BMI_layout.setSpacing(0)

        self.right_layout.addWidget(self.right_label_show_camera, 0, 0,1,3)
        self.right_layout.addWidget(self.right_label_show_detect, 0, 4,1,3)
        self.right_layout.addWidget(self.right_BMI_PRED,1,4)
        self.right_layout.addWidget(self.right_LCD,1,5)
        self.right_layout.addWidget(self.right_BMI_TRUE,1,0)
        self.right_layout.addWidget(self.right_LCD_TRUE, 1, 1)
        self.right_layout.addWidget(self.right_BMI_ERROR,1,2)
        self.right_layout.addWidget(self.right_LCD_ERROR, 1, 3)
        # self.right_layout.setSpacing(10)
        # self.right_layout.addWidget(self.right_show_BMI, 1, 3,)

    def slot_init(self):  # 各个按钮的信号槽连接
        self.left_button_1.clicked.connect(self.button_open_camera_clicked)
        self.timer_camera.timeout.connect(self.show_camera)
        # self.timer_pred.timeout.connect(self.Predict)
        self.left_button_2.clicked.connect(self.button_predict_clicked)
        self.left_button_3.clicked.connect(self.close)

    def button_predict_clicked(self):
        self.setCursor(self.cursor2)
        if self.sender().text() == '预测BMI':
            if self.timer_camera.isActive() == False:
                if self.cap.isOpened():
                    msg = QtWidgets.QMessageBox.warning(self, 'Warning', 'Something Wrong in Detection',
                                                        buttons=QtWidgets.QMessageBox.Ok)
                else:
                    msg = QtWidgets.QMessageBox.warning(self, 'Tip', 'Please Open you camera first!',
                                                        buttons=QtWidgets.QMessageBox.Ok)

            else:
                BMI_Truth, OK = QtWidgets.QInputDialog.getText(self,'输入框','输入自身BMI值')
                self.BMI_Truth = BMI_Truth
                if BMI_Truth and OK:
                    self.right_LCD_TRUE.display(BMI_Truth)

                # self.thread.stop()
                # self.thread = Thread_show_Pred()
                self.thread.Image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)  # 给另一个线程中的函数传图片，直接赋值就行
                self.thread.show_signal.connect(self.call_back)  # 给另一个线程中的回传函数连接槽，call_back()的两个参数就是你要回传的数据
                # # self.timer_pred.start(100)
                self.left_button_2.setText('停止预测')
                self.thread.start()  # 启动线程
        else:
            # self.timer_pred.stop()
            self.thread.stop()
            self.right_label_show_detect.clear()
            self.left_button_2.setText('预测BMI')
            # print(self.thread.isFinished())

        self.setCursor(self.cursor1)

    def call_back(self, showImage, BMI):
        self.right_label_show_detect.setPixmap(QtGui.QPixmap.fromImage(showImage))
        # self.right_label_show_detect.setPixmap(showImage)
        self.right_LCD.display(abs(float(BMI)))
        self.right_LCD_ERROR.display(abs(float(BMI-float(self.BMI_Truth))))

    def button_open_camera_clicked(self):
        self.setCursor(self.cursor2)
        if self.timer_camera.isActive() == False:  # 若定时器未启动
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, 'Warning',
                                                    'Please check if the camera and computer are connected correctly',
                                                    buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(30)  # 打开定时器，启动摄像头，每30ms取一帧显示
                self.left_button_1.setText('关闭相机')
        else:
            self.timer_camera.stop()  # 关闭定时器
            self.cap.release()  # 释放视频流
            self.right_label_show_camera.clear()  # 清空视频显示区域
            self.left_button_1.setText('打开相机')
            self.setCursor(self.cursor1)

    def show_camera(self):
        flag, self.image = self.cap.read()

        self.image = cv2.resize(self.image, (640, 480))
        show = self.image
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        self.thread.setImage(show)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                 QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.right_label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))
        self.setCursor(self.cursor1)

    def beauty_ui(self):  # 下面都是美化界面的，不用管
        self.left_button_1.setFixedSize(120, 80)
        self.left_button_2.setFixedSize(120, 80)
        self.left_button_3.setFixedSize(120, 80)

        self.left_button_3.setStyleSheet(
            '''
                QPushButton{background:#424242 ;border-radius:25px;}
                QPushButton:hover{background:red;}
                '''
        )
        self.left_button_2.setStyleSheet(
            '''
                QPushButton{background:#424242 ;border-radius:25px;}
                QPushButton:hover{background:#F7DC6F;}
            '''
        )
        self.left_button_1.setStyleSheet(
            '''
                QPushButton{background:#424242 ;border-radius:25px;}
                QPushButton:hover{background:green;}
            '''
        )

        self.left_widget.setStyleSheet(  ##424242
            ''' 
                QWidget#left_widget{
                    color:#232C51;
                    background:#424242;
                    border-top:1px solid darkGray;
                    border-bottom:1px solid darkGray;
                    border-left:1px solid darkGray;
                    border-top-left-radius:10px;
                    border-bottom-left-radius:10px;
                }
                QPushButton{border:none;color:white;}
                QPushButton#left_label{
                    border:none;
                    border-bottom:1px solid white;
                    font-size:18px;
                    font-weight:700;
                    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
                }
                QPushButton#left_button:hover{border-left:4px solid red;font-weight:700;}
            '''
        )

        self.right_widget.setStyleSheet(
            '''
                QWidget#right_widget{
                    color:#232C51;
                    background:white;
                    border-top:1px solid darkGray;
                    border-bottom:1px solid darkGray;
                    border-right:1px solid darkGray;
                    border-top-right-radius:10px;
                    border-bottom-right-radius:10px;
                }
                QLabel#right_lable{
                    border:none;
                    font-size:16px;
                    font-weight:700;
                    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
                }
            '''
        )

        self.setWindowOpacity(0.95)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)

        self.left_widget.setStyleSheet(
            '''
                QWidget#left_widget{
                    background:gray;
                    border-top:1px solid white;
                    border-bottom:1px solid white;
                    border-left:1px solid white;
                    border-top-left-radius:10px;
                    border-bottom-left-radius:10px;
                 }
            '''

        )

        self.main_layout.setSpacing(0)

    def Predict(self):

        show = self.image
        k, m = self.P._detected(show)
        shape2d = (m.shape[0], m.shape[1])
        img_f = np.zeros(shape2d + (3,), dtype="float32")
        img_f[:, :, :3] = [255, 255, 255]
        v = Visual(img_f)
        v.draw_keypoints_predictions(k)
        v.draw_masks_predictions(m)
        showImage = QtGui.QImage(v.image.data, v.image.shape[1], v.image.shape[0], QtGui.QImage.Format_RGB888)
        self.right_label_show_detect.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def beauty_cursor(self):
        pixmap1 = Qt.QPixmap('image/Cursor1.png')
        # pixmap = pixmap.scaled(30,30)
        self.cursor1 = Qt.QCursor(pixmap1, 0, 0)
        pixmap2 = Qt.QPixmap('image/Cursor3.png')
        # pixmap = pixmap.scaled(30,30)
        self.cursor2 = Qt.QCursor(pixmap2, 0, 0)
        self.setCursor(self.cursor1)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
            event.accept()
            # self.setCursor(self.cursor2)  # 更改鼠标图标

    def mouseMoveEvent(self, QMouseEvent):
        if QtCore.Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)  # 更改窗口位置
            QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
        # self.setCursor(self.cursor1)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    gui = MainUi()
    gui.show()
    sys.exit(app.exec_())
