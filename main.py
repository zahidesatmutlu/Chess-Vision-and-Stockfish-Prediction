from chessboard_detection import *
from PyQt5.QtWidgets import *
from chessboard_detection import *
from stockfish_interface import Ui_MainWindow
from stockfish import Stockfish

import chess
import chess.engine
import sys
import cv2


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.stockfishUi = Ui_MainWindow()
        self.stockfishUi.setupUi(self)

        self.stockfishUi.Browse.clicked.connect(self.browse_image)
        self.stockfishUi.Quit.clicked.connect(self.quit)
        self.stockfishUi.Predict.clicked.connect(self.predict_best_move)

    def browse_image(self):
        file_dialog = QFileDialog()
        file_paths, _ = file_dialog.getOpenFileNames(self, 'Open files', 'Corner-2-3/valid/images/',
                                                     'Images (*.jpg)')
        if file_paths:
            self.stockfishUi.ImageBrowser.setText(", ".join(file_paths))

    def quit(self):
        QApplication.quit()

    def predict_best_move(self):
        image_path = self.stockfishUi.ImageBrowser.text()

        # image = 'Corner-2-3/valid/images/IMG_1120_jpeg.rf.99f4ab3d304106260c5e32ec88e98aad.jpg'

        corners = detect_corners(image_path)

        transformed_image = four_point_transform(image_path, corners)

        ptsT, ptsL = plot_grid_on_transformed_image(transformed_image)

        detections, boxes = chess_pieces_detector(transformed_image)

        # calculate the grid
        xA = ptsT[0][0]
        xB = ptsT[1][0]
        xC = ptsT[2][0]
        xD = ptsT[3][0]
        xE = ptsT[4][0]
        xF = ptsT[5][0]
        xG = ptsT[6][0]
        xH = ptsT[7][0]
        xI = ptsT[8][0]

        y9 = ptsL[0][1]
        y8 = ptsL[1][1]
        y7 = ptsL[2][1]
        y6 = ptsL[3][1]
        y5 = ptsL[4][1]
        y4 = ptsL[5][1]
        y3 = ptsL[6][1]
        y2 = ptsL[7][1]
        y1 = ptsL[8][1]

        # calculate all the squares
        a8 = np.array([[xA, y9], [xB, y9], [xB, y8], [xA, y8]])
        a7 = np.array([[xA, y8], [xB, y8], [xB, y7], [xA, y7]])
        a6 = np.array([[xA, y7], [xB, y7], [xB, y6], [xA, y6]])
        a5 = np.array([[xA, y6], [xB, y6], [xB, y5], [xA, y5]])
        a4 = np.array([[xA, y5], [xB, y5], [xB, y4], [xA, y4]])
        a3 = np.array([[xA, y4], [xB, y4], [xB, y3], [xA, y3]])
        a2 = np.array([[xA, y3], [xB, y3], [xB, y2], [xA, y2]])
        a1 = np.array([[xA, y2], [xB, y2], [xB, y1], [xA, y1]])

        b8 = np.array([[xB, y9], [xC, y9], [xC, y8], [xB, y8]])
        b7 = np.array([[xB, y8], [xC, y8], [xC, y7], [xB, y7]])
        b6 = np.array([[xB, y7], [xC, y7], [xC, y6], [xB, y6]])
        b5 = np.array([[xB, y6], [xC, y6], [xC, y5], [xB, y5]])
        b4 = np.array([[xB, y5], [xC, y5], [xC, y4], [xB, y4]])
        b3 = np.array([[xB, y4], [xC, y4], [xC, y3], [xB, y3]])
        b2 = np.array([[xB, y3], [xC, y3], [xC, y2], [xB, y2]])
        b1 = np.array([[xB, y2], [xC, y2], [xC, y1], [xB, y1]])

        c8 = np.array([[xC, y9], [xD, y9], [xD, y8], [xC, y8]])
        c7 = np.array([[xC, y8], [xD, y8], [xD, y7], [xC, y7]])
        c6 = np.array([[xC, y7], [xD, y7], [xD, y6], [xC, y6]])
        c5 = np.array([[xC, y6], [xD, y6], [xD, y5], [xC, y5]])
        c4 = np.array([[xC, y5], [xD, y5], [xD, y4], [xC, y4]])
        c3 = np.array([[xC, y4], [xD, y4], [xD, y3], [xC, y3]])
        c2 = np.array([[xC, y3], [xD, y3], [xD, y2], [xC, y2]])
        c1 = np.array([[xC, y2], [xD, y2], [xD, y1], [xC, y1]])

        d8 = np.array([[xD, y9], [xE, y9], [xE, y8], [xD, y8]])
        d7 = np.array([[xD, y8], [xE, y8], [xE, y7], [xD, y7]])
        d6 = np.array([[xD, y7], [xE, y7], [xE, y6], [xD, y6]])
        d5 = np.array([[xD, y6], [xE, y6], [xE, y5], [xD, y5]])
        d4 = np.array([[xD, y5], [xE, y5], [xE, y4], [xD, y4]])
        d3 = np.array([[xD, y4], [xE, y4], [xE, y3], [xD, y3]])
        d2 = np.array([[xD, y3], [xE, y3], [xE, y2], [xD, y2]])
        d1 = np.array([[xD, y2], [xE, y2], [xE, y1], [xD, y1]])

        e8 = np.array([[xE, y9], [xF, y9], [xF, y8], [xE, y8]])
        e7 = np.array([[xE, y8], [xF, y8], [xF, y7], [xE, y7]])
        e6 = np.array([[xE, y7], [xF, y7], [xF, y6], [xE, y6]])
        e5 = np.array([[xE, y6], [xF, y6], [xF, y5], [xE, y5]])
        e4 = np.array([[xE, y5], [xF, y5], [xF, y4], [xE, y4]])
        e3 = np.array([[xE, y4], [xF, y4], [xF, y3], [xE, y3]])
        e2 = np.array([[xE, y3], [xF, y3], [xF, y2], [xE, y2]])
        e1 = np.array([[xE, y2], [xF, y2], [xF, y1], [xE, y1]])

        f8 = np.array([[xF, y9], [xG, y9], [xG, y8], [xF, y8]])
        f7 = np.array([[xF, y8], [xG, y8], [xG, y7], [xF, y7]])
        f6 = np.array([[xF, y7], [xG, y7], [xG, y6], [xF, y6]])
        f5 = np.array([[xF, y6], [xG, y6], [xG, y5], [xF, y5]])
        f4 = np.array([[xF, y5], [xG, y5], [xG, y4], [xF, y4]])
        f3 = np.array([[xF, y4], [xG, y4], [xG, y3], [xF, y3]])
        f2 = np.array([[xF, y3], [xG, y3], [xG, y2], [xF, y2]])
        f1 = np.array([[xF, y2], [xG, y2], [xG, y1], [xF, y1]])

        g8 = np.array([[xG, y9], [xH, y9], [xH, y8], [xG, y8]])
        g7 = np.array([[xG, y8], [xH, y8], [xH, y7], [xG, y7]])
        g6 = np.array([[xG, y7], [xH, y7], [xH, y6], [xG, y6]])
        g5 = np.array([[xG, y6], [xH, y6], [xH, y5], [xG, y5]])
        g4 = np.array([[xG, y5], [xH, y5], [xH, y4], [xG, y4]])
        g3 = np.array([[xG, y4], [xH, y4], [xH, y3], [xG, y3]])
        g2 = np.array([[xG, y3], [xH, y3], [xH, y2], [xG, y2]])
        g1 = np.array([[xG, y2], [xH, y2], [xH, y1], [xG, y1]])

        h8 = np.array([[xH, y9], [xI, y9], [xI, y8], [xH, y8]])
        h7 = np.array([[xH, y8], [xI, y8], [xI, y7], [xH, y7]])
        h6 = np.array([[xH, y7], [xI, y7], [xI, y6], [xH, y6]])
        h5 = np.array([[xH, y6], [xI, y6], [xI, y5], [xH, y5]])
        h4 = np.array([[xH, y5], [xI, y5], [xI, y4], [xH, y4]])
        h3 = np.array([[xH, y4], [xI, y4], [xI, y3], [xH, y3]])
        h2 = np.array([[xH, y3], [xI, y3], [xI, y2], [xH, y2]])
        h1 = np.array([[xH, y2], [xI, y2], [xI, y1], [xH, y1]])

        # transforms the squares to write FEN
        FEN_annotation = [[a8, b8, c8, d8, e8, f8, g8, h8],
                          [a7, b7, c7, d7, e7, f7, g7, h7],
                          [a6, b6, c6, d6, e6, f6, g6, h6],
                          [a5, b5, c5, d5, e5, f5, g5, h5],
                          [a4, b4, c4, d4, e4, f4, g4, h4],
                          [a3, b3, c3, d3, e3, f3, g3, h3],
                          [a2, b2, c2, d2, e2, f2, g2, h2],
                          [a1, b1, c1, d1, e1, f1, g1, h1]]

        board_FEN = []
        corrected_FEN = []
        complete_board_FEN = []

        for line in FEN_annotation:
            line_to_FEN = []
            for square in line:
                piece_on_square = connect_square_to_detection(detections, square, boxes)
                line_to_FEN.append(piece_on_square)
            corrected_FEN = [i.replace('empty', '1') for i in line_to_FEN]
            print(corrected_FEN)
            board_FEN.append(corrected_FEN)

        complete_board_FEN = [''.join(line) for line in board_FEN]

        to_FEN = '/'.join(complete_board_FEN)

        stockfish = Stockfish('C:/Users/Administrator/Downloads/stockfish_15.1_win_x64_avx2/stockfish_15.1_win_x64_avx2'
                              '/stockfish-windows-2022-x86-64-avx2.exe')
        stockfish.set_fen_position(to_FEN)
        print(to_FEN)
        print(stockfish.get_parameters())
        print(stockfish.get_best_move())
        print(stockfish.get_board_visual())
        self.stockfishUi.Best_Move.setText("Best Move: " + stockfish.get_best_move())


if __name__ == '__main__':
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    window = Main()
    window.show()
    sys.exit(app.exec_())
