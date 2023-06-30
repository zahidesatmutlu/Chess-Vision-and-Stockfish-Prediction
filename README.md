# Chess Vision and Stockfish Prediction

🙌 Best move analysis with [YOLOv8](https://github.com/ultralytics/ultralytics) and [Stockfish](https://github.com/official-stockfish/Stockfish)!

## Overview

💡 Chessboard corners and chess pieces detected with YOLOv8. The position of the chess pieces on the board was converted to digital notation and best move analysis was done with Stockfish. The interface has been designed and usage simplified.

<p align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZzVvemg5aDRiaWgwaXR1bnR4dDJucG81dDNxNWhianpnNDcwd3ZkeCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/bvqmSkwn6lte32lneE/giphy.gif" />
  
Object detection with YOLOv8, best move prediction with Stockfish
</p>

## Train
Datasets are trained with 16GB Tesla T4 GPU via Google Colab.

Chess board corners dataset: https://universe.roboflow.com/nstuchess-iz6hx/corners-hzmj3/dataset/1

Perspective transformed chess pieces dataset: https://universe.roboflow.com/nstuchess-iz6hx/chess-detection-3/dataset/1

<p align="center">
  <img src="https://i.hizliresim.com/jxi31qz.png" />
  Chess board corners detection training results (100 epoch)
</p>

<p align="center">
  <img src="https://i.hizliresim.com/jkd9kbw.png" />
  Perspective transformed chess pieces training results (75 epoch)
</p>



## Installations ⬇️

✔️ A virtual environment is created for the system. (Assuming you have [Anaconda](https://www.anaconda.com/) installed.)

```bash
conda create -n chess_vision python -y
conda activate chess_vision
```

✔️ Clone repo and install [requirements.txt](https://github.com/zahidesatmutlu/yolov5-sahi/blob/master/requirements.txt) in a [Python>=3.7.0](https://www.python.org/downloads/) (3.9 recommended) environment, including [PyTorch>=1.7](https://pytorch.org/get-started/locally/) (1.9.0 recommended).

```bash
git clone https://github.com/zahidesatmutlu/Chess-Vision-and-Stockfish-Prediction  # clone
pip install ultralytics # install
```

✔️ Install [CUDA Toolkit](https://developer.nvidia.com/cuda-11-6-0-download-archive) version 11.6 and install [PyTorch](https://pytorch.org/get-started/previous-versions/) version 1.9.0.

```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

✔️ Copy the test folder containing the images you will detect and your best.pt weight file to the project folder.

```bash
./chess_vision/%here%
```

✔️ The file structure should be like this:

```bash
chess_vision/
    .idea
    __pycache__
    runs
    venv
    best_corners.pt
    best_transformed_detection.pt
    chessboard_detection.py
    chessboard_transformed_with_grid.jpg
    download_dataset.py
    main.py
    stockfish_interface.py
    stockfish_interface.ui
```

## Resources 🤝

🔸 [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

🔸 [https://pypi.org/project/PyQt5](https://pypi.org/project/PyQt5)

🔸 [https://doc.qt.io/qt-6/qtdesigner-manual.html](https://doc.qt.io/qt-6/qtdesigner-manual.html)

🔸 [https://github.com/official-stockfish/Stockfish](https://github.com/official-stockfish/Stockfish)
