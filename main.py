import os
import cv2
import pyautogui
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import skimage 
import shutil
from SudokuSolver import SudokuSolver

def main():

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    pyautogui.hotkey("alt", "tab", interval=0.1)
    screenshot = pyautogui.screenshot()
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    preprocessed = preprocess(screenshot)
    square_contour = find_sudoku_contour(preprocessed)
    cropped_grid = crop_grid(screenshot, square_contour)

    squares_images = split_grid(cropped_grid)

    sudoku = squares_images_to_sudoku(squares_images)
    print(sudoku)
    solver = SudokuSolver(sudoku)
    solved = solver.solve()
    print("")
    print(solved)
    print("Tardo: ", solver.solve_time, " segundos")


    #cv2.imshow("Sudoku", cropped_grid)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    solve_on_website(square_contour, solved)
    

# Image Processing
def preprocess(screenshot):
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return thresh

def find_sudoku_contour(preprocessed):

    contour, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    for cnt in contour:
        if is_square(cnt):
            squares.append(cnt)
    squares = sorted(squares, key=cv2.contourArea, reverse=True)
    return squares[0]


def is_square(contour):
    aprox = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    _, _, w, h = cv2.boundingRect(aprox)
    aspectRatio = w/float(h)
    return len(aprox) == 4 and abs(aspectRatio - 1) < 0.1

def crop_grid(screenshot, square):
    x, y, w, h = cv2.boundingRect(square)
    cropped = screenshot[y:y+h , x:x+w]

    # print("Imagen recortada:", cropped.shape)

    return cropped

# Image splitting

def split_grid(cropped_grid):
    img = preprocess(cropped_grid)
    img = skimage.segmentation.clear_border(img)
    img = 255 - img
    height, _ = img.shape
    square_size = height // 9
    squares = []

    # Verificar si la carpeta "squares" existe, si no, se crea.
    if not os.path.exists("squares"):
        os.makedirs("squares")

    for i in range(9):
        for j in range(9):
            squares.append(img[i*square_size:(i+1)*square_size, j*square_size:(j+1)*square_size])

            # print(f"Guardando square_{i}_{j}.png")

            cv2.imwrite(f"squares/square_{i}_{j}.png", squares[-1])
    return squares

# Machine Learning model

def squares_images_to_sudoku(squares_images):
    knn = create_knn_model()
    sudoku = np.zeros((81), dtype=int)

    for i, image in enumerate(squares_images):

        if is_empty(image):
            sudoku[i] = 0
        else:
            sudoku[i] = predict_digit(image, knn)
    
    return sudoku.reshape(9, 9)

## Agregado
def is_empty(image):
    # Convertir la imagen en binaria utilizando un umbral adaptativo.
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Buscar contornos en la imagen binaria.
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Si no se encontraron contornos, consideramos la casilla como vacía.
    return len(contours) == 0

##



def predict_digit(img, knn):
    img = img.reshape(1, -1)
    return knn.predict(img)[0]

def create_knn_model():
    df = pd.read_csv("dataset.csv")

    # Separamos las rutas de las imágenes y las etiquetas en columnas separadas
    image_paths = df["image_path"]
    labels = df["label"]

    # Convertimos las rutas de las imágenes en matrices numéricas para el entrenamiento
    images = []
    for image_path in image_paths:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = img.flatten()  # Convertimos la imagen en un vector 1D
        images.append(img)

    # Convertimos la lista de matrices numéricas en un array numpy
    x = np.array(images)
    y = labels.values

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(x, y)
    return knn


# -- Sending the solution to the website
def solve_on_website(square_contour, solved):
    x, y, w, h = cv2.boundingRect(square_contour)
    square_size = h // 9
    for i in range(9):
        for j in range(9):
            pyautogui.click(x + j*square_size + square_size//2, y + i*square_size + square_size//2, _pause=False)
            
            if solved[i, j] == 7:
                pyautogui.keyDown('ctrl')
                pyautogui.press('7', _pause=False)
                pyautogui.keyUp('ctrl')
            else:
                pyautogui.keyDown('shift')
                pyautogui.press(str(solved[i, j]), _pause=False)
                pyautogui.keyUp('shift')
            
            


if __name__ == "__main__":
    main()
    if os.path.exists("squares"):
        shutil.rmtree("squares")