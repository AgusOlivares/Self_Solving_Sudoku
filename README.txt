# Sudoku Solver with Image Processing and Machine Learning

## Descripción del Proyecto

Este proyecto implementa un solver de Sudoku que utiliza técnicas de procesamiento de imágenes y aprendizaje automático (Machine Learning) para resolver Sudokus dibujados a mano en una imagen capturada de la pantalla. El solver toma una captura de pantalla del monitor que contiene el Sudoku, procesa la imagen para identificar las casillas y los números escritos en ellas, y luego utiliza un modelo de clasificación basado en K-Nearest Neighbors (KNN) para resolver el Sudoku.

## Requisitos

- Python 3.x
- Bibliotecas: OpenCV (cv2), PyAutoGUI, NumPy, pandas, scikit-image
- pagina web: https://sudoku.com/ (En pantalla completa)

## Cómo Funciona

1. Captura de Pantalla:
   - El programa utiliza la biblioteca PyAutoGUI para tomar una captura de pantalla de todo el monitor.

2. Procesamiento de Imágenes:
   - La captura de pantalla se convierte a escala de grises y se aplica un desenfoque gaussiano para reducir el ruido.
   - Luego, se utiliza un umbral adaptativo para convertir la imagen en binaria, resaltando las casillas del Sudoku.

3. Detección de Contornos y Recorte de la Rejilla:
   - Se encuentran los contornos en la imagen binaria para identificar las casillas del Sudoku.
   - La mayor área de contorno detectada se asume como la rejilla del Sudoku.
   - La rejilla se recorta y se divide en casillas individuales.

4. Segmentación de Imágenes:
   - Cada casilla se segmenta y se almacena en una lista para su posterior clasificación.

5. Clasificación mediante KNN:
   - Se utiliza un modelo KNN previamente entrenado para clasificar cada casilla en un número (0 para casillas vacías).
   - El modelo KNN se entrena con un conjunto de datos que contiene imágenes de números escritos a mano.

6. Resolución del Sudoku:
   - Los resultados clasificados se organizan en una matriz 9x9 para representar el Sudoku.
   - Si una casilla está vacía, se identifica con el número 0.

## Funciones Principales

- `preprocess(screenshot)`: Realiza el preprocesamiento de la captura de pantalla para destacar las casillas del Sudoku.
- `find_sudoku_contour(preprocessed)`: Encuentra los contornos en la imagen binaria y detecta la rejilla del Sudoku.
- `crop_grid(screenshot, square)`: Recorta la rejilla del Sudoku de la captura de pantalla.
- `split_grid(cropped_grid)`: Divide la rejilla recortada en casillas individuales y las almacena en una lista.
- `squares_images_to_sudoku(squares_images)`: Clasifica las casillas mediante el modelo KNN y devuelve la matriz del Sudoku resuelto.
- `is_empty(image)`: Detecta si una casilla está vacía o no mediante la detección de contornos en la imagen binaria.

## Uso del Programa

1. Ejecuta el programa y espera unos segundos para que tome la captura de pantalla.
2. Verás la imagen de la rejilla del Sudoku con las casillas resaltadas en una ventana emergente.
3. La matriz del Sudoku resuelto se mostrará en la consola.
4. (Opcional) Puedes habilitar la resolución completa del Sudoku utilizando un algoritmo de solución de Sudoku (no implementado en este código).

## Notas

- El modelo KNN utilizado para la clasificación se entrena previamente con un conjunto de datos que contiene imágenes de números escritos a mano. Asegúrate de que el archivo "dataset.csv" contenga los datos correctamente etiquetados para el entrenamiento del modelo.
- La detección de espacios vacíos en las casillas puede requerir ajustes en el umbral y el enfoque de procesamiento de imágenes según las características específicas de tus imágenes de Sudoku.

## Créditos

Este proyecto fue desarrollado por [Tu Nombre] y está basado en técnicas y algoritmos de procesamiento de imágenes y aprendizaje automático.

