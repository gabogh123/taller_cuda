from PIL import Image
import numpy as np

def image_to_text(image_path, output_path):
    # Leer la imagen en formato JPEG
    image = Image.open(image_path).convert('L')  # Convertir a escala de grises
    image_array = np.array(image)

    # Convertir la imagen a una representación textual
    height, width = image_array.shape
    with open(output_path, 'w') as file:
        file.write(f"{height} {width}\n")
        for y in range(height):
            for x in range(width):
                file.write(f"{image_array[y, x]} ")
            file.write("\n")

if __name__ == "__main__":
    # Ruta completa de la imagen de entrada y del archivo de salida
    image_path = "imagen5.jpeg"
    output_path = "imagen5.txt"

    # Convertir la imagen a texto
    image_to_text(image_path, output_path)
    print(f"Imagen convertida a texto y guardada en {output_path}")
