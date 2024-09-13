#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

using namespace std;
using namespace std::chrono;

// Función para aplicar el filtro de detección de bordes de forma serial
void edgeDetectionSerial(const vector<vector<int>>& src, vector<vector<int>>& dst, int width, int height) {
    // Definir los kernels de Sobel para X e Y
    int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}}; // Kernel Sobel en X
    int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}}; // Kernel Sobel en Y

    // Crear una imagen de salida con ceros (negro)
    dst = vector<vector<int>>(height, vector<int>(width, 0));

    // Recorrer cada píxel de la imagen (excepto los bordes)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int sumX = 0, sumY = 0;
            // Aplicar los kernels de Sobel
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    sumX += src[y + i][x + j] * gx[i + 1][j + 1];
                    sumY += src[y + i][x + j] * gy[i + 1][j + 1];
                }
            }
            // Calcular la magnitud del gradiente
            int sum = abs(sumX) + abs(sumY);
            // Asignar el valor al píxel de salida (clamp a 255)
            dst[y][x] = sum > 255 ? 255 : sum;
        }
    }
}

// Función para leer la imagen desde un archivo de texto
bool readImageFromText(const string& filename, vector<vector<int>>& image, int& width, int& height) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "Error al abrir el archivo de texto" << endl;
        return false;
    }

    file >> height >> width;
    image = vector<vector<int>>(height, vector<int>(width));

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            file >> image[y][x];
        }
    }

    file.close();
    return true;
}

// Función para guardar la imagen en un archivo de texto
void saveImageToText(const string& filename, const vector<vector<int>>& image, int width, int height) {
    ofstream file(filename);
    file << height << " " << width << endl;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            file << image[y][x] << " ";
        }
        file << endl;
    }
    file.close();
}

int main() {
    // Leer la imagen desde un archivo de texto
    vector<vector<int>> image;
    int width, height;
    if (!readImageFromText("imagen.txt", image, width, height)) {
        return -1;
    }

    // Imagen de salida
    vector<vector<int>> edgeImage;

    // Medir el tiempo de ejecución
    auto start = high_resolution_clock::now();
    edgeDetectionSerial(image, edgeImage, width, height);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    // Mostrar el tiempo de ejecución
    cout << "Tiempo de ejecución: " << duration.count() << " ms" << endl;

    // Guardar la imagen resultante en un archivo de texto
    saveImageToText("imagen_bordes.txt", edgeImage, width, height);

    return 0;
}
