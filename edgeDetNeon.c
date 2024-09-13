#include <arm_neon.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

using namespace std;
using namespace std::chrono;

// Función para aplicar filtro Sobel usando Neon ARM
void edgeDetectionNeon(const uint8_t* src, uint8_t* dst, int width, int height) {
    // Definir los kernels de Sobel para X e Y
    int8_t gx[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };  // Sobel X
    int8_t gy[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };  // Sobel Y

    // Recorrer cada píxel de la imagen (excepto los bordes)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x += 8) {  // Procesa 8 píxeles por iteración
            // Cargar 3 filas de píxeles
            uint8x8_t top = vld1_u8(&src[(y - 1) * width + x]);    // Fila superior
            uint8x8_t mid = vld1_u8(&src[y * width + x]);          // Fila del medio
            uint8x8_t bot = vld1_u8(&src[(y + 1) * width + x]);    // Fila inferior

            // Convertir a enteros con signo para aplicar la convolución
            int16x8_t top_s16 = vreinterpretq_s16_u16(vmovl_u8(top));
            int16x8_t mid_s16 = vreinterpretq_s16_u16(vmovl_u8(mid));
            int16x8_t bot_s16 = vreinterpretq_s16_u16(vmovl_u8(bot));

            // Aplicar convoluciones Sobel
            int16x8_t sumX = vmulq_n_s16(top_s16, gx[0][0]);
            sumX = vmlaq_n_s16(sumX, mid_s16, gx[1][0]);
            sumX = vmlaq_n_s16(sumX, bot_s16, gx[2][0]);

            int16x8_t sumY = vmulq_n_s16(top_s16, gy[0][0]);
            sumY = vmlaq_n_s16(sumY, mid_s16, gy[1][0]);
            sumY = vmlaq_n_s16(sumY, bot_s16, gy[2][0]);

            // Combinar las componentes X e Y
            int16x8_t magnitude = vaddq_s16(vabsq_s16(sumX), vabsq_s16(sumY));

            // Saturar a 8 bits y almacenar
            uint8x8_t result = vqmovun_s16(magnitude);
            vst1_u8(&dst[y * width + x], result);  // Almacenar el resultado
        }
    }
}

// Función para leer la imagen desde un archivo de texto
bool readImageFromText(const string& filename, vector<uint8_t>& image, int& width, int& height) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "Error al abrir el archivo de texto" << endl;
        return false;
    }

    file >> height >> width;
    image.resize(width * height);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int pixel;
            file >> pixel;
            image[y * width + x] = static_cast<uint8_t>(pixel);
        }
    }

    file.close();
    return true;
}

// Función para guardar la imagen en un archivo de texto
void saveImageToText(const string& filename, const vector<uint8_t>& image, int width, int height) {
    ofstream file(filename);
    file << height << " " << width << endl;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            file << static_cast<int>(image[y * width + x]) << " ";
        }
        file << endl;
    }
    file.close();
}

int main() {
    // Leer la imagen desde un archivo de texto
    vector<uint8_t> image;
    int width, height;
    if (!readImageFromText("imagen5.txt", image, width, height)) {
        return -1;
    }

    // Imagen de salida
    vector<uint8_t> edgeImage(width * height);

    // Medir el tiempo de ejecución
    auto start = high_resolution_clock::now();
    // Aplicar el filtro Sobel utilizando Neon ARM
    edgeDetectionNeon(image.data(), edgeImage.data(), width, height);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    // Mostrar el tiempo de ejecución
    cout << "Tiempo de ejecución: " << duration.count() << " ms" << endl;

    // Guardar la imagen resultante en un archivo de texto
    saveImageToText("imagen_bordes_neon.txt", edgeImage, width, height);

    return 0;
}

