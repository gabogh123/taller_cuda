#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

// Kernel CUDA para aplicar el filtro de detección de bordes
__global__ void edgeDetectionCUDA(const unsigned char *src, unsigned char *dst, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Definir los kernels de Sobel para X e Y
    int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    // Asegurarse de que el hilo esté dentro de los límites de la imagen
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int sumX = 0, sumY = 0;
        // Aplicar los kernels de Sobel
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int pixel = src[(y + i) * width + (x + j)];
                sumX += pixel * gx[i + 1][j + 1];
                sumY += pixel * gy[i + 1][j + 1];
            }
        }
        // Calcular la magnitud del gradiente
        int sum = abs(sumX) + abs(sumY);
        // Asignar el valor al píxel de salida (clamp a 255)
        dst[y * width + x] = (sum > 255) ? 255 : sum;
    }
}

// Función envolvente para llamar al kernel CUDA
void edgeDetectionCUDAWrapper(unsigned char *src, unsigned char *dst, int width, int height) {
    unsigned char *d_src, *d_dst;
    size_t size = width * height * sizeof(unsigned char);

    // Reservar memoria en el dispositivo (GPU)
    cudaError_t err;
    err = cudaMalloc(&d_src, size);
    if (err != cudaSuccess) {
        cerr << "Error al reservar memoria en el dispositivo: " << cudaGetErrorString(err) << endl;
        return;
    }

    err = cudaMalloc(&d_dst, size);
    if (err != cudaSuccess) {
        cerr << "Error al reservar memoria en el dispositivo: " << cudaGetErrorString(err) << endl;
        cudaFree(d_src);
        return;
    }

    // Copiar datos de la imagen desde el host (CPU) al dispositivo (GPU)
    err = cudaMemcpy(d_src, src, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cerr << "Error al copiar datos al dispositivo: " << cudaGetErrorString(err) << endl;
        cudaFree(d_src);
        cudaFree(d_dst);
        return;
    }

    // Definir la configuración de los bloques e hilos
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Llamar al kernel CUDA
    edgeDetectionCUDA<<<numBlocks, threadsPerBlock>>>(d_src, d_dst, width, height);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "Error en el kernel: " << cudaGetErrorString(err) << endl;
    }

    // Copiar los resultados desde el dispositivo (GPU) al host (CPU)
    err = cudaMemcpy(dst, d_dst, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cerr << "Error al copiar datos del dispositivo: " << cudaGetErrorString(err) << endl;
    }

    // Liberar la memoria en el dispositivo (GPU)
    cudaFree(d_src);
    cudaFree(d_dst);
}

// Función para leer la imagen desde un archivo de texto
bool readImageFromText(const string& filename, vector<unsigned char>& image, int& width, int& height) {
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
            image[y * width + x] = static_cast<unsigned char>(pixel);
        }
    }

    file.close();
    return true;
}

// Función para guardar la imagen en un archivo de texto
void saveImageToText(const string& filename, const vector<unsigned char>& image, int width, int height) {
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
    vector<unsigned char> image;
    int width, height;
    if (!readImageFromText("imagen5.txt", image, width, height)) {
        return -1;
    }

    // Imagen de salida
    vector<unsigned char> edgeImage(width * height);

    // Medir el tiempo de ejecución
    auto start = high_resolution_clock::now();
    edgeDetectionCUDAWrapper(image.data(), edgeImage.data(), width, height);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    // Mostrar el tiempo de ejecución
    cout << "Tiempo de ejecución: " << duration.count() << " ms" << endl;

    // Guardar la imagen resultante en un archivo de texto
    saveImageToText("imagen_bordes_cuda.txt", edgeImage, width, height);

    return 0;
}

