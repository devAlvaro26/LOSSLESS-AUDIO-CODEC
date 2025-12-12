# Compresor de Audio Lossless (LPC + Rice Coding)

Implementación en Python de un **códec de audio sin pérdidas (lossless)**. El sistema utiliza **Codificación Predictiva Lineal (LPC)** para modelar la señal espectralmente y **Codificación Rice** (Golomb-Rice) con adaptación dinámica de parámetros para comprimir el residuo.

## Descripción del Proyecto

El proyecto implementa un compresor de audio desde cero que consigue ratios de compresión similares a FLAC. El sistema aprovecha el **procesamiento paralelo** (multiprocessing) para acelerar tanto la codificación como la decodificación, utilizando todos los núcleos disponibles de la CPU.

### Flujo de Procesamiento (Encoder)

1.  **Pre-procesamiento (Mid-Side):** En señales estéreo, se transforma de L/R a Mid/Side para mejorar la compresión.
2.  **Tramado (Framing):** Segmentación en tramas (por defecto 4096 muestras).
3.  **Análisis LPC (Levinson-Durbin):** Para cada trama, se calcula la autocorrelación y se utiliza el algoritmo de Levinson-Durbin para encontrar los coeficientes óptimos del filtro predictor (orden configurable, por defecto 12).
4.  **Cálculo del Residuo:** Se predice la señal actual $\hat{x}[n]$ mediante la combinación lineal de muestras pasadas y los coeficientes LPC. La diferencia con la señal real es el residuo:
    $$e[n] = x[n] - \text{round}(\hat{x}[n])$$
5.  **Codificación de Entropía (Rice):**
    - **Zigzag Encoding:** Convierte el residuo (con signo) a enteros positivos para optimizar la codificación ($0 \to 0, -1 \to 1, 1 \to 2...$).
    - **Estimación de K:** Se busca el parámetro $k$ óptimo probando valores en un rango (0-16) y seleccionando aquel que minimiza el tamaño total en bits de la trama codificada.
    - **Rice Coding:** Se genera el bitstream comprimido separando el valor en cociente (unario) y resto (binario).
6.  **Empaquetado:** Se guarda un archivo binario (`.pyflac`) que contiene las cabeceras globales, y para cada trama: su metadata ($k$, padding, longitud), los coeficientes LPC y el bitstream comprimido.

### Flujo de Decodificación (Decoder)

1.  **Lectura de Tramas:** Se extraen los parámetros $k$ y los coeficientes LPC de cada bloque.
2.  **Decodificación Rice y Zigzag:** Se recupera el residuo original $e[n]$.
3.  **Síntesis LPC:** Se reconstruye la señal sumando el residuo a la predicción generada por los coeficientes recuperados:
    $$x[n] = e[n] + \text{round}(\hat{x}[n])$$
4.  **Reconstrucción Estéreo:** Si se utilizó Mid-Side, se recuperan los canales Left/Right originales a partir de Mid/Side.

## Estructura del Repositorio

| Archivo | Descripción |
| :--- | :--- |
| `pyFlac_encoder.py` | **Encoder Principal**. Versión paralelizada que procesa tramas simultáneamente. Genera `encoded.pyflac`. |
| `pyFlac_decoder.py` | **Decoder Principal**. Lee `encoded.pyflac` y reconstruye el audio en `Decoded_Audio.wav`. |
| `compare.py` | Script de utilidad para validar la compresión lossless (bit-exacta). |
| `legacy/` | Carpeta con versiones anteriores del proyecto. |

## Configuración del Algoritmo

El sistema permite ajustar los siguientes parámetros en el código:

- **FRAME_SIZE:** Tamaño de la ventana de análisis (Default: 4096 muestras). Ventanas más grandes pueden mejorar la compresión en señales estables, pero empeorarla en transitorios rápidos.
- **ORDER:** Orden del filtro LPC (Default: 12). Un orden mayor modela mejor la envolvente espectral pero requiere guardar más coeficientes por trama.

## Requisitos

El proyecto utiliza Python 3 y las siguientes librerías:

```bash
pip install numpy scipy
