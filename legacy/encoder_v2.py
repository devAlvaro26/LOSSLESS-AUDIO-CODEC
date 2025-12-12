import numpy as np
from scipy.io import wavfile
import struct

PATH_AUDIO = "SultansOfSwing_mono.wav"
FRAME_SIZE = 4096   # Tamaño de trama en muestras
ORDER = 12          # Orden del predictor LPC (4-16)

def read_file(path):
    '''
    Lee un archivo de audio y lo devuelve como un array de muestras.
    '''
    try:
        fs, data = wavfile.read(path)
        if data.ndim != 1:
            raise ValueError("Solo se soporta audio mono")
        if data.dtype != np.int16:
            raise ValueError("Solo se soporta audio int16")

        print(f"Frecuencia de muestreo: {fs} Hz")
        print(f"Tipo de datos: {data.dtype}")
        print(f"Tamaño: {str(data.shape)[1:len(str(data.shape))-2]} muestras")

        return fs, data

    except FileNotFoundError:
        print(f"Error: Archivo '{path}' no encontrado")
        return None, None
    except Exception as e:
        print(f"Error al leer archivo: {e}")
        return None, None


def autocorr(x, order):
    '''
    Autocorrelación de orden dado.
    '''
    N = len(x)
    r = np.zeros(order + 1, dtype=np.float64)
    full = np.correlate(x, x, mode='full')
    mid = len(full)//2
    for k in range(order+1):
        r[k] = full[mid + k]
    return r


def levinson_durbin(r, order):
    '''
    Algoritmo Levinson-Durbin.
    '''
    a = np.zeros(order+1, dtype=np.float64)
    a[0] = 1.0
    e = r[0]
    if e == 0:
        return np.zeros(order), 0.0

    for i in range(1, order+1):
        acc = r[i]
        for j in range(1, i):
            acc -= a[j] * r[i-j]
        k = acc / e
        a_prev = a.copy()
        a[i] = k
        for j in range(1, i):
            a[j] = a_prev[j] - k * a_prev[i-j]
        e = e * (1.0 - k*k)
        if e <= 0:
            e = 1e-12

    return a[1:], e


def LPC(frame, order):
    '''
    Calcula coeficientes LPC y valor residual para una trama.
    '''
    frame = frame.astype(np.float64)
    r = autocorr(frame, order)
    coefs, e = levinson_durbin(r, order)
    coefs = coefs.astype(np.float32)
    N = len(frame)

    predicted = np.zeros(N, dtype=np.float64)
    residual = np.zeros(N, dtype=np.int64)
    
    # Las primeras 'order' muestras se guardan crudas
    if N > 0:
        head = min(order, N)
        residual[:head] = np.int64(np.round(frame[:head]))

    for n in range(order, N):
        # 1. Predecir
        prediction_val = np.dot(coefs, frame[n-order:n][::-1])
        
        # 2. REDONDEAR la predicción
        predicted_int = np.round(prediction_val)
        
        # 3. Calcular residuo entero
        residual_val = frame[n] - predicted_int
        residual[n] = np.int64(residual_val)

    return residual, coefs


def zigzag_encode(x):
    '''
    Zigzag encoding para enteros.
    '''
    x = int(x)
    if x >= 0:
        return 2 * x
    else:
        return -2 * x - 1


def optimal_k(residual):
    '''
    Estima el mejor valor de k.
    '''
    if len(residual) == 0:
        return 0
    
    k = 0
    min_bits = np.inf
    
    # Probar k de 0 a 16
    for k_opt in range(17):
        total_bits = 0
        for sample in residual:
            folded = zigzag_encode(sample)
            q = folded >> k_opt
            total_bits += q + 1 + k_opt  # unary + separator + remainder
        
        if total_bits < min_bits:
            min_bits = total_bits
            k = k_opt
    
    return k


def rice_encode(residual, k):
    '''
    Codificación Rice
    '''
    bits = []
    
    for sample in residual:
        # Zigzag encoding para enteros
        folded = zigzag_encode(sample)
        
        # Dividir en quotient y remainder
        q = folded >> k
        r = folded & ((1 << k) - 1)
        
        # Unary para quotient: q ceros seguidos de un uno
        for _ in range(q):
            bits.append(0)
        bits.append(1)
        
        # Binario para remainder (k bits, MSB primero)
        for i in range(k - 1, -1, -1):
            bits.append((r >> i) & 1)
    
    return bits


def bits_to_bytes(bits):
    '''
    Convierte lista de bits a bytes.
    '''
    padding = (8 - (len(bits) % 8)) % 8
    if padding:
        bits.extend([0] * padding)
    
    byte_array = bytearray()
    for i in range(0, len(bits), 8):
        byte_val = 0
        for j in range(8):
            byte_val = (byte_val << 1) | bits[i + j]
        byte_array.append(byte_val)
    
    return bytes(byte_array), padding


def process_frames(data, order, frame_size=FRAME_SIZE):
    '''
    Procesa el audio en tramas, aplicando LPC y codificación Rice.
    '''
    num_samples = len(data)
    num_frames = (num_samples + frame_size - 1) // frame_size
    print(f"\nProcesando {num_frames} tramas de {frame_size} muestras")
    
    frames_data = []
    
    for i in range(num_frames):
        start = i * frame_size
        end = min(start + frame_size, num_samples)
        frame = data[start:end]
        valid_length = len(frame)

        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - valid_length), mode='constant')
        
        # Calcular LPC
        residual, coefs = LPC(frame, order)
        
        # Solo codificar la parte válida
        residual = residual[:valid_length]
        
        # Estimar k óptimo para esta trama (usar el valor estimado)
        k = optimal_k(residual)

        # Codificar con Rice (rice_encode aplica el zigzag internamente)
        bits = rice_encode(residual, k)
        byte_data, padding = bits_to_bytes(bits)
        
        frames_data.append({
            'bytes': byte_data,
            'padding': padding,
            'k': k,
            'coefs': coefs,
            'length': valid_length,
        })
        
        #Animacion de progreso
        print(f"\rTrama {i+1}/{num_frames} procesada", end='', flush=True)
    
    print("\n")
    return frames_data


def save_audio_encoded(filename, frames_data, sample_rate, predictor_order, frame_size):
    '''
    Guarda en formato binario.
    '''
    with open(filename, 'wb') as f:
        # Cabeceras de datos
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<H', predictor_order))
        f.write(struct.pack('<H', frame_size))
        f.write(struct.pack('<I', len(frames_data)))
        
        # Para cada trama: metadata + coeficientes + datos
        for frame in frames_data:
            f.write(struct.pack('<B', frame['k']))
            f.write(struct.pack('<B', frame['padding']))
            f.write(struct.pack('<H', frame['length']))
            f.write(struct.pack('<H', len(frame['bytes'])))
            
            # Coeficientes
            for coef in frame['coefs']:
                f.write(struct.pack('<f', coef))
            
            # Datos comprimidos
            f.write(frame['bytes'])
    
    # Estadísticas
    compressed_size = sum(len(frame['bytes']) for frame in frames_data)
    original_size = sum(frame['length'] for frame in frames_data) * 2
    ratio = (compressed_size / original_size) * 100 if original_size > 0 else 0.0
    print("================= Estadísticas de Compresión ================")
    print(f"Tamaño original: {original_size:,} bytes")
    print(f"Tamaño comprimido: {compressed_size:,} bytes")
    print(f"Ratio: {ratio:.2f}%")


if __name__ == "__main__":
    try:
        fs, data = read_file(PATH_AUDIO)
        if data is not None:
            frames_data = process_frames(data, ORDER, frame_size=FRAME_SIZE)
            save_audio_encoded("encoded_v2.bin", frames_data, fs, ORDER, FRAME_SIZE)
    except Exception as e:
        print(f"Error durante la codificación: {e}")