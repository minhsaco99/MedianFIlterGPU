import numpy as np
from numba import cuda
import numba
import math
import time
import cv2
import matplotlib.pyplot as plt

iter_ = 10
# median filter params
win_size = 11
win_size_flat = win_size * win_size
rn = math.floor(win_size/2)
med = math.ceil(win_size_flat/2)
shape = 2000
image_shape = (shape, shape)
image_path = '/home/minh/Downloads/1.jpeg'

# matrix mul params
row = col = 5000


@cuda.jit
def matrix_mul_gpu(A, B, C):
    """Perform matrix multiplication of C = A * B
    """
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp

@cuda.jit
def median_filter_gpu(img, out_img):
    row, col = cuda.grid(2)

    windows = cuda.local.array(win_size_flat, numba.int32)

    if row < out_img.shape[0] and col < out_img.shape[1]:
        count = 0
        for i in range(row-rn, row+rn+1):
            for j in range(col-rn, col+rn+1):
                if i >= 0 and j >= 0 and i < out_img.shape[0] and j < out_img.shape[1]:
                    windows[count] = img[i, j]
                else:
                    windows[count] = 0
                count += 1
        
        for i in range(win_size_flat):
            for j in range(win_size_flat):
                if windows[i] > windows[j]:
                    windows[i], windows[j] = windows[j], windows[i]

        out_img[row, col] = windows[med]

def test_matmul(row, col):
    host_data_1 = np.random.rand(row, col)
    host_data_2 = np.random.rand(row, col)
    
    device_data_1 = cuda.to_device(host_data_1)
    device_data_2 = cuda.to_device(host_data_2)
    
    out_device_data = cuda.device_array((row, col))
    time_cpu = []
    time_gpu = []
    for i in range(iter_):
        start = time.time()
        out = np.matmul(host_data_1, host_data_2)
        end = time.time()
        print(f'====CPU time:{end - start} with shape {row, col}')
        if i != 0:
            time_cpu.append(end - start)

        threadsperblock = (32, 32)
        blockspergrid_x = int(math.ceil(out_device_data.shape[0] / threadsperblock[0]))
        blockspergrid_y = int(math.ceil(out_device_data.shape[1] / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        start = time.time()
        matrix_mul_gpu[blockspergrid, threadsperblock](device_data_1, device_data_2, out_device_data)
        end = time.time()
        print(f'====GPU time:{end -start} with shape {row, col}')
        
        if i != 0:
            time_gpu.append(end - start)

        out_gpu = out_device_data.copy_to_host()
        print(f'====Error with cpu compute:{np.sum(out - out_gpu)}')
        print()
        del out_gpu

    print(f"=====Average time CPU:{sum(time_cpu)/(iter_-1)}")
    print(f"=====Average time GPU:{sum(time_gpu)/(iter_-1)}")

def test_median_filter(image, image_shape):
    image = cv2.resize(image, image_shape)
    device_img = cuda.to_device(image)
    out_device_img = cuda.device_array(image.shape)

    time_cpu = []
    time_gpu = []
    for i in range(iter_):
        start = time.time()
        cpu_med = cv2.medianBlur(image, win_size)
        end = time.time()
        print(f'====CPU time:{end - start} with image shape {image_shape}')
        if i != 0:
            time_cpu.append(end - start)
            
        threadsperblock = (32, 32)
        blockspergrid_x = int(math.ceil(image.shape[0] / threadsperblock[0]))
        blockspergrid_y = int(math.ceil(image.shape[1] / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        start = time.time()
        median_filter_gpu[blockspergrid, threadsperblock](device_img,
                                    out_device_img)
        end = time.time()
        print(f'====GPU time:{end - start} with image shape {image_shape}')
        
        if i != 0:
            time_gpu.append(end - start)

        out_gpu_img = out_device_img.copy_to_host()


        # out = np.concatenate((image, cpu_med, out_gpu_img), axis=1)
        # out = cv2.resize(out, (1000, 700))
        # cv2.imshow("cpu", out/255.0)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # del out_gpu_img
    plt.figure(figsize=(9, 3))
    
    ax1 = plt.subplot(131)
    plt.imshow(image, cmap='gray')
    ax1.title.set_text('Original image')

    ax2 = plt.subplot(132)
    plt.imshow(cpu_med, cmap='gray')
    ax2.title.set_text('CPU med')

    ax3 = plt.subplot(133)
    plt.imshow(out_gpu_img.astype(int), cmap='gray')
    ax3.title.set_text('GPU med')

    plt.suptitle(f'Median filter image size {image_shape}, kernel size {win_size}')
    plt.show()
    print(f"=====Average time CPU:{sum(time_cpu)/(iter_-1)}")
    print(f"=====Average time GPU:{sum(time_gpu)/(iter_-1)}")
    print()

if __name__ == '__main__':
    # print('******Test matrix mul******')
    # test_matmul(row, col)
    # print()
    print('******Test median filter******')
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    test_median_filter(image, image_shape)
    