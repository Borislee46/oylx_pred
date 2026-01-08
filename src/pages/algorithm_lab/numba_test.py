import time
import numpy as np
import pandas as pd
import streamlit as st
from numba import njit, prange, vectorize, float64

def python_pi(n):
    """纯 Python 实现蒙特卡罗估算 Pi"""
    acc = 0
    for i in range(n):
        x = np.random.random()
        y = np.random.random()
        if (x**2 + y**2) < 1.0:
            acc += 1
    return 4.0 * acc / n

@njit
def numba_pi(n):
    """Numba JIT 实现"""
    acc = 0
    for i in range(n):
        x = np.random.random()
        y = np.random.random()
        if (x**2 + y**2) < 1.0:
            acc += 1
    return 4.0 * acc / n

@njit(parallel=True)
def numba_pi_parallel(n):
    """Numba 并行实现"""
    acc = 0
    for i in prange(n):
        x = np.random.random()
        y = np.random.random()
        if (x**2 + y**2) < 1.0:
            acc += 1
    return 4.0 * acc / n

def mandelbrot_python(h, w, max_iter):
    """NumPy 向量化实现曼德博集合"""
    y, x = np.ogrid[-1.4:1.4:h*1j, -2:0.8:w*1j]
    c = x + y*1j
    z = c
    divtime = max_iter + np.zeros(z.shape, dtype=int)
    for i in range(max_iter):
        z = z**2 + c
        diverge = z*np.conj(z) > 2**2
        div_now = diverge & (divtime == max_iter)
        divtime[div_now] = i
        z[diverge] = 2
    return divtime

@njit
def mandelbrot_numba(h, w, max_iter):
    """Numba JIT 实现曼德博集合"""
    image = np.zeros((h, w), dtype=np.int32)
    for i in range(h):
        for j in range(w):
            c = complex(-2.0 + j * 2.8 / w, -1.4 + i * 2.8 / h)
            z = 0.0j
            for k in range(max_iter):
                z = z * z + c
                if (z.real * z.real + z.imag * z.imag) >= 4.0:
                    image[i, j] = k
                    break
            else:
                image[i, j] = max_iter
    return image

@njit(parallel=True)
def mandelbrot_numba_parallel(h, w, max_iter):
    """Numba 并行实现曼德博集合"""
    image = np.zeros((h, w), dtype=np.int32)
    for i in prange(h):
        for j in range(w):
            c = complex(-2.0 + j * 2.8 / w, -1.4 + i * 2.8 / h)
            z = 0.0j
            for k in range(max_iter):
                z = z * z + c
                if (z.real * z.real + z.imag * z.imag) >= 4.0:
                    image[i, j] = k
                    break
            else:
                image[i, j] = max_iter
    return image

@vectorize([float64(float64, float64)], target='cpu')
def numba_vectorize_add(a, b):
    return a + b

@vectorize([float64(float64, float64)], target='parallel')
def numba_vectorize_add_parallel(a, b):
    return a + b

@njit(parallel=True)
def sobel_numba_parallel(image):
    """Numba 并行实现 Sobel 边缘检测卷积"""
    h, w = image.shape
    out = np.zeros_like(image)
    for i in prange(1, h - 1):
        for j in range(1, w - 1):
            gx = (image[i-1, j+1] + 2*image[i, j+1] + image[i+1, j+1]) - \
                 (image[i-1, j-1] + 2*image[i, j-1] + image[i+1, j-1])
            gy = (image[i+1, j-1] + 2*image[i+1, j] + image[i+1, j+1]) - \
                 (image[i-1, j-1] + 2*image[i-1, j] + image[i-1, j+1])
            out[i, j] = np.sqrt(gx**2 + gy**2)
    return out

def run_numba_acceleration_test():
    st.header("Numba & NumPy 加速性能深度测试")
    st.caption("numba==0.63.1 | numpy==2.3.1")
    
    st.info("本实验旨在对比纯 Python、原生 NumPy 以及 Numba (JIT/Parallel) 在不同计算任务下的性能表现。")

    test_tabs = st.tabs([
        "蒙特卡罗 Pi 估算", 
        "曼德博集合 (Mandelbrot)", 
        "向量化 Ufunc 测试",
        "Sobel 卷积算子"
    ])

    with test_tabs[0]:
        st.subheader("任务 1：蒙特卡罗 Pi 估算")
        st.write("涉及大量循环和随机数生成的计算密集型任务。")
        
        n_points = st.select_slider("样本点数 (N)", options=[10**4, 10**5, 10**6, 10**7, 10**8], value=10**6, key="pi_n")
        
        if st.button("运行 Pi 估算对比", key="btn_pi"):
            numba_pi(100) # 预热
            numba_pi_parallel(100)
            
            results = []
            if n_points <= 10**6:
                start = time.perf_counter()
                python_res = python_pi(n_points)
                results.append({"实现方式": "Pure Python", "耗时 (s)": time.perf_counter() - start, "结果": python_res})
            else:
                st.warning(f"由于 N={n_points:,} 过大，已跳过 Pure Python 测试。")

            start = time.perf_counter()
            numba_res = numba_pi(n_points)
            results.append({"实现方式": "Numba JIT", "耗时 (s)": time.perf_counter() - start, "结果": numba_res})

            start = time.perf_counter()
            numba_p_res = numba_pi_parallel(n_points)
            results.append({"实现方式": "Numba Parallel", "耗时 (s)": time.perf_counter() - start, "结果": numba_p_res})

            st.table(pd.DataFrame(results))
            if len(results) >= 2:
                speedup = results[0]["耗时 (s)"] / results[1]["耗时 (s)"]
                st.success(f"Numba JIT 相比第一个方案加速了约 **{speedup:.1f}x**")

    with test_tabs[1]:
        st.subheader("任务 2：曼德博集合计算")
        st.write("嵌套循环计算，测试 JIT 对复杂分支逻辑的优化。")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            h_w = st.selectbox("图像分辨率", [256, 512, 1024, 2048], index=1, key="m_res")
        with col_m2:
            max_iter = st.slider("最大迭代次数", 10, 500, 100, key="m_iter")

        if st.button("生成曼德博集合", key="btn_m"):
            mandelbrot_numba(10, 10, 10) # 预热
            times = {}
            
            start = time.perf_counter()
            _ = mandelbrot_python(h_w, h_w, max_iter)
            times["NumPy Vectorized"] = time.perf_counter() - start
            
            start = time.perf_counter()
            _ = mandelbrot_numba(h_w, h_w, max_iter)
            times["Numba JIT"] = time.perf_counter() - start

            start = time.perf_counter()
            img_nb_p = mandelbrot_numba_parallel(h_w, h_w, max_iter)
            times["Numba Parallel"] = time.perf_counter() - start

            st.json(times)
            st.image(img_nb_p.astype(np.float32) / max_iter, caption="Mandelbrot (Numba Parallel)", use_container_width=True)

    with test_tabs[2]:
        st.subheader("任务 3：向量化 Ufunc 测试")
        st.write("测试 `@vectorize` 对自定义数学函数的自动 SIMD 和并行化。")
        
        array_size = st.select_slider("数组大小", options=[10**5, 10**6, 10**7, 5 * 10**7], value=10**6, key="u_size")
        
        if st.button("开始 Ufunc 压测", key="btn_u"):
            a, b = np.random.random(array_size), np.random.random(array_size)
            numba_vectorize_add(a[:10], b[:10]) # 预热
            
            perf = []
            start = time.perf_counter()
            _ = a + b
            perf.append({"方式": "Native NumPy (+)", "耗时 (ms)": (time.perf_counter() - start) * 1000})
            
            start = time.perf_counter()
            _ = numba_vectorize_add(a, b)
            perf.append({"方式": "Numba Vectorize (Single)", "耗时 (ms)": (time.perf_counter() - start) * 1000})
            
            start = time.perf_counter()
            _ = numba_vectorize_add_parallel(a, b)
            perf.append({"方式": "Numba Vectorize (Parallel)", "耗时 (ms)": (time.perf_counter() - start) * 1000})
            
            st.dataframe(pd.DataFrame(perf))

    with test_tabs[3]:
        st.subheader("任务 4：Sobel 边缘检测卷积算子")
        st.write("模拟图像处理中的卷积操作，典型的局部内存访问。")
        
        img_size = st.selectbox("模拟图像尺寸", [512, 1024, 2048, 4096], index=1, key="s_size")
        
        if st.button("运行 Sobel 压测", key="btn_s"):
            test_img = np.random.random((img_size, img_size)).astype(np.float32)
            sobel_numba_parallel(test_img[:10, :10]) # 预热
            
            start = time.perf_counter()
            sobel_res = sobel_numba_parallel(test_img)
            duration = time.perf_counter() - start
            
            st.metric("Numba Parallel 耗时", f"{duration:.4f} s")
            st.write(f"处理速度: {(img_size**2 / duration / 1e6):.2f} MPixels/s")
            st.image(sobel_res / (sobel_res.max() + 1e-6), caption="Sobel Filter Output", use_container_width=True)

if __name__ == "__main__":
    run_numba_acceleration_test()
