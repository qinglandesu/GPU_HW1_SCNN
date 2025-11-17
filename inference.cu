#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>

// TODO
/*
__device__ __host__ uint32_t __builtin_bswap32(uint32_t val) {
    return ((val & 0x000000FF) << 24) |
           ((val & 0x0000FF00) << 8) |
           ((val & 0x00FF0000) >> 8) |
           ((val & 0xFF000000) >> 24);
}
*/
// 多通道 2D 卷积：输入/输出都是 NCHW，这里 N=1（batch=1）
__global__ void conv2d_forward(
    const float* __restrict__ in,   // [C_in, H_in, W_in]
    const float* __restrict__ w,    // [C_out, C_in, K, K]
    const float* __restrict__ b,    // [C_out]
    float* __restrict__ out,        // [C_out, H_out, W_out]
    int C_in, int H_in, int W_in,
    int C_out,
    int K, int stride, int padding
);

// 2D 最大池化：NCHW，N=1
__global__ void maxpool2d_forward(
    const float* __restrict__ in,   // [C, H_in, W_in]
    float* __restrict__ out,        // [C, H_out, W_out]
    int C, int H_in, int W_in,
    int K, int stride
);

// IF 脉冲神经元：逐元素更新膜电位并生成 0/1 脉冲
__global__ void ifnode_forward(
    const float* __restrict__ in,   // 输入电流
    float* __restrict__ v,          // 膜电位（需要跨时间步保留）
    float* __restrict__ out,        // 脉冲输出 0/1
    int N,                          // 元素总数
    float threshold                 // 阈值，一般 1.0f
);

// Flatten：把 [C,H,W] 展平成 [C*H*W]
__global__ void flatten_forward(
    const float* __restrict__ in,   // [C, H, W]
    float* __restrict__ out,        // [C*H*W]
    int C, int H, int W
);

// 全连接层 y = W x + b
__global__ void linear_forward(
    const float* __restrict__ x,    // [in_features]
    const float* __restrict__ W,    // [out_features, in_features]
    const float* __restrict__ b,    // [out_features]
    float* __restrict__ y,          // [out_features]
    int in_features,
    int out_features
);

// 把当前时间步的 logits 累加到 logits_sum 上： logits_sum += logits_t
__global__ void add_logits(
    const float* __restrict__ logits_t, // [10]
    float* __restrict__ logits_sum,     // [10]
    int num_classes                     // =10
);

__global__ void conv2d_forward_batch(
    const float* __restrict__ in,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int N, int C_in, int H_in, int W_in,
    int C_out,
    int K, int stride, int padding
);

__global__ void maxpool2d_forward_batch(
    const float* __restrict__ in,
    float* __restrict__ out,
    int N, int C, int H_in, int W_in,
    int K, int stride
);

__global__ void linear_forward_batch(
    const float* __restrict__ x,
    const float* __restrict__ W,
    const float* __restrict__ b,
    float* __restrict__ y,
    int N,
    int in_features,
    int out_features
);

__global__ void add_logits_batch(
    const float* __restrict__ logits_t,  // [N, num_classes]
    float* __restrict__ logits_sum,      // [N, num_classes]
    int N,
    int num_classes
);

// ===================================================================================
// Helper for CUDA Error Handling - DO NOT MODIFY BEGIN
// ===================================================================================
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}
// ===================================================================================
// Helper for CUDA Error Handling - DO NOT MODIFY END
// ===================================================================================

// ===================================================================================
// Data and Parameter Loading Functions - DO NOT MODIFY BEGIN
// ===================================================================================
std::vector<std::vector<float>> read_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { std::cerr << "Cannot open file: " << path << std::endl; return {}; }
    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
    file.read((char*)&magic_number, 4); magic_number = __builtin_bswap32(magic_number);
    file.read((char*)&num_images, 4); num_images = __builtin_bswap32(num_images);
    file.read((char*)&num_rows, 4); num_rows = __builtin_bswap32(num_rows);
    file.read((char*)&num_cols, 4); num_cols = __builtin_bswap32(num_cols);
    std::vector<std::vector<float>> images(num_images, std::vector<float>(num_rows * num_cols));
    std::vector<unsigned char> buffer(num_rows * num_cols);
    for (int i = 0; i < num_images; ++i) {
        file.read((char*)buffer.data(), buffer.size());
        for (size_t j = 0; j < buffer.size(); ++j) {
            images[i][j] = (static_cast<float>(buffer[j]) / 255.0f - 0.5f) / 0.5f; // Normalization
        }
    }
    return images;
}

std::vector<int> read_mnist_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { std::cerr << "Cannot open file: " << path << std::endl; return {}; }
    int magic_number = 0, num_items = 0;
    file.read((char*)&magic_number, 4); magic_number = __builtin_bswap32(magic_number);
    file.read((char*)&num_items, 4); num_items = __builtin_bswap32(num_items);
    std::vector<int> labels(num_items);
    std::vector<unsigned char> buffer(num_items);
    file.read((char*)buffer.data(), num_items);
    for(int i = 0; i < num_items; ++i) { labels[i] = static_cast<int>(buffer[i]); }
    return labels;
}

std::vector<float> read_param(const std::string& path) {
    std::ifstream file(path);
    if (!file) { std::cerr << "Cannot open parameter file: " << path << std::endl; return {}; }
    std::vector<float> params; float param;
    while (file >> param) { params.push_back(param); }
    return params;
}

// ===================================================================================
// Data and Parameter Loading Functions - DO NOT MODIFY END
// ===================================================================================


std::vector<int> scnn_inference(
    const std::vector<std::vector<float>>& images,
    // Device pointers for parameters
    float* d_conv1_w, float* d_conv1_b, float* d_conv2_w, float* d_conv2_b,
    float* d_fc1_w,   float* d_fc1_b,   float* d_fc2_w,   float* d_fc2_b,
    float* d_fc3_w,   float* d_fc3_b
    // YOU CAN ADD MORE PARAMETERS HERE!!!
    )
{
    std::vector<int> predictions;
    const int num_images = images.size();
    predictions.reserve(num_images);

    // SNN-specific parameter, must match training
    const int T = 8;
    const int BATCH = 32;
    // 输入尺寸
    const int IMG_C = 1;
    const int IMG_H = 28;
    const int IMG_W = 28;
    // conv1: 1x28x28 -> 6x24x24 (K=5, S=1, P=0)
    const int C1_IN_C  = 1;
    const int C1_OUT_C = 6;
    const int C1_K     = 5;
    const int C1_STR   = 1;
    const int C1_PAD   = 0;
    const int C1_H     = (IMG_H + 2*C1_PAD - C1_K) / C1_STR + 1; // 24
    const int C1_W     = (IMG_W + 2*C1_PAD - C1_K) / C1_STR + 1; // 24
    const int C1_N     = C1_OUT_C * C1_H * C1_W;
    // pool1: 2x2, stride=2 -> 6x12x12
    const int P1_K   = 2;
    const int P1_STR = 2;
    const int P1_H   = C1_H / 2; // 12
    const int P1_W   = C1_W / 2; // 12
    const int P1_N   = C1_OUT_C * P1_H * P1_W;  // 通道数不变
    // conv2: 6x12x12 -> 16x8x8 (K=5)
    const int C2_IN_C  = 6;
    const int C2_OUT_C = 16;
    const int C2_K     = 5;
    const int C2_STR   = 1;
    const int C2_PAD   = 0;
    const int C2_H     = (P1_H + 2*C2_PAD - C2_K) / C2_STR + 1; // 8
    const int C2_W     = (P1_W + 2*C2_PAD - C2_K) / C2_STR + 1; // 8
    const int C2_N     = C2_OUT_C * C2_H * C2_W;
    // pool2: 2x2 -> 16x4x4
    const int P2_K   = 2;
    const int P2_STR = 2;
    const int P2_H   = C2_H / 2; // 4
    const int P2_W   = C2_W / 2; // 4
    const int P2_N   = C2_OUT_C * P2_H * P2_W; // 16*4*4 = 256
    // 全连接层尺寸
    const int FC1_IN  = P2_N;   // 256
    const int FC1_OUT = 120;
    const int FC2_IN  = FC1_OUT;
    const int FC2_OUT = 84;
    const int FC3_IN  = FC2_OUT;
    const int FC3_OUT = 10;

    // 分配中间特征图和膜电位的 GPU 缓冲区
    // conv1 / IF1 / pool1
    float *d_conv1_out = nullptr, *d_if1_v = nullptr, *d_if1_out = nullptr;
    float *d_pool1_out = nullptr;
    checkCudaErrors(cudaMalloc(&d_conv1_out, BATCH * C1_N * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_if1_v, BATCH * C1_N * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_if1_out, BATCH * C1_N * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_pool1_out, BATCH * P1_N * sizeof(float)));
    // conv2 / IF2 / pool2
    float *d_conv2_out = nullptr, *d_if2_v = nullptr, *d_if2_out = nullptr;
    float *d_pool2_out = nullptr;
    checkCudaErrors(cudaMalloc(&d_conv2_out, BATCH * C2_N * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_if2_v, BATCH * C2_N * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_if2_out, BATCH * C2_N * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_pool2_out, BATCH * P2_N * sizeof(float))); // 16x4x4
    // flatten 后的向量
    // float* d_flat = nullptr;
    // checkCudaErrors(cudaMalloc(&d_flat, FC1_IN * sizeof(float))); // 256
    // FC1 / IF3
    float *d_fc1_out = nullptr, *d_if3_v = nullptr, *d_if3_out = nullptr;
    checkCudaErrors(cudaMalloc(&d_fc1_out, BATCH * FC1_OUT * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_if3_v, BATCH * FC1_OUT * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_if3_out, BATCH * FC1_OUT * sizeof(float)));
    // FC2 / IF4
    float *d_fc2_out = nullptr, *d_if4_v = nullptr, *d_if4_out = nullptr;
    checkCudaErrors(cudaMalloc(&d_fc2_out, BATCH * FC2_OUT * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_if4_v, BATCH * FC2_OUT * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_if4_out, BATCH * FC2_OUT * sizeof(float)));
    // FC3 输出 logits
    float* d_fc3_out = nullptr;
    checkCudaErrors(cudaMalloc(&d_fc3_out, BATCH * FC3_OUT * sizeof(float)));
    // logits 累积缓冲区
    float* d_logits_sum = nullptr;
    checkCudaErrors(cudaMalloc(&d_logits_sum, BATCH * FC3_OUT * sizeof(float)));
    // host 端读取 logits 用于 argmax
    std::vector<float> h_logits(BATCH * FC3_OUT);

    // kernel 启动配置（简单用 1D 配置，conv/pool 自己在实现里用 3D 也可以）
    const int THREADS = 256;

    std::vector<float> h_all_images(num_images * IMG_C * IMG_H * IMG_W);
    for (int i = 0; i < num_images; ++i) {
        std::copy(
            images[i].begin(), images[i].end(),
            h_all_images.begin() + i * IMG_C * IMG_H * IMG_W
        );
    }
    float* d_all_images = nullptr;
    checkCudaErrors(cudaMalloc(
        &d_all_images,
        h_all_images.size() * sizeof(float)
    ));
    checkCudaErrors(cudaMemcpy(
        d_all_images,
        h_all_images.data(),
        h_all_images.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    // --- Loop over each image ---
    for (int base = 0; base < num_images; base += BATCH) {
        int cur_batch = std::min(BATCH, num_images - base);
        // images[i] 大小是 28*28
        const float* d_input = d_all_images + base * IMG_C * IMG_H * IMG_W;

        // 把所有 IF 膜电位清零
        checkCudaErrors(cudaMemset(d_if1_v, 0, cur_batch * C1_N * sizeof(float)));
        checkCudaErrors(cudaMemset(d_if2_v, 0, cur_batch * C2_N * sizeof(float)));
        checkCudaErrors(cudaMemset(d_if3_v, 0, cur_batch * FC1_OUT * sizeof(float)));
        checkCudaErrors(cudaMemset(d_if4_v, 0, cur_batch * FC2_OUT * sizeof(float)));
        // logits_sum 清零
        checkCudaErrors(cudaMemset(d_logits_sum, 0, cur_batch * FC3_OUT * sizeof(float)));

        // (1) conv1: [1,28,28] -> [6,24,24]
        {
            dim3 block(16, 16);
            dim3 grid(
                (C1_W + block.x - 1) / block.x,
                (C1_H + block.y - 1) / block.y,
                cur_batch * C1_OUT_C
            );
            conv2d_forward_batch<<<grid, block>>>(
                d_input,
                d_conv1_w, d_conv1_b,
                d_conv1_out,
                cur_batch, C1_IN_C, IMG_H, IMG_W,
                C1_OUT_C,
                C1_K, C1_STR, C1_PAD
            );
            checkCudaErrors(cudaGetLastError());
        }
        // 在 T 个时间步上循环
        for (int t = 0; t < T; ++t) {
            
            // (2) IF1: conv1_out -> if1_out (0/1)，更新 d_if1_v
            {
                int blocks = (cur_batch * C1_N + THREADS - 1) / THREADS;
                ifnode_forward<<<blocks, THREADS>>>(
                    d_conv1_out,
                    d_if1_v,
                    d_if1_out,
                    cur_batch * C1_N,
                    1.0f
                );
                checkCudaErrors(cudaGetLastError());
            }

            // (3) pool1: [6,24,24] -> [6,12,12]
            {
                dim3 block(16, 16);
                dim3 grid(
                    (P1_W + block.x - 1) / block.x,
                    (P1_H + block.y - 1) / block.y,
                    cur_batch * C1_OUT_C
                );
                maxpool2d_forward_batch<<<grid, block>>>(
                    d_if1_out,
                    d_pool1_out,
                    cur_batch, C1_OUT_C, C1_H, C1_W,
                    P1_K, P1_STR
                );
                checkCudaErrors(cudaGetLastError());
            }

            // (4) conv2: [6,12,12] -> [16,8,8]
            {
                dim3 block(16, 16);
                dim3 grid(
                    (C2_W + block.x - 1) / block.x,
                    (C2_H + block.y - 1) / block.y,
                    cur_batch * C2_OUT_C
                );
                conv2d_forward_batch<<<grid, block>>>(
                    d_pool1_out,
                    d_conv2_w, d_conv2_b,
                    d_conv2_out,
                    cur_batch, C2_IN_C, P1_H, P1_W,
                    C2_OUT_C,
                    C2_K, C2_STR, C2_PAD
                );
                checkCudaErrors(cudaGetLastError());
            }

            // (5) IF2: conv2_out -> if2_out
            {
                int blocks = (cur_batch * C2_N + THREADS - 1) / THREADS;
                ifnode_forward<<<blocks, THREADS>>>(
                    d_conv2_out,
                    d_if2_v,
                    d_if2_out,
                    cur_batch * C2_N,
                    1.0f
                );
                checkCudaErrors(cudaGetLastError());
            }

            // (6) pool2: [16,8,8] -> [16,4,4]
            {
                dim3 block(16, 16);
                dim3 grid(
                    (P2_W + block.x - 1) / block.x,
                    (P2_H + block.y - 1) / block.y,
                    cur_batch * C2_OUT_C
                );
                maxpool2d_forward_batch<<<grid, block>>>(
                    d_if2_out,
                    d_pool2_out,
                    cur_batch, C2_OUT_C, C2_H, C2_W,
                    P2_K, P2_STR
                );
                checkCudaErrors(cudaGetLastError());
            }

            /*
            // (7) flatten: [16,4,4] -> [256]
            {
                int N = FC1_IN;
                int blocks = (N + THREADS - 1) / THREADS;
                flatten_forward<<<blocks, THREADS>>>(
                    d_pool2_out,
                    d_flat,
                    C2_OUT_C, P2_H, P2_W
                );
                checkCudaErrors(cudaGetLastError());
            }
            */

            // (8) fc1 + IF3: [256] -> [120] -> 0/1
            {
                int blocks_fc1 = (cur_batch * FC1_OUT + THREADS - 1) / THREADS;
                linear_forward_batch<<<blocks_fc1, THREADS>>>(
                    d_pool2_out,
                    d_fc1_w, d_fc1_b,
                    d_fc1_out,
                    cur_batch,
                    FC1_IN, FC1_OUT
                );
                checkCudaErrors(cudaGetLastError());

                ifnode_forward<<<blocks_fc1, THREADS>>>(
                    d_fc1_out,
                    d_if3_v,
                    d_if3_out,
                    cur_batch * FC1_OUT,
                    1.0f
                );
                checkCudaErrors(cudaGetLastError());
            }

            // (9) fc2 + IF4: [120] -> [84] -> 0/1
            {
                int blocks_fc2 = (cur_batch * FC2_OUT + THREADS - 1) / THREADS;
                linear_forward_batch<<<blocks_fc2, THREADS>>>(
                    d_if3_out,
                    d_fc2_w, d_fc2_b,
                    d_fc2_out,
                    cur_batch,
                    FC2_IN, FC2_OUT
                );
                checkCudaErrors(cudaGetLastError());

                ifnode_forward<<<blocks_fc2, THREADS>>>(
                    d_fc2_out,
                    d_if4_v,
                    d_if4_out,
                    cur_batch * FC2_OUT,
                    1.0f
                );
                checkCudaErrors(cudaGetLastError());
            }

            // (10) fc3: [84] -> [10] (最终输出不再过 IF)
            {
                int blocks_fc3 = (cur_batch * FC3_OUT + THREADS - 1) / THREADS;
                linear_forward_batch<<<blocks_fc3, THREADS>>>(
                    d_if4_out,
                    d_fc3_w, d_fc3_b,
                    d_fc3_out,
                    cur_batch,
                    FC3_IN, FC3_OUT
                );
                checkCudaErrors(cudaGetLastError());
            }

            // (11) logits 累加：logits_sum += fc3_out
            {
                int blocks = (cur_batch * FC3_OUT + THREADS - 1) / THREADS;
                add_logits_batch<<<blocks, THREADS>>>(
                    d_fc3_out,
                    d_logits_sum,
                    cur_batch,
                    FC3_OUT
                );
                checkCudaErrors(cudaGetLastError());
            }
        } // T loop

        // 5. 把 logits_sum 拷回 CPU，除以 T，然后 argmax 得到预测类别
        checkCudaErrors(cudaMemcpy(
            h_logits.data(),
            d_logits_sum,
            cur_batch * FC3_OUT * sizeof(float),
            cudaMemcpyDeviceToHost
        ));
        for (int n = 0; n < cur_batch; ++n) {
            int pred = 0;
            float best = h_logits[n * FC3_OUT] / T;
            for (int k = 1; k < FC3_OUT; ++k) {
                float v = h_logits[n * FC3_OUT + k] / T;
                if (v > best) {
                    best = v;
                    pred = k;
                }
            }
            predictions.push_back(pred);
        }
    } // image loop

    // 释放中间 GPU 内存
    cudaFree(d_all_images);

    cudaFree(d_conv1_out);
    cudaFree(d_if1_v);
    cudaFree(d_if1_out);
    cudaFree(d_pool1_out);

    cudaFree(d_conv2_out);
    cudaFree(d_if2_v);
    cudaFree(d_if2_out);
    cudaFree(d_pool2_out);

    //cudaFree(d_flat);

    cudaFree(d_fc1_out);
    cudaFree(d_if3_v);
    cudaFree(d_if3_out);

    cudaFree(d_fc2_out);
    cudaFree(d_if4_v);
    cudaFree(d_if4_out);

    cudaFree(d_fc3_out);
    cudaFree(d_logits_sum);

    // Memory is freed in main.

    return predictions;
}

// ===================================================================================
// Main Function -  DO NOT MODIFY BEGIN
// ===================================================================================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model_and_data_dir>" << std::endl;
        return 1;
    }
	std::string dir = argv[1];
	
    // Load test data
    // TODO "/../../.." +
    auto images = read_mnist_images(dir + "/../../.." + "/data/FashionMNIST/raw/t10k-images-idx3-ubyte");
    auto labels = read_mnist_labels(dir + "/../../.." + "/data/FashionMNIST/raw/t10k-labels-idx1-ubyte");
    if (images.empty() || labels.empty()) return 1;

    // Load model parameters to host memory
    auto conv1_w = read_param(dir + "/conv1.weight.txt");
    auto conv1_b = read_param(dir + "/conv1.bias.txt");
    auto conv2_w = read_param(dir + "/conv2.weight.txt");
    auto conv2_b = read_param(dir + "/conv2.bias.txt");
    auto fc1_w = read_param(dir + "/fc1.weight.txt");
    auto fc1_b = read_param(dir + "/fc1.bias.txt");
    auto fc2_w = read_param(dir + "/fc2.weight.txt");
    auto fc2_b = read_param(dir + "/fc2.bias.txt");
    auto fc3_w = read_param(dir + "/fc3.weight.txt");
    auto fc3_b = read_param(dir + "/fc3.bias.txt");
    
    // --- 1. Allocate all necessary GPU memory ---
    // Device pointers for parameters
    float *d_conv1_w, *d_conv1_b, *d_conv2_w, *d_conv2_b;
    float *d_fc1_w, *d_fc1_b, *d_fc2_w, *d_fc2_b, *d_fc3_w, *d_fc3_b;

    // Allocate parameters
    checkCudaErrors(cudaMalloc(&d_conv1_w, conv1_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv1_b, conv1_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv2_w, conv2_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv2_b, conv2_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc1_w,   fc1_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc1_b,   fc1_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc2_w,   fc2_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc2_b,   fc2_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc3_w,   fc3_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc3_b,   fc3_b.size() * sizeof(float)));

    // --- 2. Copy constant parameters from host to device ---
    checkCudaErrors(cudaMemcpy(d_conv1_w, conv1_w.data(), conv1_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv1_b, conv1_b.data(), conv1_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv2_w, conv2_w.data(), conv2_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv2_b, conv2_b.data(), conv2_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc1_w, fc1_w.data(), fc1_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc1_b, fc1_b.data(), fc1_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc2_w, fc2_w.data(), fc2_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc2_b, fc2_b.data(), fc2_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc3_w, fc3_w.data(), fc3_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc3_b, fc3_b.data(), fc3_b.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();
    
// ===================================================================================
// Main Function -  DO NOT MODIFY END
// ===================================================================================

    // --- 3. Perform inference ---
    // Pass device pointers to the inference function
    std::vector<int> predictions = scnn_inference(images,
        d_conv1_w, d_conv1_b, d_conv2_w, d_conv2_b,
        d_fc1_w, d_fc1_b, d_fc2_w, d_fc2_b, d_fc3_w, d_fc3_b
        // YOU CAN ADD MORE PARAMETERS HERE!!!
        );
    
// ===================================================================================
// Main Function -  DO NOT MODIFY BEGIN
// ===================================================================================

    // Synchronize to ensure all GPU work is done before stopping the timer
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Stop timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    
    // --- 4. Free all allocated GPU memory ---
    checkCudaErrors(cudaFree(d_conv1_w));
    checkCudaErrors(cudaFree(d_conv1_b));
    checkCudaErrors(cudaFree(d_conv2_w));
    checkCudaErrors(cudaFree(d_conv2_b));
    checkCudaErrors(cudaFree(d_fc1_w));
    checkCudaErrors(cudaFree(d_fc1_b));
    checkCudaErrors(cudaFree(d_fc2_w));
    checkCudaErrors(cudaFree(d_fc2_b));
    checkCudaErrors(cudaFree(d_fc3_w));
    checkCudaErrors(cudaFree(d_fc3_b));
    
    // Calculate accuracy
    int correct_predictions = 0;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (predictions[i] == labels[i]) {
            correct_predictions++;
        }
    }
    double accuracy = static_cast<double>(correct_predictions) / labels.size();
    
    // Output result in the required format
    std::cout << std::fixed << std::setprecision(4) << diff.count() << ":" << accuracy << std::endl;
    
    return 0;
}
// ===================================================================================
// Main Function -  DO NOT MODIFY END
// ===================================================================================


// 多通道 2D 卷积：输入/输出都是 NCHW，这里 N=1（batch=1）
__global__ void conv2d_forward(
    const float* __restrict__ in,   // [C_in, H_in, W_in]
    const float* __restrict__ w,    // [C_out, C_in, K, K]
    const float* __restrict__ b,    // [C_out]
    float* __restrict__ out,        // [C_out, H_out, W_out]
    int C_in, int H_in, int W_in,
    int C_out,
    int K, int stride, int padding
)
{
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int co    = blockIdx.z;  // 输出通道

    // 计算输出空间大小
    int H_out = (H_in + 2 * padding - K) / stride + 1;
    int W_out = (W_in + 2 * padding - K) / stride + 1;

    if (co < C_out && h_out < H_out && w_out < W_out) {
        // 对应的输入中心起点（左上角）
        int h_in_start = h_out * stride - padding;
        int w_in_start = w_out * stride - padding;

        float sum = b[co];  // 先加上 bias

        // 卷积求和
        for (int ci = 0; ci < C_in; ++ci) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    int h_in = h_in_start + kh;
                    int w_in = w_in_start + kw;

                    // 带 padding 时要判断越界
                    if (h_in < 0 || h_in >= H_in || w_in < 0 || w_in >= W_in)
                        continue;

                    int idx_in = ci * H_in * W_in + h_in * W_in + w_in;
                    // w 的 index: [co, ci, kh, kw]
                    int idx_w  = ((co * C_in + ci) * K + kh) * K + kw;

                    sum += in[idx_in] * w[idx_w];
                }
            }
        }

        int idx_out = co * H_out * W_out + h_out * W_out + w_out;
        out[idx_out] = sum;
    }
}

// 2D 最大池化：NCHW，N=1
__global__ void maxpool2d_forward(
    const float* __restrict__ in,   // [C, H_in, W_in]
    float* __restrict__ out,        // [C, H_out, W_out]
    int C, int H_in, int W_in,
    int K, int stride
)
{
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int c     = blockIdx.z;   // 一个 block.z 对应一个通道

    int H_out = (H_in - K) / stride + 1;
    int W_out = (W_in - K) / stride + 1;

    if (c < C && h_out < H_out && w_out < W_out) {
        int h_start = h_out * stride;
        int w_start = w_out * stride;

        float max_val = -1e30f;  // 很小的初始值
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int h = h_start + kh;
                int w = w_start + kw;
                int idx_in = c * H_in * W_in + h * W_in + w;
                float v = in[idx_in];
                if (v > max_val) max_val = v;
            }
        }

        int idx_out = c * H_out * W_out + h_out * W_out + w_out;
        out[idx_out] = max_val;
    }
}

// IF 脉冲神经元：逐元素更新膜电位并生成 0/1 脉冲
__global__ void ifnode_forward(
    const float* __restrict__ in,   // 输入电流
    float* __restrict__ v,          // 膜电位（需要跨时间步保留）
    float* __restrict__ out,        // 脉冲输出 0/1
    int N,                          // 元素总数
    float threshold                 // 阈值，一般 1.0f
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float vi = v[i] + in[i];          // 更新膜电位
        if (vi >= threshold) {
            out[i] = 1.0f;               // 产生脉冲
            v[i]   = 0.0f;               // 重置膜电位（这里采用 reset_to_zero）
        } else {
            out[i] = 0.0f;
            v[i]   = vi;                 // 没有发放就保持新的膜电位
        }
    }
}

// Flatten：把 [C,H,W] 展平成 [C*H*W]
__global__ void flatten_forward(
    const float* __restrict__ in,   // [C, H, W]
    float* __restrict__ out,        // [C*H*W]
    int C, int H, int W
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = C * H * W;
    if (idx < N) {
        // in 的内存本来就是 C 维优先：in[c*H*W + h*W + w]
        // 展平后 out[idx] 顺序相同，直接拷贝即可
        out[idx] = in[idx];
    }
}

// 全连接层 y = W x + b
__global__ void linear_forward(
    const float* __restrict__ x,    // [in_features]
    const float* __restrict__ W,    // [out_features, in_features]
    const float* __restrict__ b,    // [out_features]
    float* __restrict__ y,          // [out_features]
    int in_features,
    int out_features
)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o < out_features) {
        float sum = 0.0f;
        const float* w_row = W + o * in_features;  // 第 o 行
        for (int j = 0; j < in_features; ++j) {
            sum += w_row[j] * x[j];
        }
        y[o] = sum + b[o];
    }
}

// 把当前时间步的 logits 累加到 logits_sum 上： logits_sum += logits_t
__global__ void add_logits(
    const float* __restrict__ logits_t, // [10]
    float* __restrict__ logits_sum,     // [10]
    int num_classes                     // =10
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_classes) {
        logits_sum[i] += logits_t[i];
    }
}

// in:  [N, C_in, H_in, W_in]
// out: [N, C_out, H_out, W_out]
__global__ void conv2d_forward_batch(
    const float* __restrict__ in,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int N, int C_in, int H_in, int W_in,
    int C_out,
    int K, int stride, int padding
){
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int co_n  = blockIdx.z; // 合并了 batch 和 out_channel

    int H_out = (H_in + 2 * padding - K) / stride + 1;
    int W_out = (W_in + 2 * padding - K) / stride + 1;

    int total_channels = N * C_out;
    if (co_n >= total_channels || h_out >= H_out || w_out >= W_out) return;

    int n  = co_n / C_out;   // batch index
    int co = co_n % C_out;   // output channel index

    float sum = b[co];

    for (int ci = 0; ci < C_in; ++ci) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int h_in = h_out * stride - padding + kh;
                int w_in = w_out * stride - padding + kw;
                if (h_in < 0 || h_in >= H_in || w_in < 0 || w_in >= W_in) continue;

                int idx_in = ((n * C_in + ci) * H_in + h_in) * W_in + w_in;
                int idx_w  = ((co * C_in + ci) * K + kh) * K + kw;

                sum += in[idx_in] * w[idx_w];
            }
        }
    }

    int idx_out = ((n * C_out + co) * H_out + h_out) * W_out + w_out;
    out[idx_out] = sum;
}

// in:  [N, C, H_in, W_in]
// out: [N, C, H_out, W_out]
__global__ void maxpool2d_forward_batch(
    const float* __restrict__ in,
    float* __restrict__ out,
    int N, int C, int H_in, int W_in,
    int K, int stride
){
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int c_n   = blockIdx.z; // 合并 batch 和 channel

    int H_out = (H_in - K) / stride + 1;
    int W_out = (W_in - K) / stride + 1;

    int total = N * C;
    if (c_n >= total || h_out >= H_out || w_out >= W_out) return;

    int n = c_n / C;
    int c = c_n % C;

    int h_start = h_out * stride;
    int w_start = w_out * stride;

    float max_val = -1e30f;
    for (int kh = 0; kh < K; ++kh) {
        for (int kw = 0; kw < K; ++kw) {
            int h = h_start + kh;
            int w = w_start + kw;
            int idx_in = ((n * C + c) * H_in + h) * W_in + w;
            float v = in[idx_in];
            if (v > max_val) max_val = v;
        }
    }

    int idx_out = ((n * C + c) * H_out + h_out) * W_out + w_out;
    out[idx_out] = max_val;
}

// x: [N, in_features]
// W: [out_features, in_features]
// b: [out_features]
// y: [N, out_features]
__global__ void linear_forward_batch(
    const float* __restrict__ x,
    const float* __restrict__ W,
    const float* __restrict__ b,
    float* __restrict__ y,
    int N,
    int in_features,
    int out_features
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_features;
    if (idx >= total) return;

    int n  = idx / out_features;
    int o  = idx % out_features;

    const float* x_row = x + n * in_features;
    const float* w_row = W + o * in_features;

    float sum = b[o];
    for (int j = 0; j < in_features; ++j) {
        sum += w_row[j] * x_row[j];
    }
    y[n * out_features + o] = sum;
}

// logits_sum[n, k] += logits_t[n, k]
__global__ void add_logits_batch(
    const float* __restrict__ logits_t,  // [N, num_classes]
    float* __restrict__ logits_sum,      // [N, num_classes]
    int N,
    int num_classes
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * num_classes;
    if (idx < total) {
        logits_sum[idx] += logits_t[idx];
    }
}
