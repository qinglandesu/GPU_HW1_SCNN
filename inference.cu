#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>

/**/
__device__ __host__ uint32_t __builtin_bswap32(uint32_t val) {
    return ((val & 0x000000FF) << 24) |
           ((val & 0x0000FF00) << 8) |
           ((val & 0x00FF0000) >> 8) |
           ((val & 0xFF000000) >> 24);
}

// conv1: 1x28x28 -> 6x24x24, K=5
// 权重： [C_out, C_in, K, K] = [6, 1, 5, 5] 共 150 个 float
__constant__ float d_conv1_w_const[6 * 1 * 5 * 5];
__constant__ float d_conv1_b_const[6];
__constant__ float d_conv2_w_const[16 * 6 * 5 * 5];
__constant__ float d_conv2_b_const[16];


// IF 脉冲神经元：逐元素更新膜电位并生成 0/1 脉冲
__global__ void ifnode_forward(
    const float* __restrict__ in,   // 输入电流
    float* __restrict__ v,          // 膜电位（需要跨时间步保留）
    float* __restrict__ out,        // 脉冲输出 0/1
    int N,                          // 元素总数
    float threshold                 // 阈值，一般 1.0f
);

// 1D 最大池化：NCHW
__global__ void maxpool1d_forward_batch(
    const float* in,
    float* out,
    int N, int C, int H_in, int W_in,
    int K, int stride
);

// 全连接层 y = W x + b
__global__ void fc_if_forward_batch(
    const float* __restrict__ x,
    const float* __restrict__ W,
    const float* __restrict__ b,
    float* __restrict__ v,
    float* __restrict__ out,
    int N,
    int in_features,
    int out_features,
    float threshold
);

// 把当前时间步的 logits 累加到 logits_sum 上： logits_sum += logits_t
__global__ void fc3_and_accumulate_batch(
    const float* __restrict__ x,      // [N,84]
    const float* __restrict__ W,      // [10,84]
    const float* __restrict__ b,      // [10]
    float* __restrict__ logits_sum,   // [N,10]
    int N, int IN, int OUT            // IN=84, OUT=10
);

template<int BLOCK_H, int BLOCK_W>
__global__ void conv1_forward_shared_const(
    const float* __restrict__ in,
    float* __restrict__ out,
    int N,
    int H_in,
    int W_in
);

template<int BLOCK_H, int BLOCK_W>
__global__ void conv2_forward_shared_const(
    const float* __restrict__ in,
    float* __restrict__ out,
    int N,
    int H_in,
    int W_in
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
    const int BATCH = 256;
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
    checkCudaErrors(cudaMalloc(&d_pool2_out, BATCH * P2_N * sizeof(float)));
    // FC1 / IF3
    float *d_if3_v = nullptr, *d_if3_out = nullptr;
    checkCudaErrors(cudaMalloc(&d_if3_v, BATCH * FC1_OUT * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_if3_out, BATCH * FC1_OUT * sizeof(float)));
    // FC2 / IF4
    float *d_if4_v = nullptr, *d_if4_out = nullptr;
    checkCudaErrors(cudaMalloc(&d_if4_v, BATCH * FC2_OUT * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_if4_out, BATCH * FC2_OUT * sizeof(float)));
    // FC3 输出 logits
    // logits 累积缓冲区
    float* d_logits_sum = nullptr;
    checkCudaErrors(cudaMalloc(&d_logits_sum, BATCH * FC3_OUT * sizeof(float)));
    // host 端读取 logits 用于 argmax
    std::vector<float> h_logits(BATCH * FC3_OUT);

    // kernel 启动配置（简单用 1D 配置，conv/pool 自己在实现里用 3D 也可以）
    const int THREADS = 128;

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
            constexpr int BLOCK_H = 16;
            constexpr int BLOCK_W = 16;

            dim3 block(BLOCK_W, BLOCK_H);
            dim3 grid(
                (C1_W + BLOCK_W - 1) / BLOCK_W, // C1_W=24
                (C1_H + BLOCK_H - 1) / BLOCK_H, // C1_H=24
                cur_batch * C1_OUT_C            // 6
            );

            size_t shared_bytes =
                C1_IN_C * (BLOCK_H + C1_K - 1) * (BLOCK_W + C1_K - 1) * sizeof(float);
            // = 1 * 20 * 20 = 400 float ≈ 1.6KB

            conv1_forward_shared_const<BLOCK_H, BLOCK_W>
                <<<grid, block, shared_bytes>>>(
                    d_input,       // [cur_batch,1,28,28]
                    d_conv1_out,   // [cur_batch,6,24,24]
                    cur_batch,
                    IMG_H, IMG_W
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
                int total = cur_batch * C1_OUT_C * P1_H * P1_W;
                int blocks = (total + THREADS - 1) / THREADS;
                maxpool1d_forward_batch<<<blocks, THREADS>>>(
                    d_if1_out,
                    d_pool1_out,
                    cur_batch, C1_OUT_C, C1_H, C1_W,
                    P1_K, P1_STR
                );
                checkCudaErrors(cudaGetLastError());
            }

            // (4) conv2: [6,12,12] -> [16,8,8]
            {
                constexpr int BLOCK_H = 8;
                constexpr int BLOCK_W = 8;

                dim3 block(BLOCK_W, BLOCK_H);
                dim3 grid(
                    (C2_W + BLOCK_W - 1) / BLOCK_W, // C2_W=8
                    (C2_H + BLOCK_H - 1) / BLOCK_H, // C2_H=8
                    cur_batch * C2_OUT_C            // 16
                );

                size_t shared_bytes =
                    C2_IN_C * (BLOCK_H + C2_K - 1) * (BLOCK_W + C2_K - 1) * sizeof(float);
                // = 6 * 12 * 12 = 864 float ≈ 3.4KB

                conv2_forward_shared_const<BLOCK_H, BLOCK_W>
                    <<<grid, block, shared_bytes>>>(
                        d_pool1_out,   // [cur_batch,6,12,12]
                        d_conv2_out,   // [cur_batch,16,8,8]
                        cur_batch,
                        P1_H, P1_W     // 12, 12
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
                int total = cur_batch * C2_OUT_C * P2_H * P2_W;
                int blocks = (total + THREADS - 1) / THREADS;
                maxpool1d_forward_batch<<<blocks, THREADS>>>(
                    d_if2_out,
                    d_pool2_out,
                    cur_batch, C2_OUT_C, C2_H, C2_W,
                    P2_K, P2_STR
                );
                checkCudaErrors(cudaGetLastError());
            }

            // (8) fc1 + IF3: [N,256] -> [N,120] -> spike
            {
                int total = cur_batch * FC1_OUT;
                int blocks_fc1 = (total + THREADS - 1) / THREADS;
                fc_if_forward_batch<<<blocks_fc1, THREADS>>>(
                    d_pool2_out,   // x: [N,256]
                    d_fc1_w, d_fc1_b,
                    d_if3_v,       // v: [N,120]
                    d_if3_out,     // out: [N,120]
                    cur_batch,
                    FC1_IN, FC1_OUT,
                    1.0f           // threshold
                );
                checkCudaErrors(cudaGetLastError());
            }

            // (9) fc2 + IF4: [N,120] -> [N,84] -> spike
            {
                int total = cur_batch * FC2_OUT;
                int blocks_fc2 = (total + THREADS - 1) / THREADS;
                fc_if_forward_batch<<<blocks_fc2, THREADS>>>(
                    d_if3_out,     // x: [N,120]
                    d_fc2_w, d_fc2_b,
                    d_if4_v,       // v: [N,84]
                    d_if4_out,     // out: [N,84]
                    cur_batch,
                    FC2_IN, FC2_OUT,
                    1.0f
                );
                checkCudaErrors(cudaGetLastError());
            }

            // (10) fc3: [84] -> [10] (最终输出不再过 IF)
            // (11) logits 累加：logits_sum += fc3_out
            {
                int total = cur_batch * FC3_OUT;
                int blocks = (total + THREADS - 1) / THREADS;
                fc3_and_accumulate_batch<<<blocks, THREADS>>>(
                    d_if4_out, d_fc3_w, d_fc3_b,
                    d_logits_sum,
                    cur_batch,
                    FC3_IN, FC3_OUT
                );
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

    cudaFree(d_if3_v);
    cudaFree(d_if3_out);

    cudaFree(d_if4_v);
    cudaFree(d_if4_out);

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
	
    // Load test data"/../../.." +"/../../.." +
    auto images = read_mnist_images(dir +  "/data/FashionMNIST/raw/t10k-images-idx3-ubyte");
    auto labels = read_mnist_labels(dir +  "/data/FashionMNIST/raw/t10k-labels-idx1-ubyte");
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

    // 拷贝到 constant memory
    checkCudaErrors(cudaMemcpyToSymbol(
        d_conv1_w_const,
        conv1_w.data(),
        conv1_w.size() * sizeof(float),
        0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(
        d_conv1_b_const,
        conv1_b.data(),
        conv1_b.size() * sizeof(float),
        0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(
        d_conv2_w_const, conv2_w.data(),
        conv2_w.size() * sizeof(float),
        0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(
        d_conv2_b_const, conv2_b.data(),
        conv2_b.size() * sizeof(float),
        0, cudaMemcpyHostToDevice));


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

// in:  [N, C, H_in, W_in]
// out: [N, C, H_out, W_out]
__global__ void maxpool1d_forward_batch(
    const float* in,
    float* out,
    int N, int C, int H_in, int W_in,
    int K, int stride
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = (H_in - K) / stride + 1;
    int W_out = (W_in - K) / stride + 1;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c     = (idx / (W_out * H_out)) % C;
    int n     = idx / (W_out * H_out * C);

    float max_val = -1e30f;
    int h0 = h_out * stride, w0 = w_out * stride;
    for(int kh=0; kh<K; ++kh)
        for(int kw=0; kw<K; ++kw) {
            int h = h0 + kh, w = w0 + kw;
            int idx_in = ((n*C+c)*H_in + h)*W_in + w;
            float v = in[idx_in];
            if (v > max_val) max_val = v;
        }

    out[idx] = max_val;
}

// x:  [N, in_features]
// W:  [out_features, in_features]
// b:  [out_features]
// v:  [N, out_features]    膜电位（跨时间步累积）
// out:[N, out_features]    spike (0/1)
__global__ void fc_if_forward_batch(
    const float* __restrict__ x,
    const float* __restrict__ W,
    const float* __restrict__ b,
    float* __restrict__ v,
    float* __restrict__ out,
    int N,
    int in_features,
    int out_features,
    float threshold
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_features;
    if (idx >= total) return;

    int n  = idx / out_features;
    int o  = idx % out_features;

    const float* x_row = x + n * in_features;
    const float* w_row = W + o * in_features;

    float sum = b[o];
    #pragma unroll
    for (int j = 0; j < in_features; ++j) {
        sum += w_row[j] * x_row[j];
    }

    // IF 累积 & 发放
    float vi = v[idx] + sum;
    if (vi >= threshold) {
        out[idx] = 1.0f;
        v[idx]   = 0.0f;
    } else {
        out[idx] = 0.0f;
        v[idx]   = vi;
    }
}

// logits_sum[n, k] += logits_t[n, k]
__global__ void fc3_and_accumulate_batch(
    const float* __restrict__ x,      // [N,84]
    const float* __restrict__ W,      // [10,84]
    const float* __restrict__ b,      // [10]
    float* __restrict__ logits_sum,   // [N,10]
    int N, int IN, int OUT            // IN=84, OUT=10
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * OUT;
    if (idx >= total) return;

    int n = idx / OUT;
    int o = idx % OUT;

    const float* x_row = x + n * IN;
    const float* w_row = W + o * IN;

    float sum = b[o];
    #pragma unroll
    for (int j = 0; j < IN; ++j)
        sum += w_row[j] * x_row[j];

    logits_sum[idx] += sum;
}

// 通用 shared-memory 卷积 core
// in:  [N, C_IN, H_in, W_in]
// w:   [C_OUT, C_IN, K, K]
// b:   [C_OUT]
// out: [N, C_OUT, H_out, W_out]
template<
    int C_IN,
    int C_OUT,
    int K,
    int STRIDE,
    int PADDING,
    int BLOCK_H,
    int BLOCK_W
>
__device__ void conv_shared_core(
    const float* __restrict__ in,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    float* __restrict__ s_in,  // shared memory: [C_IN, TILE_H, TILE_W]
    int N,
    int H_in,
    int W_in
){
    int H_out = (H_in + 2 * PADDING - K) / STRIDE + 1;
    int W_out = (W_in + 2 * PADDING - K) / STRIDE + 1;

    int out_w0 = blockIdx.x * BLOCK_W;
    int out_h0 = blockIdx.y * BLOCK_H;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int co_n = blockIdx.z;
    int n  = co_n / C_OUT;
    int co = co_n % C_OUT;
    if (n >= N) return;

    constexpr int TILE_H = BLOCK_H + K - 1;
    constexpr int TILE_W = BLOCK_W + K - 1;

    // -------- 1. load 输入 patch -> shared memory --------
    for (int ci = 0; ci < C_IN; ++ci) {
        for (int th = ty; th < TILE_H; th += BLOCK_H) {
            int h_in = out_h0 * STRIDE - PADDING + th;

            for (int tw = tx; tw < TILE_W; tw += BLOCK_W) {
                int w_in = out_w0 * STRIDE - PADDING + tw;

                float val = 0.0f;
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    int idx_in = ((n * C_IN + ci) * H_in + h_in) * W_in + w_in;
                    val = in[idx_in];
                }

                int idx_s = (ci * TILE_H + th) * TILE_W + tw;
                s_in[idx_s] = val;
            }
        }
    }

    __syncthreads();

    // -------- 2. 每个 thread 计算一个输出像素 --------
    int h_out = out_h0 + ty;
    int w_out = out_w0 + tx;
    if (h_out >= H_out || w_out >= W_out) return;

    float sum = b[co];
    int w_base_co = co * (C_IN * K * K);

    #pragma unroll
    for (int ci = 0; ci < C_IN; ++ci) {
        int w_base_ci = w_base_co + ci * (K * K);

        #pragma unroll
        for (int kh = 0; kh < K; ++kh) {
            int th = ty + kh;

            #pragma unroll
            for (int kw = 0; kw < K; ++kw) {
                int tw = tx + kw;

                int idx_s = (ci * TILE_H + th) * TILE_W + tw;
                float vin = s_in[idx_s];

                int idx_w = w_base_ci + kh * K + kw;
                float ww  = w[idx_w];

                sum += vin * ww;
            }
        }
    }

    int idx_out = ((n * C_OUT + co) * H_out + h_out) * W_out + w_out;
    out[idx_out] = sum;
}

// conv1: 1x28x28 -> 6x24x24
template<int BLOCK_H, int BLOCK_W>
__global__ void conv1_forward_shared_const(
    const float* __restrict__ in,
    float* __restrict__ out,
    int N,
    int H_in,
    int W_in
){
    // shared memory: [C_IN=1, TILE_H, TILE_W]
    extern __shared__ float s_mem[];

    conv_shared_core<
        1, 6,        // C_IN, C_OUT
        5, 1, 0,     // K, STRIDE, PADDING
        BLOCK_H, BLOCK_W
    >(in,
      d_conv1_w_const,   // constant 权重
      d_conv1_b_const,   // constant 偏置
      out,
      s_mem,
      N, H_in, W_in);
}

// conv2: 6x12x12 -> 16x8x8
template<int BLOCK_H, int BLOCK_W>
__global__ void conv2_forward_shared_const(
    const float* __restrict__ in,
    float* __restrict__ out,
    int N,
    int H_in,
    int W_in
){
    // shared memory: [C_IN=6, TILE_H, TILE_W]
    extern __shared__ float s_mem[];

    conv_shared_core<
        6, 16,       // C_IN, C_OUT
        5, 1, 0,     // K, STRIDE, PADDING
        BLOCK_H, BLOCK_W
    >(in,
      d_conv2_w_const,
      d_conv2_b_const,
      out,
      s_mem,
      N, H_in, W_in);
}
