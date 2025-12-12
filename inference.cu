#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <unordered_map>
#include <queue>
#include <mutex>

/*
__device__ __host__ uint32_t __builtin_bswap32(uint32_t val) {
    return ((val & 0x000000FF) << 24) |
           ((val & 0x0000FF00) << 8) |
           ((val & 0x00FF0000) >> 8) |
           ((val & 0xFF000000) >> 24);
}*/

__constant__ float d_conv1_w_const[6 * 1 * 5 * 5];
__constant__ float d_conv1_b_const[6];
__constant__ float d_conv2_w_const[16 * 6 * 5 * 5];
__constant__ float d_conv2_b_const[16];

// SNN-specific parameter, must match training
const int T = 8;
const int BATCH = 512;
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

// kernel 启动配置（简单用 1D 配置，conv/pool 自己在实现里用 3D 也可以）
const int THREADS = 128;

// 静态内存池定义
// 计算总内存需求（单位为元素个数，不是字节）
const size_t TOTAL_ELEMENTS_PER_STREAM = 
    BATCH * (C1_N * 3 +    // conv1_out + if1_v + if1_out
                 P1_N +         // pool1_out
                 C2_N * 2 +     // if2_v + if2_out  
                 P2_N +         // pool2_out
                 FC1_OUT * 2 +  // if3_v + if3_out
                 FC2_OUT * 2 +  // if4_v + if4_out
                 FC3_OUT);      // logits_sum
// 双流需要两套内存
const size_t TOTAL_ELEMENTS = TOTAL_ELEMENTS_PER_STREAM * 2;
// 静态device内存池（全局声明）
__device__ float d_static_pool[TOTAL_ELEMENTS];

// 修正：使用主机端指针管理偏移，kernel中通过参数传递
float* d_conv1_out = nullptr;
float* d_if1_v = nullptr;
float* d_if1_out = nullptr;
float* d_pool1_out = nullptr;
float* d_if2_v = nullptr;
float* d_if2_out = nullptr;
float* d_pool2_out = nullptr;
float* d_if3_v = nullptr;
float* d_if3_out = nullptr;
float* d_if4_v = nullptr;
float* d_if4_out = nullptr;
float* d_logits_sum = nullptr;
// 第二套内存（用于stream2）
float* d_conv1_out_2 = nullptr;
float* d_if1_v_2 = nullptr;
float* d_if1_out_2 = nullptr;
float* d_pool1_out_2 = nullptr;
float* d_if2_v_2 = nullptr;
float* d_if2_out_2 = nullptr;
float* d_pool2_out_2 = nullptr;
float* d_if3_v_2 = nullptr;
float* d_if3_out_2 = nullptr;
float* d_if4_v_2 = nullptr;
float* d_if4_out_2 = nullptr;
float* d_logits_sum_2 = nullptr;

// 初始化静态内存池指针
void init_static_pointers() {
    // 获取device上静态内存的起始地址
    void* base_ptr = nullptr;
    cudaError_t err = cudaGetSymbolAddress(&base_ptr, d_static_pool);
    if (err != cudaSuccess) {
        printf("Error getting symbol address: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // 转换为正确的指针类型
    float* device_base_ptr = reinterpret_cast<float*>(base_ptr);
    
    // 流1的内存分配 - 计算偏移量（主机端计算）
    size_t offset = 0;
    d_conv1_out = device_base_ptr + offset;  // 起始地址
    offset += BATCH * C1_N;
    
    d_if1_v = device_base_ptr + offset;
    offset += BATCH * C1_N;
    
    d_if1_out = device_base_ptr + offset;
    offset += BATCH * C1_N;
    
    d_pool1_out = device_base_ptr + offset;
    offset += BATCH * P1_N;
    
    d_if2_v = device_base_ptr + offset;
    offset += BATCH * C2_N;
    
    d_if2_out = device_base_ptr + offset;
    offset += BATCH * C2_N;
    
    d_pool2_out = device_base_ptr + offset;
    offset += BATCH * P2_N;
    
    d_if3_v = device_base_ptr + offset;
    offset += BATCH * FC1_OUT;
    
    d_if3_out = device_base_ptr + offset;
    offset += BATCH * FC1_OUT;
    
    d_if4_v = device_base_ptr + offset;
    offset += BATCH * FC2_OUT;
    
    d_if4_out = device_base_ptr + offset;
    offset += BATCH * FC2_OUT;
    
    d_logits_sum = device_base_ptr + offset;
    offset += BATCH * FC3_OUT;
    
    // 流2的内存分配（接在流1后面）
    d_conv1_out_2 = device_base_ptr + offset;
    offset += BATCH * C1_N;
    
    d_if1_v_2 = device_base_ptr + offset;
    offset += BATCH * C1_N;
    
    d_if1_out_2 = device_base_ptr + offset;
    offset += BATCH * C1_N;
    
    d_pool1_out_2 = device_base_ptr + offset;
    offset += BATCH * P1_N;
    
    d_if2_v_2 = device_base_ptr + offset;
    offset += BATCH * C2_N;
    
    d_if2_out_2 = device_base_ptr + offset;
    offset += BATCH * C2_N;
    
    d_pool2_out_2 = device_base_ptr + offset;
    offset += BATCH * P2_N;
    
    d_if3_v_2 = device_base_ptr + offset;
    offset += BATCH * FC1_OUT;
    
    d_if3_out_2 = device_base_ptr + offset;
    offset += BATCH * FC1_OUT;
    
    d_if4_v_2 = device_base_ptr + offset;
    offset += BATCH * FC2_OUT;
    
    d_if4_out_2 = device_base_ptr + offset;
    offset += BATCH * FC2_OUT;
    
    d_logits_sum_2 = device_base_ptr + offset;
    offset += BATCH * FC3_OUT;
    
    // 验证指针计算
    size_t total_calculated = offset;  // 现在offset就是总元素数
    if (total_calculated != TOTAL_ELEMENTS) {
        printf("Warning: Memory calculation mismatch! Expected %zu, got %zu\n", 
               TOTAL_ELEMENTS, total_calculated);
    }
    
    //printf("Static memory pool initialized. Total elements: %zu (%.2f MB)\n",
    //       TOTAL_ELEMENTS, TOTAL_ELEMENTS * sizeof(float) / (1024.0 * 1024.0));
}

void process_batch(
    cudaStream_t stream,
    const float* d_input, int cur_batch,
    float* d_conv1_out, float* d_if1_v, float* d_if1_out,
    float* d_pool1_out, float* d_if2_v, float* d_if2_out, 
    float* d_pool2_out, float* d_if3_v, float* d_if3_out,
    float* d_if4_v, float* d_if4_out, float* d_logits_sum,
    float* d_fc1_w,   float* d_fc1_b,   float* d_fc2_w,   float* d_fc2_b,
    float* d_fc3_w,   float* d_fc3_b);

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

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // 分配中间特征图和膜电位的 GPU 缓冲区
    init_static_pointers();
    // 检查分配是否成功
    if (!d_conv1_out || !d_if1_v || !d_if1_out || !d_pool1_out ||
        !d_if2_v || !d_if2_out || !d_pool2_out ||
        !d_if3_v || !d_if3_out || !d_if4_v || !d_if4_out || !d_logits_sum ||
        !d_conv1_out_2 || !d_if1_v_2 || !d_if1_out_2 || !d_pool1_out_2 ||
        !d_if2_v_2 || !d_if2_out_2 || !d_pool2_out_2 ||
        !d_if3_v_2 || !d_if3_out_2 || !d_if4_v_2 || !d_if4_out_2 || !d_logits_sum_2)
    {
        std::cerr << "GPU memory allocation failed!" << std::endl;
        return predictions;
    }

    // host 端读取 logits 用于 argmax
    std::vector<float> h_logits(BATCH * FC3_OUT);

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
    for (int base = 0; base < num_images; base += BATCH* 2) {
        int cur_batch = std::min(BATCH, num_images - base);
        // images[i] 大小是 28*28
        const float* d_input = d_all_images + base * IMG_C * IMG_H * IMG_W;

        // 把所有 IF 膜电位清零
        cudaMemset(d_if1_v, 0, cur_batch * C1_N * sizeof(float));
        cudaMemset(d_if2_v, 0, cur_batch * C2_N * sizeof(float));
        cudaMemset(d_if3_v, 0, cur_batch * FC1_OUT * sizeof(float));
        cudaMemset(d_if4_v, 0, cur_batch * FC2_OUT * sizeof(float));
        // logits_sum 清零
        cudaMemset(d_logits_sum, 0, cur_batch * FC3_OUT * sizeof(float));

        // 流1: 处理第一个batch
        process_batch(stream1, d_input, cur_batch, 
                    d_conv1_out, d_if1_v, d_if1_out, 
                    d_pool1_out, d_if2_v, d_if2_out,
                    d_pool2_out, d_if3_v, d_if3_out,
                    d_if4_v, d_if4_out, d_logits_sum,
                    d_fc1_w, d_fc1_b, d_fc2_w, d_fc2_b,
                    d_fc3_w, d_fc3_b);

        int cur_batch_2 = std::min(BATCH, num_images - base - BATCH);
        if(cur_batch_2 > 0)
        {
            const float* d_input_2 = d_all_images + (base + BATCH) * IMG_C * IMG_H * IMG_W;
            
            // 把所有 IF 膜电位清零
            cudaMemset(d_if1_v_2, 0, cur_batch_2 * C1_N * sizeof(float));
            cudaMemset(d_if2_v_2, 0, cur_batch_2 * C2_N * sizeof(float));
            cudaMemset(d_if3_v_2, 0, cur_batch_2 * FC1_OUT * sizeof(float));
            cudaMemset(d_if4_v_2, 0, cur_batch_2 * FC2_OUT * sizeof(float));
            // logits_sum 清零
            cudaMemset(d_logits_sum_2, 0, cur_batch_2 * FC3_OUT * sizeof(float));

            // 流2: 处理第二个batch（与流1并行）
            process_batch(stream2, d_input_2, cur_batch_2, 
                        d_conv1_out_2, d_if1_v_2, d_if1_out_2, 
                        d_pool1_out_2, d_if2_v_2, d_if2_out_2,
                        d_pool2_out_2, d_if3_v_2, d_if3_out_2,
                        d_if4_v_2, d_if4_out_2, d_logits_sum_2,
                        d_fc1_w, d_fc1_b, d_fc2_w, d_fc2_b,
                        d_fc3_w, d_fc3_b);
        }

        // 等待两个流完成
        cudaStreamSynchronize(stream1);
        // 5. 把 logits_sum 拷回 CPU，除以 T，然后 argmax 得到预测类别
        cudaMemcpy(
            h_logits.data(),
            d_logits_sum,
            cur_batch * FC3_OUT * sizeof(float),
            cudaMemcpyDeviceToHost
        );
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
        cudaStreamSynchronize(stream2);
        // 5. 把 logits_sum 拷回 CPU，除以 T，然后 argmax 得到预测类别
        if(cur_batch_2 > 0)
        {
            cudaMemcpy(
                h_logits.data(),
                d_logits_sum_2,
                cur_batch_2 * FC3_OUT * sizeof(float),
                cudaMemcpyDeviceToHost
            );
            for (int n = 0; n < cur_batch_2; ++n) {
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
        }
    } // image loop

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

    // 拷贝到 constant memory
    cudaMemcpyToSymbol(
        d_conv1_w_const,
        conv1_w.data(),
        conv1_w.size() * sizeof(float),
        0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(
        d_conv1_b_const,
        conv1_b.data(),
        conv1_b.size() * sizeof(float),
        0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(
        d_conv2_w_const, conv2_w.data(),
        conv2_w.size() * sizeof(float),
        0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(
        d_conv2_b_const, conv2_b.data(),
        conv2_b.size() * sizeof(float),
        0, cudaMemcpyHostToDevice);


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

// 添加更多PTX优化函数

// 使用PTX优化的IF神经元
// 简化的PTX优化 - 只在关键计算使用PTX
__global__ void ifnode_forward_ptx(
    const float* __restrict__ in,
    float* __restrict__ v,
    float* __restrict__ out,
    int N,
    float threshold)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float current_v, current_in;
    
    // 使用PTX加载（比普通加载更快）
    asm("ld.global.f32 %0, [%1];" : "=f"(current_in) : "l"(in + i));
    asm("ld.global.f32 %0, [%1];" : "=f"(current_v) : "l"(v + i));
    
    // 普通C++计算（编译器会优化）
    float new_v = current_v + current_in;
    
    if (new_v >= threshold) {
        out[i] = 1.0f;
        v[i] = 0.0f;
    } else {
        out[i] = 0.0f;
        v[i] = new_v;
    }
}

// 使用PTX优化的最大池化
__global__ void maxpool2d_forward_batch_ptx(
    const float* __restrict__ in,
    float* __restrict__ out,
    int N, int C,
    int H_in, int W_in,
    int K, int stride)
{
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int nc = blockIdx.z;

    int H_out = (H_in - K) / stride + 1;
    int W_out = (W_in - K) / stride + 1;

    if (w_out >= W_out || h_out >= H_out || nc >= N * C) return;

    int n = nc / C;
    int c = nc % C;

    int h0 = h_out * stride;
    int w0 = w_out * stride;

    float max_val;
    asm("mov.f32 %0, 0ff0000000;" : "=f"(max_val)); // -inf

    // 针对2x2池化的PTX优化
    if (K == 2 && stride == 2) {
        int idx_base = ((n * C + c) * H_in + h0) * W_in + w0;
        
        float v00, v01, v10, v11;
        asm("ld.global.f32 %0, [%1];" : "=f"(v00) : "l"(in + idx_base));
        asm("ld.global.f32 %0, [%1];" : "=f"(v01) : "l"(in + idx_base + 1));
        asm("ld.global.f32 %0, [%1];" : "=f"(v10) : "l"(in + idx_base + W_in));
        asm("ld.global.f32 %0, [%1];" : "=f"(v11) : "l"(in + idx_base + W_in + 1));

        // PTX最大值计算
        asm("max.f32 %0, %1, %2;" : "=f"(max_val) : "f"(v00), "f"(v01));
        asm("max.f32 %0, %1, %2;" : "=f"(max_val) : "f"(max_val), "f"(v10));
        asm("max.f32 %0, %1, %2;" : "=f"(max_val) : "f"(max_val), "f"(v11));
    } else {
        // 通用版本
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int idx = ((n * C + c) * H_in + (h0 + kh)) * W_in + (w0 + kw);
                float val;
                asm("ld.global.f32 %0, [%1];" : "=f"(val) : "l"(in + idx));
                asm("max.f32 %0, %1, %2;" : "=f"(max_val) : "f"(max_val), "f"(val));
            }
        }
    }

    int idx_out = ((n * C + c) * H_out + h_out) * W_out + w_out;
    asm("st.global.f32 [%0], %1;" :: "l"(out + idx_out), "f"(max_val));
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

// 优化IF神经元核函数
__global__ void ifnode_forward_optimized(
    const float* __restrict__ in,
    float* __restrict__ v,
    float* __restrict__ out,
    int N,
    float threshold
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 使用向量化处理
    const int VEC_SIZE = 4;
    int vec_i = i * VEC_SIZE;
    
    if (vec_i + VEC_SIZE - 1 < N) {
        // 向量化加载
        float4 in_vec = reinterpret_cast<const float4*>(in + vec_i)[0];
        float4 v_vec = reinterpret_cast<const float4*>(v + vec_i)[0];
        float4 out_vec;
        
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            float* in_ptr = reinterpret_cast<float*>(&in_vec) + j;
            float* v_ptr = reinterpret_cast<float*>(&v_vec) + j;
            float* out_ptr = reinterpret_cast<float*>(&out_vec) + j;
            
            float vi = *v_ptr + *in_ptr;
            if (vi >= threshold) {
                *out_ptr = 1.0f;
                *v_ptr = 0.0f;
            } else {
                *out_ptr = 0.0f;
                *v_ptr = vi;
            }
        }
        
        // 向量化存储
        reinterpret_cast<float4*>(out + vec_i)[0] = out_vec;
        reinterpret_cast<float4*>(v + vec_i)[0] = v_vec;
    } else {
        // 处理剩余元素
        for (int j = 0; j < VEC_SIZE && vec_i + j < N; ++j) {
            int idx = vec_i + j;
            float vi = v[idx] + in[idx];
            if (vi >= threshold) {
                out[idx] = 1.0f;
                v[idx] = 0.0f;
            } else {
                out[idx] = 0.0f;
                v[idx] = vi;
            }
        }
    }
}

// in:  [N, C, H_in, W_in]
// out: [N, C, H_out, W_out]
// H_out = (H_in - K) / stride + 1
// W_out = (W_in - K) / stride + 1
__global__ void maxpool2d_forward_batch_fast(
    const float* __restrict__ in,
    float* __restrict__ out,
    int N, int C,
    int H_in, int W_in,
    int K, int stride
){
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int nc    = blockIdx.z;   // 合并 N 和 C

    int H_out = (H_in - K) / stride + 1;
    int W_out = (W_in - K) / stride + 1;

    if (w_out >= W_out || h_out >= H_out || nc >= N * C) return;

    int n = nc / C;
    int c = nc % C;

    int h0 = h_out * stride;
    int w0 = w_out * stride;

    // 输入 feature map 的这一块左上角在 global 内存中的 index
    int idx_base_in = ((n * C + c) * H_in + h0) * W_in + w0;

    float max_val = -1e30f;

    if (K == 2 && stride == 2) {
        // 为 K=2, stride=2 做展开优化（你的 LeNet 就是这种）
        float v00 = in[idx_base_in];
        float v01 = in[idx_base_in + 1];
        float v10 = in[idx_base_in + W_in];
        float v11 = in[idx_base_in + W_in + 1];
        max_val = fmaxf(fmaxf(v00, v01), fmaxf(v10, v11));
    } else {
        // 通用版本（应对其他 K/stride）
        for (int kh = 0; kh < K; ++kh) {
            int h = h0 + kh;
            int row_base = ((n * C + c) * H_in + h) * W_in;
            for (int kw = 0; kw < K; ++kw) {
                int w = w0 + kw;
                float v = in[row_base + w];
                if (v > max_val) max_val = v;
            }
        }
    }

    int idx_out = ((n * C + c) * H_out + h_out) * W_out + w_out;
    out[idx_out] = max_val;
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

// 优化版本：使用向量化加载和更好的内存访问模式
__global__ void fc_if_forward_batch_optimized(
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
    const int TILE_SIZE = 4; // 使用float4向量化
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_features;
    if (idx >= total) return;

    int n = idx / out_features;
    int o = idx % out_features;

    const float* x_row = x + n * in_features;
    const float* w_row = W + o * in_features;

    // 使用向量化累加
    float sum = b[o];
    
    // 主循环：向量化处理
    int j = 0;
    for (; j <= in_features - TILE_SIZE; j += TILE_SIZE) {
        float4 x_vec = reinterpret_cast<const float4*>(x_row + j)[0];
        float4 w_vec = reinterpret_cast<const float4*>(w_row + j)[0];
        
        sum += x_vec.x * w_vec.x;
        sum += x_vec.y * w_vec.y;
        sum += x_vec.z * w_vec.z;
        sum += x_vec.w * w_vec.w;
    }
    
    // 处理剩余元素
    for (; j < in_features; ++j) {
        sum += w_row[j] * x_row[j];
    }

    // IF神经元逻辑
    float vi = v[idx] + sum;
    if (vi >= threshold) {
        out[idx] = 1.0f;
        v[idx] = 0.0f;
    } else {
        out[idx] = 0.0f;
        v[idx] = vi;
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

// 融合卷积和IF神经元的核函数
template<
    int C_IN,
    int C_OUT,
    int K,
    int STRIDE,
    int PADDING,
    int BLOCK_H,
    int BLOCK_W
>
__device__ void conv_if_shared_core(
    const float* __restrict__ in,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ v,          // 膜电位（需要跨时间步保留）
    float* __restrict__ out,        // 脉冲输出 0/1
    float* __restrict__ s_in,       // shared memory: [C_IN, TILE_H, TILE_W]
    int N,
    int H_in,
    int W_in,
    float threshold                 // IF神经元阈值
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

    // 卷积计算
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

    // -------- 3. IF神经元计算 --------
    int idx_out = ((n * C_OUT + co) * H_out + h_out) * W_out + w_out;
    int idx_v = idx_out;  // 膜电位与输出同形状
    
    float vi = v[idx_v] + sum;  // 更新膜电位
    
    if (vi >= threshold) {
        out[idx_out] = 1.0f;    // 产生脉冲
        v[idx_v] = 0.0f;        // 重置膜电位
    } else {
        out[idx_out] = 0.0f;
        v[idx_v] = vi;          // 保持膜电位
    }
}

// 融合的conv2+IF2核函数
template<int BLOCK_H, int BLOCK_W>
__global__ void conv2_if2_forward_shared_const(
    const float* __restrict__ in,
    float* __restrict__ v,          // IF2膜电位
    float* __restrict__ out,        // IF2输出
    int N,
    int H_in,
    int W_in,
    float threshold
){
    // shared memory: [C_IN=6, TILE_H, TILE_W]
    extern __shared__ float s_mem[];

    conv_if_shared_core<
        6, 16,       // C_IN, C_OUT
        5, 1, 0,     // K, STRIDE, PADDING
        BLOCK_H, BLOCK_W
    >(in,
      d_conv2_w_const,
      d_conv2_b_const,
      v,
      out,
      s_mem,
      N, H_in, W_in,
      threshold);
}

// 优化版本：减少bank冲突，更好的shared memory访问
template<int BLOCK_H, int BLOCK_W>
__global__ void conv2_if2_forward_shared_const_optimized(
    const float* __restrict__ in,
    float* __restrict__ v,
    float* __restrict__ out,
    int N,
    int H_in,
    int W_in,
    float threshold
){
    extern __shared__ float s_mem[];
    
    constexpr int C_IN = 6, C_OUT = 16, K = 5;
    constexpr int TILE_H = BLOCK_H + K - 1;
    constexpr int TILE_W = BLOCK_W + K - 1;
    
    // 重新组织shared memory布局减少bank冲突
    #define S_IN(ci, h, w) s_mem[(ci) * (TILE_H * TILE_W) + (h) * TILE_W + (w)]
    
    int out_w0 = blockIdx.x * BLOCK_W;
    int out_h0 = blockIdx.y * BLOCK_H;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int co_n = blockIdx.z;
    int n = co_n / C_OUT;
    int co = co_n % C_OUT;
    
    if (n >= N) return;
    
    int H_out = (H_in - K) / 1 + 1;
    int W_out = (W_in - K) / 1 + 1;
    
    // 1. 优化shared memory加载：使用向量化
    for (int ci = 0; ci < C_IN; ++ci) {
        for (int th = ty; th < TILE_H; th += BLOCK_H) {
            int h_in = out_h0 + th;
            for (int tw = tx; tw < TILE_W; tw += BLOCK_W) {
                int w_in = out_w0 + tw;
                
                float val = 0.0f;
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    int idx_in = ((n * C_IN + ci) * H_in + h_in) * W_in + w_in;
                    val = in[idx_in];
                }
                S_IN(ci, th, tw) = val;
            }
        }
    }
    
    __syncthreads();
    
    // 2. 卷积计算 + IF
    int h_out = out_h0 + ty;
    int w_out = out_w0 + tx;
    
    if (h_out < H_out && w_out < W_out) {
        float sum = d_conv2_b_const[co];
        int w_base_co = co * (C_IN * K * K);
        
        // 展开内层循环
        #pragma unroll
        for (int ci = 0; ci < C_IN; ++ci) {
            int w_base_ci = w_base_co + ci * (K * K);
            
            #pragma unroll
            for (int kh = 0; kh < K; ++kh) {
                int th = ty + kh;
                
                #pragma unroll  
                for (int kw = 0; kw < K; ++kw) {
                    int tw = tx + kw;
                    
                    float vin = S_IN(ci, th, tw);
                    int idx_w = w_base_ci + kh * K + kw;
                    float ww = d_conv2_w_const[idx_w];
                    
                    sum += vin * ww;
                }
            }
        }
        
        // IF神经元
        int idx_out = ((n * C_OUT + co) * H_out + h_out) * W_out + w_out;
        float vi = v[idx_out] + sum;
        
        if (vi >= threshold) {
            out[idx_out] = 1.0f;
            v[idx_out] = 0.0f;
        } else {
            out[idx_out] = 0.0f;
            v[idx_out] = vi;
        }
    }
    
    #undef S_IN
}

void process_batch(
    cudaStream_t stream,
    const float* d_input, int cur_batch,
    float* d_conv1_out, float* d_if1_v, float* d_if1_out,
    float* d_pool1_out, float* d_if2_v, float* d_if2_out, 
    float* d_pool2_out, float* d_if3_v, float* d_if3_out,
    float* d_if4_v, float* d_if4_out, float* d_logits_sum,
    float* d_fc1_w,   float* d_fc1_b,   float* d_fc2_w,   float* d_fc2_b,
    float* d_fc3_w,   float* d_fc3_b)
{
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
            ifnode_forward_ptx<<<blocks, THREADS>>>(
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
            dim3 block(16, 16);  // 每个 block 覆盖 16x16 个 (h_out, w_out)
            dim3 grid(
                (P1_W + block.x - 1) / block.x,  // P1_W = 12
                (P1_H + block.y - 1) / block.y,  // P1_H = 12
                cur_batch * C1_OUT_C             // 合并 N 和 C
            );

            maxpool2d_forward_batch_ptx<<<grid, block>>>(
                d_if1_out,    // in:  [cur_batch, 6, 24, 24]
                d_pool1_out,  // out: [cur_batch, 6, 12, 12]
                cur_batch, C1_OUT_C,
                C1_H, C1_W,
                P1_K, P1_STR  // K=2, stride=2
            );
            checkCudaErrors(cudaGetLastError());
        }

        // (4+5) 融合 conv2 + IF2: [6,12,12] -> [16,8,8] -> IF脉冲
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

            conv2_if2_forward_shared_const_optimized<BLOCK_H, BLOCK_W>
                <<<grid, block, shared_bytes>>>(
                    d_pool1_out,   // [cur_batch,6,12,12]
                    d_if2_v,       // IF2膜电位
                    d_if2_out,     // IF2输出脉冲
                    cur_batch,
                    P1_H, P1_W,    // 12, 12
                    1.0f           // 阈值
                );
            checkCudaErrors(cudaGetLastError());
        }

        // (6) pool2: [16,8,8] -> [16,4,4]
        {
            dim3 block(16, 16);
            dim3 grid(
                (P2_W + block.x - 1) / block.x,  // P2_W = 4
                (P2_H + block.y - 1) / block.y,  // P2_H = 4
                cur_batch * C2_OUT_C             // 合并 N 和 C
            );

            maxpool2d_forward_batch_ptx<<<grid, block>>>(
                d_if2_out,    // in:  [cur_batch, 16, 8, 8]
                d_pool2_out,  // out: [cur_batch, 16, 4, 4]
                cur_batch, C2_OUT_C,
                C2_H, C2_W,   // H_in=8, W_in=8
                P2_K, P2_STR  // K=2, stride=2
            );
            checkCudaErrors(cudaGetLastError());
        }

        // (8) fc1 + IF3: [N,256] -> [N,120] -> spike
        {
            int total = cur_batch * FC1_OUT;
            int blocks_fc1 = (total + THREADS - 1) / THREADS;
            fc_if_forward_batch_optimized<<<blocks_fc1, THREADS>>>(
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
            fc_if_forward_batch_optimized<<<blocks_fc2, THREADS>>>(
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
}