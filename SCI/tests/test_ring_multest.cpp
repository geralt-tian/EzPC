/*
Authors: Deevashwer Rathee
Copyright:
Copyright (c) 2021 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "LinearOT/linear-ot.h"
#include "utils/emp-tool.h"
#include <iostream>

#include "FloatingPoint/floating-point.h"
#include "FloatingPoint/fp-math.h"
#include <random>
#include <limits>
#include "float_utils.h"

#include "Millionaire/millionaire.h"
#include "Millionaire/millionaire_with_equality.h"
#include "BuildingBlocks/truncation.h"
#include "BuildingBlocks/aux-protocols.h"
#include <chrono>
using namespace sci;
using namespace std;

int party, port = 32000;
string address = "127.0.0.1";
IOPack *iopack;
OTPack *otpack;
LinearOT *prod;

// 全局变量
int dim1 = 1;
int dim2 = 1;
int dim3 = 1;
int bwA = 22;
int bwB = 15;
int bwC = bwA + bwB; // 矩阵位宽
uint64_t mask_bwC = (bwC == 64 ? -1 : ((1ULL << bwC) - 1));
bool signed_B = true;           // 表示矩阵B是否为有符号数
bool accumulate = true;         // 决定是否累加结果
bool precomputed_MSBs = false;  // 决定是否预计算最高有效位
MultMode mode = MultMode::None; // 乘法模式

/////////////////////compare_with_eq
int bitlength = 64; // 假设每个数是32位
int radix_base = 4; // 基数为4
uint64_t alpha = 4;
uint64_t alpha_ = 1ULL << 14;
uint64_t beta = alpha_ * 2;
uint64_t h = 15;

Truncation *trunc_oracle;
//////////////////////////MUX
AuxProtocols *aux;

void assign_lower_h_bits(int32_t dim1, int32_t dim2, int32_t dim3, uint64_t *inA, uint64_t *inB, uint64_t *inA_, uint64_t *inB_, int32_t h)
{
    // Create a mask that has the lowest h bits set to 1
    uint64_t mask = (h == 64) ? ~0ULL : (1ULL << h) - 1;

    // Assign the lower h bits from inA to inA_
    for (int i = 0; i < dim1; i++)
    {
        for (int j = 0; j < dim2; j++)
        {
            inA_[i * dim2 + j] = inA[i * dim2 + j] & mask;
        }
    }

    // Assign the lower h bits from inB to inB_
    for (int i = 0; i < dim2; i++)
    {
        for (int j = 0; j < dim3; j++)
        {
            inB_[i * dim3 + j] = inB[i * dim3 + j] & mask;
        }
    }
}

//////////////////////
// 初始化
///////////////////////////////

int main(int argc, char **argv)
{
    ArgMapping amap;
    amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
    amap.arg("p", port, "Port Number");
    amap.arg("ip", address, "IP Address of server (ALICE)");
    amap.arg("m", precomputed_MSBs, "MSB_to_Wrap Optimization?");
    amap.arg("a", ::accumulate, "Accumulate?");

    amap.parse(argc, argv);

    iopack = new IOPack(party, port, "127.0.0.1");
    otpack = new OTPack(iopack, party);

    uint64_t comm_start = iopack->get_comm();
    auto time_start = chrono::high_resolution_clock::now();

    prod = new LinearOT(party, iopack, otpack);

    PRG128 prg; //(fix_key);

    uint64_t *inA = new uint64_t[dim1 * dim2]; // 1*100
    uint64_t *inB = new uint64_t[dim2 * dim3]; // 100*35
    int dim = (::accumulate ? dim1 * dim3 : dim1 * dim2 * dim3);
    uint64_t *outax = new uint64_t[dim];
    // 使用PRG128分配并初始化随机矩阵inA和inB
    // prg.random_data(inA, dim1 * dim2 * sizeof(uint64_t));
    // prg.random_data(inB, dim2 * dim3 * sizeof(uint64_t));

    inA[0] = 20000;
    inB[0] = 1;
    std::cout << "input inA[" << 0 << "] = " << inA[0] << std::endl;
    std::cout << "input inB[" << 0 << "] = " << inB[0] << std::endl;
    /////////////step2 //check
    //   inA[0] = inA[0]+alpha_;
    //   inB[0] = inB[0]+alpha_;

    uint64_t *inA_ = new uint64_t[dim1 * dim2]; // 1*100
    uint64_t *inB_ = new uint64_t[dim2 * dim3]; // 100*35

    inA_[0] = (inA[0] + alpha_) & mask_bwC;
    inB_[0] = (inB[0] + alpha_) & mask_bwC;
    // step5 //check
    uint64_t *inA_h = new uint64_t[dim1 * dim2]; // 1*100
    uint64_t *inB_h = new uint64_t[dim2 * dim3]; // 100*35
    assign_lower_h_bits(dim1, dim2, dim3, inA_, inB_, inA_h, inB_h, h);
    std::cout<< "=========STEP5 extract the lower h bits==========="<<std::endl;
    std::cout << "inA_h[" << 0 << "] = " << inA_h[0] << std::endl;
    std::cout << "inB_h[" << 0 << "] = " << inB_h[0] << std::endl;

    // step6 check
    uint64_t comm_start_tr = iopack->get_comm();
    trunc_oracle = new Truncation(party, iopack, otpack);
    uint64_t *outtrunc = new uint64_t[dim];
    if (party == sci::ALICE)
    {
        trunc_oracle->truncate_and_reduce(dim, inA_h, outtrunc, 7, 15); // shift=h-s,hypothesis s=8
        //std::cout << "outtrunc[" << 0 << "] = " << outtrunc[0] << std::endl;
    }
    else
    {
        trunc_oracle->truncate_and_reduce(dim, inB_h, outtrunc, 7, 15);      // shift=h-s,hypothesis s=8,outtrunc是0-255
        //std::cout << "outtrunc[" << 0 << "] = " << outtrunc[0] << std::endl; // outtrunc是<i>，范围是0-255
    }
    std::cout<<"=========STEP6 Truncate_reduce==========="<<std::endl;
    std::cout << std::dec << "outtrunc = " << outtrunc[0] << std::endl;
    uint64_t comm_end_tr = iopack->get_comm();
    std::cout << "TR Bytes Sent: " << (comm_end_tr - comm_start_tr) << "bytes" << std::endl;
    // step7 check
    uint64_t comm_start_lut = iopack->get_comm();
    std::cout<<"=========STEP7 LookUp Table   ==========="<<std::endl;
    // 重跑一个有256个的
    std::vector<std::vector<uint64_t>> data = {
        {4194303, 4194302, 4194302, 4194302, 4194302, 4194301, 4194301, 4194300,
         4194300, 4194299, 4194299, 4194298, 4194297, 4194297, 4194296, 4194295,
         4194294, 4194292, 4194291, 4194290, 4194288, 4194286, 4194285, 4194282,
         4194280, 4194278, 4194275, 4194272, 4194269, 4194266, 4194263, 4194259,
         4194255, 4194250, 4194246, 4194241, 4194236, 4194230, 4194224, 4194218,
         4194211, 4194204, 4194197, 4194189, 4194181, 4194172, 4194163, 4194154,
         4194144, 4194134, 4194124, 4194113, 4194101, 4194090, 4194078, 4194066,
         4194053, 4194040, 4194027, 4194014, 4194000, 4193987, 4193973, 4193959,
         4193945, 4193931, 4193918, 4193904, 4193891, 4193878, 4193865, 4193853,
         4193842, 4193831, 4193820, 4193811, 4193802, 4193795, 4193788, 4193783,
         4193779, 4193777, 4193776, 4193776, 4193779, 4193783, 4193789, 4193797,
         4193807, 4193820, 4193835, 4193852, 4193872, 4193895, 4193920, 4193948,
         4193979, 4194013, 4194050, 4194090, 4194133, 4194179, 4194228, 4194280,
         31, 90, 151, 216, 283, 354, 427, 503, 582, 663, 747, 834, 922, 1013,
         1105, 1199, 1295, 1393, 1491, 1591, 1692, 1793, 1895, 1997, 2099, 2201,
         2303, 2404, 2505, 2605, 2703, 2801, 2897, 2991, 3083, 3174, 3262, 3349,
         3433, 3514, 3593, 3669, 3742, 3813, 3880, 3945, 4006, 4065, 4120, 4172,
         4221, 4267, 4310, 4350, 4387, 4421, 4452, 4480, 4505, 4528, 4548, 4565,
         4580, 4593, 4603, 4611, 4617, 4621, 4624, 4624, 4623, 4621, 4617, 4612,
         4605, 4598, 4589, 4580, 4569, 4558, 4547, 4535, 4522, 4509, 4496, 4482,
         4469, 4455, 4441, 4427, 4413, 4400, 4386, 4373, 4360, 4347, 4334, 4322,
         4310, 4299, 4287, 4276, 4266, 4256, 4246, 4237, 4228, 4219, 4211, 4203,
         4196, 4189, 4182, 4176, 4170, 4164, 4159, 4154, 4150, 4145, 4141, 4137,
         4134, 4131, 4128, 4125, 4122, 4120, 4118, 4115, 4114, 4112, 4110, 4109,
         4108, 4106, 4105, 4104, 4103, 4103, 4102, 4101, 4101, 4100, 4100, 4099,
         4099, 4098, 4098, 4098, 4098, 4097},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 10,
         11, 13, 15, 18, 20, 23, 27, 30, 34, 39, 44, 49, 55, 61, 68, 76, 84, 93, 103,
         114, 125, 137, 150, 163, 178, 193, 210, 227, 245, 265, 285, 306, 328, 351,
         375, 399, 424, 450, 477, 504, 532, 560, 588, 616, 644, 672, 700, 727, 753,
         778, 801, 823, 844, 862, 878, 891, 900, 907, 910, 908, 902, 891, 874, 852,
         824, 789, 747, 697, 640, 574, 500, 416, 324, 221, 108, 137438953457,
         137438953323, 137438953178, 137438953021, 137438952853, 137438952673,
         137438952482, 137438952278, 137438952062, 137438951834, 137438951594,
         137438951342, 137438951078, 137438950803, 137438950515, 137438950216,
         137438949906, 137438949586, 137438949255, 137438948913, 137438948563,
         137438948203, 137438947835, 137438947459, 137438947076, 137438946686,
         137438946290, 137438945889, 137438945484, 137438945075, 137438944664,
         137438944250, 137438943836, 137438943421, 137438943006, 137438942593,
         137438942182, 137438941775, 137438941371, 137438940972, 137438940578,
         137438940191, 137438939811, 137438939439, 137438939075, 137438938720,
         137438938376, 137438938041, 137438937717, 137438937405, 137438937105,
         137438936816, 137438936541, 137438936278, 137438936028, 137438935792,
         137438935569, 137438935360, 137438935164, 137438934982, 137438934814,
         137438934659, 137438934518, 137438934390, 137438934275, 137438934172,
         137438934083, 137438934005, 137438933939, 137438933885, 137438933842,
         137438933809, 137438933787, 137438933774, 137438933771, 137438933777,
         137438933791, 137438933812, 137438933841, 137438933877, 137438933920,
         137438933968, 137438934021, 137438934080, 137438934142, 137438934209,
         137438934279, 137438934352, 137438934428, 137438934506, 137438934585,
         137438934667, 137438934749, 137438934832, 137438934915, 137438934998,
         137438935082, 137438935164, 137438935246, 137438935328, 137438935408,
         137438935486, 137438935563, 137438935639, 137438935713, 137438935785,
         137438935854, 137438935922, 137438935988, 137438936051, 137438936112,
         137438936171, 137438936227, 137438936281, 137438936333, 137438936383,
         137438936430, 137438936475, 137438936517, 137438936558, 137438936596,
         137438936632, 137438936666, 137438936698, 137438936729, 137438936757,
         137438936784, 137438936808, 137438936832, 137438936853, 137438936873,
         137438936892, 137438936909, 137438936925, 137438936940, 137438936954,
         137438936967, 137438936978, 137438936989, 137438937008, 137438937016,
         137438937023, 137438937030, 137438937036, 137438937042, 137438937047,
         137438937051, 137438937055, 137438937059, 137438937062, 137438937065,
         137438937068, 137438937070, 137438937073, 137438937074, 137438937076}};
    int32_t T_size = sizeof(uint64_t) * 8;
    int bw_xlut = 8;
    int bw_ylut;
    aux = new AuxProtocols(party, iopack, otpack);
    if (T_size == 8)
        bw_ylut = 7;
    else
        bw_ylut = 29;
    uint64_t *a_alice = new uint64_t[dim];
    uint64_t *b_alice = new uint64_t[dim];
    a_alice[0] = 0;
    b_alice[0] = 0;
    uint64_t **spec_a = new uint64_t *[dim];
    uint64_t *a_bob = new uint64_t[dim];
    uint64_t N = 1ULL << bw_xlut;

    for (int i = 0; i < dim; i++)
    {
        spec_a[i] = new uint64_t[N];
        for (int j = 0; j < N; j++)
        {
            spec_a[i][j] = data[i][j];
            // std::cout << "i = " << i << ", j = " << j << ", data = " << data[j][i] << std::endl;
        }
    }
    // for (int i = 0; i < dim; ++i) {
    //     std::cout << "spec[" << i << "]: ";
    //     for (int j = 0; j < N; ++j) {
    //         std::cout << spec[i][j] << " "; // Output the value in decimal format
    //     }
    //     std::cout << std::endl; // Move to the next line after each row
    // }
    uint64_t *outtrunc1 = new uint64_t[dim];
    for (size_t i = 0; i < dim; i++)
    {
        outtrunc1[0]=0;
    }
    
    uint64_t *outtrunc_a = new uint64_t[dim];
        if (party == ALICE)
    {
        iopack->io->send_data(outtrunc, dim * sizeof(uint64_t));
    }
    else
    {                                                                            // party == BOB
        iopack->io->recv_data(outtrunc1, dim * sizeof(uint64_t));
        outtrunc_a[0] = (outtrunc[0] + outtrunc1[0]) & ((1ULL<<8) - 1);
        std::cout << "outtrunc_a[" << 0 << "] = " << outtrunc_a[0] << std::endl;
    }
    if (party == ALICE)
    {
        aux->lookup_table<uint64_t>(spec_a, nullptr, nullptr, dim, bw_xlut, 22); // bw_xlut是outtrunc的位宽
    }
    else
    {                                                                            // party == BOB
        aux->lookup_table<uint64_t>(nullptr, outtrunc_a, a_bob, dim, bw_xlut, 22); // a_bob是查询到的斜率
    }
    if (party != ALICE)
        std::cout << "a_bob[" << 0 << "] = " << a_bob[0] << std::endl;

    /////选择截距
    uint64_t **spec_b = new uint64_t *[dim];
    uint64_t *b_bob = new uint64_t[dim];

    for (int i = 0; i < dim; i++)
    {
        spec_b[i] = new uint64_t[N];
        for (int j = 0; j < N; j++)
        {
            spec_b[i][j] = data[i + 1][j];
            // std::cout << "i = " << i << ", j = " << j << ", data = " << data[j][i+1] << std::endl;
        }
    }
    // for (int i = 0; i < dim; ++i) {
    //     std::cout << "spec[" << i << "]: ";
    //     for (int j = 0; j < N; ++j) {
    //         std::cout << spec[i][j] << " "; // Output the value in decimal format
    //     }
    //     std::cout << std::endl; // Move to the next line after each row
    // }
    // for (int i; i < dim; i++)
    // {
    //     outtrunc[i] = outtrunc[i] - 16384;
    // }
    if (party == ALICE)
    {
        aux->lookup_table<uint64_t>(spec_b, nullptr, nullptr, dim, bw_xlut, 37);
    }
    else
    {                                                                            // party == BOB
        aux->lookup_table<uint64_t>(nullptr, outtrunc_a, b_bob, dim, bw_xlut, 37); // b_bob是查询到的截距  重要问题，这里的outtrunc应该是两边share加起来，代码里只有Bob的outtrunc check
    }
    if (party != ALICE)
        std::cout << "b_bob[" << 0 << "] = " << b_bob[0] << std::endl;

    uint64_t comm_end_lut = iopack->get_comm();
    cout << "LUT Bytes Sent: " << (comm_end_lut - comm_start_lut) << "bytes" << endl;
    //////////////////////step8
    std::cout<<"=========STEP8 matrix_multiplication   ==========="<<std::endl;
    uint64_t comm_start_mult = iopack->get_comm();
    uint8_t *msbA = nullptr;
    uint8_t *msbB = nullptr;
    if (precomputed_MSBs)
    { // 预计算MSB
        msbA = new uint8_t[dim1 * dim2];
        msbB = new uint8_t[dim2 * dim3];
        prod->aux->MSB(inA, msbA, dim1 * dim2, bwA);
        prod->aux->MSB(inB, msbB, dim2 * dim3, bwB);
    }

    //   test_matrix_multiplication(inA, inB, outC, false);
    // test_matrix_multiplication(inA, inB, outC, true);

    if (party == ALICE)
    {
        prod->matrix_multiplication(dim1, dim2, dim3, inA_h, a_alice, outax, bwA, bwB, bwC,
                                    true, signed_B, ::accumulate, mode,
                                    msbA, msbB);
    }
    else
    {
        prod->matrix_multiplication(dim1, dim2, dim3, inB_h, a_bob, outax, bwA, bwB, bwC,
                                    true, signed_B, ::accumulate, mode,
                                    msbA, msbB);
    }
    /////////////////////////
    for (int i = 0; i < dim1 * dim3; i++)
    {
        std::cout << "step8 outax[" << i << "] =" << outax[i] << std::endl;
    }

    uint64_t comm_end_mult = iopack->get_comm();
    cout << "Mult Bytes Sent: " << (comm_end_mult - comm_start_mult) << "bytes" << endl;
    /////////////////////////////step9
    std::cout<<"=========STEP9 ADDITION <z>  ==========="<<std::endl;
    uint64_t *z = new uint64_t[dim];
    if (party == ALICE)
    {
        z[0] = ((outax[0] + b_alice[0]) & mask_bwC);
    }
    else
    {
        z[0] = (outax[0] + b_bob[0]) & mask_bwC;
    }
    std::cout << "z[" << 0 << "] = " << z[0] << std::endl;
    //////////////////////MillionaireWithEquality step10
    std::cout<<"=========STEP10 CMP   ==========="<<std::endl;

    uint64_t C = static_cast<int64_t>(1ULL << 20); // 2^62 两边都加一个很大的数防止做减法的时候到负数溢出

    // 比较两个数
    // ALICE 和 BOB 分别输入自己的数据
    uint64_t *local_data1 = new uint64_t[dim2 * dim3]; // 100*35
    if (party == ALICE)
    {
        for (int i = 0; i < dim; i++)
            local_data1[i] = inA_[i];
    }
    else
    {
        for (int i = 0; i < dim; i++)
            local_data1[i] = ((1ULL << 15) - inB_[i]) & mask_bwC; // 得设置环操作
    }

    // 调用 compare_with_eq 函数进行比较

    uint8_t *msb = new uint8_t[dim];
    // 参与方分别传入自己的数据
    uint64_t comm_start_msb = iopack->get_comm();
    prod->aux->MSB(local_data1, msb, dim1 * dim2, bwC);//重要问题，有符号数的第一位是1是负数，直接用MSB比较是有问题的，可以先比较符号位，然后再用MSB
    uint64_t comm_end_msb = iopack->get_comm();
    // millionaire.compare_with_eq(res_cmp_b, res_eq_b, local_data1, 1, bitlength, true, radix_base);//line10  这里生成的可能是结果的share，真tm是
    std::cout << "local_data1[0]=" << local_data1[0] << std::endl;
    std::cout << "msb[0] = " << static_cast<int>(msb[0]) << std::endl;
    // std::cout << "res_cmp_b[0] = " << static_cast<int>(res_eq_b[0]) << std::endl;
    //  输出比较结果

    cout << "MSB Bytes Sent: " << (comm_end_msb - comm_start_msb) << "bytes" << endl;

    ///////////step11
    std::cout << inB_[0] << std::endl;

    if (party == ALICE)
    {
        for (int i = 0; i < dim; i++)
            local_data1[i] = inA_[i] & mask_bwC;
    }
    else
    {
        for (int i = 0; i < dim; i++)
            local_data1[i] = (-inB_[i]) & mask_bwC;
        ; // 设置环操作
    }

    for (int i = 0; i < dim; i++)
        local_data1[i] = -inB_[i] & mask_bwC;
    std::cout << local_data1[0] << std::endl;

    uint8_t *msb_ = new uint8_t[dim];
    uint64_t comm_start_cmp = iopack->get_comm();
    prod->aux->MSB(local_data1, msb_, dim1 * dim2, bwC);

    std::cout << "msb_[0] = " << static_cast<int>(msb_[0]) << std::endl;

    // uint64_t comm_end_cmp = iopack->get_comm();
    // cout << "CMP Bytes Sent: " << (comm_end_cmp - comm_start_cmp) <<"bytes"<< endl;

    ////////////////////////////////step12
    std::cout<<"=========STEP12 MUX   ==========="<<std::endl;
    // aux = new AuxProtocols(party, iopack, otpack);
    uint64_t comm_start_mux = iopack->get_comm();
    uint8_t *MUX_sel = new uint8_t[dim1];
    int bw_x = 37, bw_y = 37;

    uint64_t *MUX_data1 = new uint64_t[dim1];
    uint64_t *MUX_output_u = new uint64_t[dim1];
    MUX_output_u[0] = 0;

    MUX_sel[0] = msb_[0] ^ msb[0];
    // MUX_data1[0] = z[0];
    MUX_data1[0] = z[0];

    std::cout << "MUX_sel[" << 0 << "] = " << static_cast<int>(MUX_sel[0]) << std::endl;
    std::cout << "MUX_data1[" << 0 << "] =" << MUX_data1[0] << std::endl;

    aux->multiplexer(MUX_sel, MUX_data1, MUX_output_u, dim1, bw_x, bw_y);
    std::cout << "MUX_output_u[" << 0 << "] =" << MUX_output_u[0] << std::endl;

    uint64_t comm_end_mux = iopack->get_comm();
    std::cout << "MUX Bytes Sent: " << (comm_end_mux - comm_start_mux) << "bytes" << std::endl;

    /////////step13
    uint64_t *MUX_output_v = new uint64_t[dim1];
    MUX_output_v[0] = 0;
    if (party == ALICE)
    {
        MUX_sel[0] = msb[0];
        aux->multiplexer(MUX_sel, inA, MUX_output_v, dim1, bw_x, bw_y);
        std::cout << "MUX_output_v[" << 0 << "] =" << MUX_output_v[0] << std::endl;
    }
    else
    {
        MUX_sel[0] = msb[0];
        aux->multiplexer(MUX_sel, inB, MUX_output_v, dim1, bw_x, bw_y);
        std::cout << "MUX_output_v[" << 0 << "] =" << MUX_output_v[0] << std::endl;
    }

    //////////step14

    ///////////输出时间和通信
    uint64_t comm_end = iopack->get_comm();
    cout << "Tptal Bytes Sent: " << (comm_end - comm_start) << "bytes" << endl;

    auto time_end = chrono::high_resolution_clock::now();
    cout << "Total time: "
         << chrono::duration_cast<chrono::milliseconds>(time_end - time_start).count()
         << " ms" << endl;
    ///////////////////////////////check
    if (party == ALICE)
    {
        iopack->io->send_data(MUX_output_u, dim * sizeof(uint64_t));
        iopack->io->send_data(MUX_output_v, dim * sizeof(uint64_t));
    }
    else
    {
        uint64_t *MUX_rec_u = new uint64_t[dim];
        uint64_t *MUX_rec_v = new uint64_t[dim];
        iopack->io->recv_data(MUX_rec_u, dim * sizeof(uint64_t));
        iopack->io->recv_data(MUX_rec_v, dim * sizeof(uint64_t));
        uint64_t result = (MUX_rec_v[0] + MUX_rec_u[0] + MUX_output_u[0] + MUX_output_v[0]) & mask_bwC;

        // The result is automatically modulo 2^64 because of uint64_t
        std::cout << "The input is: " << (inA[0]+inB[0]) << std::endl;
        std::cout << "The result mod 2^37 is: " << result << std::endl;
        std::cout << "for input from (0-32,767) output shuould be : " << (((inA[0]+inB[0])*a_bob[0]+b_bob[0])& mask_bwC) << std::endl;
    }

    ////////////////////////
    delete[] inA_;
    delete[] inB_;
    delete[] inA;
    delete[] inB;
    delete[] outax;
    delete prod;
}
