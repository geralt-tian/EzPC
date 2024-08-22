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
using namespace sci;
using namespace std;

int party, port = 32000;
string address = "127.0.0.1";
IOPack *iopack;
OTPack *otpack;
LinearOT *prod;


//全局变量
int dim1 = 1;
int dim2 = 1;
int dim3 = 1;
int bwA = 22;
int bwB = 15;
int bwC = bwA + bwB;//矩阵位宽
bool signed_B = true;//表示矩阵B是否为有符号数
bool accumulate = true;//决定是否累加结果
bool precomputed_MSBs = false;//决定是否预计算最高有效位
MultMode mode = MultMode::None;//乘法模式

uint64_t maskA = (bwA == 64 ? -1 : ((1ULL << bwA) - 1));//限制矩阵条目的位宽
uint64_t maskB = (bwB == 64 ? -1 : ((1ULL << bwB) - 1));
uint64_t maskC = (bwC == 64 ? -1 : ((1ULL << bwC) - 1));
/////////////////////compare_with_eq
int bitlength = 64; // 假设每个数是32位
int radix_base = 4; // 基数为4
uint64_t alpha=4;
uint64_t alpha_=2^14;
uint64_t beta=alpha_*2;
uint64_t h=15;

Truncation *trunc_oracle;
//////////////////////////MUX
AuxProtocols *aux;


void assign_lower_h_bits(int32_t dim1, int32_t dim2, int32_t dim3, uint64_t *inA, uint64_t *inB, uint64_t *inA_, uint64_t *inB_, int32_t h) {
    // Create a mask that has the lowest h bits set to 1
    uint64_t mask = (h == 64) ? ~0ULL : (1ULL << h) - 1;

    // Assign the lower h bits from inA to inA_
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            inA_[i * dim2 + j] = inA[i * dim2 + j] & mask;
        }
    }

    // Assign the lower h bits from inB to inB_
    for (int i = 0; i < dim2; i++) {
        for (int j = 0; j < dim3; j++) {
            inB_[i * dim3 + j] = inB[i * dim3 + j] & mask;
        }
    }
}

//////////////////////
//初始化
///////////////////////////////

int main(int argc, char **argv) {
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.arg("m", precomputed_MSBs, "MSB_to_Wrap Optimization?");
  amap.arg("a", ::accumulate, "Accumulate?");

  amap.parse(argc, argv);

  iopack = new IOPack(party, port, "127.0.0.1");
  otpack = new OTPack(iopack, party);
  prod = new LinearOT(party, iopack, otpack);

  PRG128 prg; //(fix_key);

  uint64_t *inA = new uint64_t[dim1 * dim2];//1*100
  uint64_t *inB = new uint64_t[dim2 * dim3];//100*35
  int dim = (::accumulate ? dim1 * dim3 : dim1 * dim2 * dim3);
  uint64_t *outC = new uint64_t[dim];
  //使用PRG128分配并初始化随机矩阵inA和inB
  //prg.random_data(inA, dim1 * dim2 * sizeof(uint64_t));
  //prg.random_data(inB, dim2 * dim3 * sizeof(uint64_t));
  
  inA[0] = 32767;
  inB[0] = 10;
//   step2
//   inA[0] = inA[0]+alpha_;
//   inB[0] = inB[0]+alpha_;
//step5
    uint64_t *inA_ = new uint64_t[dim1 * dim2];//1*100
    uint64_t *inB_ = new uint64_t[dim2 * dim3];//100*35

    assign_lower_h_bits(dim1, dim2, dim3, inA, inB, inA_, inB_, h);
  /////////////////////////
//     std::cout << "Generated random data for inB:" << std::endl;
//     for (int i = 0; i < dim1 * dim2; i++) {
//         std::cout << "inA[" << i << "] = " <<  inA[i] << std::endl;
//     }
//     std::cout << "Generated random data for inB:" << std::endl;
//     for (int i = 0; i < dim1 * dim2; i++) {
//         std::cout << "inB[" << i << "] = " << inB[i] << std::endl;
//     }
//   /////////////////////////
    std::cout << "inA_[" << 0 << "] = " <<  inA_[0] << std::endl;
    std::cout << "inB_[" << 0 << "] = " <<  inB_[0] << std::endl;
//step6
    
    trunc_oracle = new Truncation(party, iopack, otpack);
    uint64_t *outtrunc = new uint64_t[dim];
        if (party == ALICE) {
        trunc_oracle->truncate_and_reduce(dim, inA_, outtrunc, 10, 15);//shift=h-s,hypothesis s=5
        std::cout << "outtrunc[" << 0 << "] = " << outtrunc[0] << std::endl;
    } else  {
        trunc_oracle->truncate_and_reduce(dim, inB_, outtrunc, 10, 15);//shift=h-s,hypothesis s=5
        std::cout << "outtrunc[" << 0 << "] = " << outtrunc[0] << std::endl;
    }

//step7
int32_t T_size = sizeof(uint64_t) * 8;
int bw_xlut = 8;
int bw_ylut;
aux = new AuxProtocols(party, iopack, otpack);
if (T_size == 8)
  bw_ylut = 7;
else
  bw_ylut = 29;
  uint64_t a_alice=1;
  uint64_t b_alice=10;
  uint64_t **spec_a = new uint64_t *[dim];
  uint64_t *a_bob = new uint64_t[dim];
  uint64_t N = 1ULL << bw_xlut;

    for (int i = 0; i < dim; i++) {
        spec_a[i] = new uint64_t[N];
        prg.random_data(spec_a[i], N * sizeof(uint64_t));
        for (int j = 0; j < N; j++) {
            spec_a[i][j] = spec_a[i][j] % 1001; // Restrict values to range 0-1000
        }
    }
    // for (int i = 0; i < dim; ++i) {
    //     std::cout << "spec[" << i << "]: ";
    //     for (int j = 0; j < N; ++j) {
    //         std::cout << spec[i][j] << " "; // Output the value in decimal format
    //     }
    //     std::cout << std::endl; // Move to the next line after each row
    // }

  if (party == ALICE) {
    aux->lookup_table<uint64_t>(spec_a, nullptr, nullptr, dim, bw_xlut, bw_ylut);
  } else { // party == BOB
    aux->lookup_table<uint64_t>(nullptr, outtrunc, a_bob, dim, bw_xlut, bw_ylut);
  }

std::cout << "y_a[" << 0 << "] = " << a_bob[0] << std::endl;

/////选择截距
  uint64_t **spec_b = new uint64_t *[dim];
  uint64_t *b_bob = new uint64_t[dim];

    for (int i = 0; i < dim; i++) {
        spec_b[i] = new uint64_t[N];
        prg.random_data(spec_b[i], N * sizeof(uint64_t));
        for (int j = 0; j < N; j++) {
            spec_b[i][j] = spec_b[i][j] % 1001; // Restrict values to range 0-1000
        }
    }
    // for (int i = 0; i < dim; ++i) {
    //     std::cout << "spec[" << i << "]: ";
    //     for (int j = 0; j < N; ++j) {
    //         std::cout << spec[i][j] << " "; // Output the value in decimal format
    //     }
    //     std::cout << std::endl; // Move to the next line after each row
    // }

  if (party == ALICE) {
    aux->lookup_table<uint64_t>(spec_b, nullptr, nullptr, dim, bw_xlut, bw_ylut);
  } else { // party == BOB
    aux->lookup_table<uint64_t>(nullptr, outtrunc, b_bob, dim, bw_xlut, bw_ylut);
  }

std::cout << "y_b[" << 0 << "] = " << b_bob[0] << std::endl;


//////////////////////step8
  for (int i = 0; i < dim1 * dim2; i++) {
    inA[i] &= maskA;
  }
  std::cout << "maskA = " << maskA << std::endl;
  for (int i = 0; i < dim2 * dim3; i++) {
    inB[i] &= maskB;
  }

  uint8_t *msbA = nullptr;
  uint8_t *msbB = nullptr;
  if (precomputed_MSBs) { //预计算MSB
    msbA = new uint8_t[dim1 * dim2];
    msbB = new uint8_t[dim2 * dim3];
    prod->aux->MSB(inA, msbA, dim1 * dim2, bwA);
    prod->aux->MSB(inB, msbB, dim2 * dim3, bwB);
  }

//   test_matrix_multiplication(inA, inB, outC, false);
  //test_matrix_multiplication(inA, inB, outC, true);
    prod->matrix_multiplication(dim1, dim2, dim3, inA, inB, outC, bwA, bwB, bwC,
                              true, signed_B, ::accumulate, mode,
                              msbA, msbB);
    /////////////////////////
      std::cout << "Generated random data for outC:" << std::endl;
    for (int i = 0; i < dim1 * dim3; i++) {
        std::cout << "outC[" << i << "] =" << outC[i] << std::endl;
    }
    //////////////////////MillionaireWithEquality
    MillionaireWithEquality millionaire(party, iopack, otpack, bitlength, radix_base);
      // 比较两个数
// ALICE 和 BOB 分别输入自己的数据
    uint64_t local_data1;
    if (party == ALICE) {
        local_data1 = inA[0];
    } else if (party == BOB) {
        local_data1 = 2^14;
    }

    // 调用 compare_with_eq 函数进行比较
    uint8_t res_cmp[1]; // 存储比较结果
    uint8_t res_eq[1];  // 存储相等性结果
    bool b;
    bool b_;//b'
    // 参与方分别传入自己的数据
    millionaire.compare_with_eq(res_cmp, res_eq, &local_data1, 1, bitlength, true, radix_base);//line10

    // 输出比较结果
    if (party == ALICE) {
        if (res_cmp[0] || res_eq[0]) {
            std::cout << "inA[0]'s value >= beita's value\n";
            b=1;
        } else {
            std::cout << "inA[0]'s value < beita's value\n";
            b=0;
        }
    }
    uint64_t local_data2;
    if (party == ALICE) {
        local_data2 = inA[0];
    } else if (party == BOB) {
        local_data2 = 0;
    }
    millionaire.compare_with_eq(res_cmp, res_eq, &local_data2, 1, bitlength, true, radix_base);//line10

    if (party == ALICE) {
    if (res_cmp[0] || res_eq[0]) {
            std::cout << "inA[0]'s value >= beita's value\n";
            b_=1;
        } else {
            std::cout << "inA[0]'s value < beita's value\n";
            b_=0;
        }
    }

////////////////////////////////
    //aux = new AuxProtocols(party, iopack, otpack);

    uint8_t *MUX_sel = new uint8_t[dim1];
    int bw_x = 32, bw_y = 32;

    uint64_t *MUX_data1 = new uint64_t[dim1];
    uint64_t *MUX_output1 = new uint64_t[dim1];
    if (party == ALICE) {
        MUX_sel[0] = b^b_;
        MUX_data1[0] = outC[0]+1;
    } else if (party == BOB) {
        MUX_sel[0] = 0;
        MUX_data1[0] = outC[0]+1;//ax+d
    }
    std::cout << "MUX_sel[" << 0 << "] =" << MUX_sel[0] << std::endl;
    std::cout << "MUX_data1[" << 0 << "] =" << MUX_data1[0] << std::endl;

    aux->multiplexer(MUX_sel, MUX_data1, MUX_output1, dim1, bw_x, bw_y);
    std::cout << "MUX_sel[" << 0 << "] =" << MUX_sel[0] << std::endl;
    std::cout << "MUX_data1[" << 0 << "] =" << MUX_data1[0] << std::endl;
    std::cout << "MUX_output1[" << 0 << "] =" << MUX_output1[0] << std::endl;



///////////////////////////////
  cout << "Precomputed MSBs: " << precomputed_MSBs << endl;
  cout << "Accumulate: " << ::accumulate << endl;
  mode = MultMode::None;
  cout << "Mode: None" << endl;

  delete[] inA_;
  delete[] inB_;
  delete[] inA;
  delete[] inB;
  delete[] outC;
  delete prod;

}
