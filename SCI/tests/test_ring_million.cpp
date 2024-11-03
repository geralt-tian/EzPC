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
#include <matplotlibcpp.h>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

using namespace sci;
using namespace std;
namespace plt = matplotlibcpp;
using namespace plt;
int party, port = 32000;
string address = "127.0.0.1";
IOPack *iopack;
OTPack *otpack;
LinearOT *prod;
XTProtocol *ext;

int bwL = 21; // 矩阵位宽
uint64_t mask_bwL = (bwL == 64 ? -1 : ((1ULL << bwL) - 1));

bool signed_B = true;           // 表示矩阵B是否为有符号数
bool accumulate = true;         // 决定是否累加结果
bool precomputed_MSBs = false;  // 决定是否预计算最高有效位
MultMode mode = MultMode::None; // 乘法模式

// uint64_t la = 14;//la=5 f=5,la=14,f=12
uint64_t lb = 10;
// uint64_t f = 12;
uint64_t la = 6; // la=5 f=5,la=14,f=12
uint64_t f = 12;
uint64_t h = f + 2;
uint64_t Tk = f - 1;
uint64_t alpha = 3.5 * pow(2, f);

uint64_t mask_l_Tk = (bwL == 64 ? -1 : ((1ULL << (bwL - Tk)) - 1));
uint64_t mask_lah1 = ((la + h + 1) == 64 ? -1 : ((1ULL << (la + h + 1)) - 1));
uint64_t mask_lla = ((la + bwL) == 64 ? -1 : ((1ULL << (la + bwL)) - 1));
uint64_t s = 6;
// s = 5(低精度)，s = 6(高)， s = 7 与 s = 6 误差相差不大
Truncation *trunc_oracle;
AuxProtocols *aux;
MillionaireWithEquality *mill_eq;
//////////////////////
// 初始化
///////////////////////////////

int main(int argc, char **argv)
{
    ArgMapping amap;
    int dim = 10000;
    uint8_t acc = 1;
    uint64_t init_input = 0;
    uint64_t step_size = 2;
    
    amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
    amap.arg("p", port, "Port Number");
    amap.arg("ip", address, "IP Address of server (ALICE)");
    amap.arg("m", precomputed_MSBs, "MSB_to_Wrap Optimization?");
    amap.arg("a", ::accumulate, "Accumulate?");
    amap.arg("dim", dim, "Dimension parameter for accumulation");
    amap.arg("init_input", init_input, "init_input for accumulation");
    amap.arg("step_size", step_size, "step_size for accumulation");
    amap.arg("acc", acc, "acc=0 low, acc=1 general (default), acc =2 high");

    amap.parse(argc, argv);
    std::cout << "Parsed dimension (dim) = " << dim << std::endl;
    iopack = new IOPack(party, port, "127.0.0.1");
    otpack = new OTPack(iopack, party);
mill_eq = new MillionaireWithEquality(party, iopack, otpack);
    uint64_t comm_start = iopack->get_comm();

    prod = new LinearOT(party, iopack, otpack);

    uint64_t *inA = new uint64_t[dim];
    uint64_t *inB = new uint64_t[dim];

    uint64_t *outax = new uint64_t[dim];
    for (int i = 0; i < dim; i++)
    {
        inA[i] = init_input + i * step_size;
        inB[i] = init_input + i * step_size;
    }

    

    uint8_t *Drelu = new uint8_t[dim];
    uint8_t *msbA = new uint8_t[dim];
    uint8_t *msbB = new uint8_t[dim];
    uint8_t *res_eq = new uint8_t[dim];
    uint8_t *res_cmp = new uint8_t[dim];
    uint8_t *wrap = new uint8_t[dim];
    uint64_t STEP3_comm_start = iopack->get_comm();
    // Drelu = MSB , Alice ^1
    int cmpbwl[] = {6, 7, 8, 9, 10, 11, 20};
    // auto Total_time;
    int flag = 0;
    auto Total_time = 0.2;
    for (int i = 0; i < 7; i++)
    {
        int bitwidth = cmpbwl[i];

        for (int j = 0; j < 6; j++)
        {
            uint64_t STEP3_comm_start = iopack->get_comm();
            if (party == ALICE)
            {
                // prod->aux->MSB(inA, msbA, dim, bwL);
                flag++;
                // prod->aux->mill->compare(msbA, inA, dim, bitwidth, true, false, j + 3);

                mill_eq->compare_with_eq(res_cmp,res_eq,inA, dim, bitwidth, true,  j + 3);
                
                // uint64_t STEP3_comm_end = iopack->get_comm();
            }
            else
            {


                // prod->aux->MSB(inB, msbB, dim, bwL);
                auto time_start = chrono::high_resolution_clock::now();
                // prod->aux->mill->compare(msbA, inB, dim, bitwidth, true, false, j + 3);
                mill_eq->compare_with_eq(res_cmp,res_eq, inB, dim, bitwidth, true,  j + 3);
                auto time_end = chrono::high_resolution_clock::now();
                Total_time = chrono::duration_cast<chrono::microseconds>(time_end - time_start).count();
            }
            uint64_t STEP3_comm_end = iopack->get_comm();

            if (party == ALICE)
            {
                double Total_MSBytes_ALICE = static_cast<double>(STEP3_comm_end - STEP3_comm_start) / dim * 8;
                iopack->io->send_data(&Total_MSBytes_ALICE, sizeof(double));
            }
            else
            {
                double recv_Total_MSBytes_ALICE;
                iopack->io->recv_data(&recv_Total_MSBytes_ALICE, sizeof(double));

                double Total_MSBytes_BOB = static_cast<double>(STEP3_comm_end - STEP3_comm_start) / dim * 8;
                double Total_MSBytes = Total_MSBytes_BOB + recv_Total_MSBytes_ALICE;

                std::ofstream csvFile("/home/zhaoqian/EzPC/SCI/tests/auto_compare_eq_test_output.csv", std::ios::app);

                if (!csvFile.is_open())
                {
                    std::cerr << "无法打开文件用于写入: auto_compare_eq_test_output.csv" << std::endl;
                    return 1; // 或其他适当的错误处理
                }

                // 仅在文件为空时写入列名
                csvFile.seekp(0, std::ios::end);
                if (csvFile.tellp() == 0)
                {
                    csvFile << "bitwidth, M , Total_MSBytes, time\n";
                }
                csvFile << bitwidth << ","
                        << j + 3 << ","
                        << Total_MSBytes << ","
                        << Total_time
                        << "\n";
            }
            // std::cout << "STEP3_communication = " << (STEP3_comm_end - STEP3_comm_start) / dim * 8 << " bytes" << std::endl;
        }
    }

    // cout << "Total time: "
    //      << chrono::duration_cast<chrono::milliseconds>(time_end - time_start).count()
    //      << " ms" << endl;

    // uint64_t STEP3_comm_end = iopack->get_comm();
    // mill->compare(msbA, inA, dim, bwL, true); // computing greater_than

    delete[] inA;
    delete[] inB;
    delete[] outax;
    delete prod;
}
