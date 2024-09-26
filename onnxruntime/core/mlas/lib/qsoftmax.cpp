/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qsoftmax.cpp

Abstract:

    This module implements miscellaneous computation routines.

    Our usage requires building platform specific versions of the algorithm to
    target different instruction sets. The implementation below targets the
    base instruction set (typically SSE2) while assembly implementations target
    newer instruction sets (such as FMA3).

--*/

#include "mlasi.h"


struct MLAS_QSOFTMAX_WORK_BLOCK {
    const void* Input;
    void* Output;
    size_t N;
    size_t D;
    const float* LoopupTable;
    float Scale;
    float ZeroPoint;
    size_t ThreadCountN;
    bool is_signed;
};


extern void MlasQSoftmaxI8KernelAVX2(size_t N,
                 size_t D,
                 const int8_t* x_data,
                 int8_t* y_data,
                 const float* lookup_table,
                 float y_scale,
                 int8_t yzp,
                 float* tempaddr);

extern void MlasQSoftmaxU8KernelAVX2(size_t N,
                 size_t D,
                 const uint8_t* x_data,
                 uint8_t* y_data,
                 const float* lookup_table,
                 float y_scale,
                 uint8_t yzp,
                 float* tempaddr);

void
MlasComputeQSoftmaxThreaded(
    void* Context,
    ptrdiff_t Index
)
/*++

Routine Description:

    This routine is invoked from a worker thread to execute a segment of a
    softmax or log softmax operation.

Arguments:

    Context - Supplies the pointer to the context for the threaded operation.

    ThreadId - Supplies the current index of the threaded operation.

Return Value:

    None.

--*/
{
    const auto* WorkBlock = (MLAS_QSOFTMAX_WORK_BLOCK*)Context;

    //
    // Partition the operation along the N dimension.
    //

    size_t n;
    size_t CountN;

    MlasPartitionWork(Index, WorkBlock->ThreadCountN, WorkBlock->N, &n, &CountN);
    size_t packBSize = (WorkBlock->D*sizeof(float) + ThreadedBufAlignment - 1) / ThreadedBufAlignment;
    packBSize *= ThreadedBufAlignment;

    MlasThreadedBufAlloc(packBSize);

    float *tempaddr = reinterpret_cast <float*>(ThreadedBufHolder.get());

    //
    // Compute the softmax or log softmax function.
    //

    const size_t D = WorkBlock->D;
    const float Scale = WorkBlock->Scale;
    const float ZeroPoint = WorkBlock->ZeroPoint;
    const float* LoopupTable = WorkBlock->LoopupTable;

    const int8_t* Input = reinterpret_cast <const int8_t*>(WorkBlock->Input) + n * D;
    int8_t* Output = reinterpret_cast <int8_t*>(WorkBlock->Output) + n * D;

#if defined(MLAS_SSE2_INTRINSICS)
    // TODO: Use std::hardware_constructive_interference_size
    constexpr size_t CacheLineSize = 64;
    constexpr size_t ElementsPerCacheLine = CacheLineSize / sizeof(float);
#endif

    while (CountN > 0) {
#if defined(MLAS_SSE2_INTRINSICS)
        //
        // Prefetch the next row of the input buffer.
        //

        for (size_t i = 0; i * ElementsPerCacheLine < D; i++) {
            _mm_prefetch((char*)(Input + D) + i * CacheLineSize, _MM_HINT_T0);
        }
#endif
        if (WorkBlock->is_signed) {
            MlasQSoftmaxI8KernelAVX2(1, D, (Input), Output, LoopupTable, Scale, ZeroPoint, tempaddr);
        } else {
            MlasQSoftmaxU8KernelAVX2(1, D, reinterpret_cast <const uint8_t*>(Input), reinterpret_cast <uint8_t*>(Output), LoopupTable, Scale, ZeroPoint, tempaddr);
        }

        Input += D;
        Output += D;
        CountN--;
    }
}

void
MLASCALL
MlasComputeQSoftmax(
    const void* Input,
    void* Output,
    size_t N,
    size_t D,
    const float* LoopupTable,
    float Scale,
    float ZeroPoint,
    bool is_signed,
    MLAS_THREADPOOL* ThreadPool
)
/*++

Routine Description:

    This routine computes the quantized softmax function.

    N.B. This implementation supports in place updates of the output buffer.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of rows to process.

    D - Supplies the number of columns per row to process.

    LoopupTable - Supplies lookup exp values.

    Scale - quantization params.
    ZeroPoint - quantization params.
    is_signed - int8 or uint8.

    ThreadPool - Supplies the thread pool object to use, else nullptr if the
        base library threading support should be used.

Return Value:

    None.

--*/
{

    MLAS_QSOFTMAX_WORK_BLOCK WorkBlock;

    //
    // Capture the softmax parameters to the work block.
    //

    WorkBlock.LoopupTable = LoopupTable;
    WorkBlock.Scale = Scale;
    WorkBlock.ZeroPoint = ZeroPoint;
    WorkBlock.Input = Input;
    WorkBlock.Output = Output;
    WorkBlock.N = N;
    WorkBlock.D = D;
    WorkBlock.is_signed = is_signed;

    //
    // Compute the number of target threads given the complexity of the softmax
    // operation. Limit the number of threads to the number of rows and try to
    // keep each thread processing a minimum number of elements before using
    // another thread.
    //

    ptrdiff_t ThreadCountN = MlasGetMaximumThreadCount(ThreadPool);

    if (size_t(ThreadCountN) > N) {
        ThreadCountN = ptrdiff_t(N);
    }

    WorkBlock.ThreadCountN = ThreadCountN;

    MlasExecuteThreaded(MlasComputeQSoftmaxThreaded, &WorkBlock, ThreadCountN, ThreadPool);
}
