/***************************************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Tests for device-wide GEMM interface
*/

#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/numeric_types.h"

#include "../../common/cutlass_unit_test.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"

#include "testbed.h"

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   2 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   1 x   1
//       Threadblock:   8 x  32 x  8
CUTLASS_TEST_L1(SM50_device_qgemm_nn, 8x32x8_8x32x1_2x4_4x8_1x1, {
    using precision = cutlass::Quaternion<float>;
    using ThreadblockShape = cutlass::gemm::GemmShape<8, 32, 8>;
    using WarpShape = cutlass::gemm::GemmShape<8, 32, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = cutlass::gemm::device::Gemm<
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::RowMajor,
        precision,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   1 x   1
//       Threadblock:  16 x  32 x  8
CUTLASS_TEST_L0(SM50_device_qgemm_nn, 16x32x8_16x32x1_4x4_4x8_1x1, {
    using precision = cutlass::Quaternion<float>;
    using ThreadblockShape = cutlass::gemm::GemmShape<16, 32, 8>;
    using WarpShape = cutlass::gemm::GemmShape<16, 32, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = cutlass::gemm::device::Gemm<
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::RowMajor,
        precision,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   2 x   2
//    Threads / Warp:   4 x   8
//     Warps / Block:   1 x   2
//       Threadblock:   8 x  32 x  8
CUTLASS_TEST_L2(SM50_device_qgemm_nn, 8x32x8_8x16x1_2x2_4x8_1x2, {
    using precision = cutlass::Quaternion<float>;
    using ThreadblockShape = cutlass::gemm::GemmShape<8, 32, 8>;
    using WarpShape = cutlass::gemm::GemmShape<8, 16, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = cutlass::gemm::device::Gemm<
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::RowMajor,
        precision,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   2 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   1 x   2
//       Threadblock:   8 x  64 x  8
CUTLASS_TEST_L1(SM50_device_qgemm_nn, 8x64x8_8x32x1_2x4_4x8_1x2, {
    using precision = cutlass::Quaternion<float>;
    using ThreadblockShape = cutlass::gemm::GemmShape<8, 64, 8>;
    using WarpShape = cutlass::gemm::GemmShape<8, 32, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = cutlass::gemm::device::Gemm<
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::RowMajor,
        precision,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   2
//    Threads / Warp:   4 x   8
//     Warps / Block:   1 x   2
//       Threadblock:  16 x  32 x  8
CUTLASS_TEST_L1(SM50_device_qgemm_nn, 16x32x8_16x16x1_4x2_4x8_1x2, {
    using precision = cutlass::Quaternion<float>;
    using ThreadblockShape = cutlass::gemm::GemmShape<16, 32, 8>;
    using WarpShape = cutlass::gemm::GemmShape<16, 16, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = cutlass::gemm::device::Gemm<
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::RowMajor,
        precision,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   1 x   2
//       Threadblock:  16 x  64 x  8
CUTLASS_TEST_L1(SM50_device_qgemm_nn, 16x64x8_16x32x1_4x4_4x8_1x2, {
    using precision = cutlass::Quaternion<float>;
    using ThreadblockShape = cutlass::gemm::GemmShape<16, 64, 8>;
    using WarpShape = cutlass::gemm::GemmShape<16, 32, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = cutlass::gemm::device::Gemm<
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::RowMajor,
        precision,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   8 x   4
//     Warps / Block:   1 x   2
//       Threadblock:  32 x  32 x  8
CUTLASS_TEST_L1(SM50_device_qgemm_nn, 32x32x8_32x16x1_4x4_8x4_1x2, {
    using precision = cutlass::Quaternion<float>;
    using ThreadblockShape = cutlass::gemm::GemmShape<32, 32, 8>;
    using WarpShape = cutlass::gemm::GemmShape<32, 16, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = cutlass::gemm::device::Gemm<
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::RowMajor,
        precision,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   1
//       Threadblock:  32 x  32 x  8
CUTLASS_TEST_L2(SM50_device_qgemm_nn, 32x32x8_16x32x1_4x4_4x8_2x1, {
    using precision = cutlass::Quaternion<float>;
    using ThreadblockShape = cutlass::gemm::GemmShape<32, 32, 8>;
    using WarpShape = cutlass::gemm::GemmShape<16, 32, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = cutlass::gemm::device::Gemm<
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::RowMajor,
        precision,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   2 x   2
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   2
//       Threadblock:  16 x  32 x  8
CUTLASS_TEST_L2(SM50_device_qgemm_nn, 16x32x8_8x16x1_2x2_4x8_2x2, {
    using precision = cutlass::Quaternion<float>;
    using ThreadblockShape = cutlass::gemm::GemmShape<16, 32, 8>;
    using WarpShape = cutlass::gemm::GemmShape<8, 16, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = cutlass::gemm::device::Gemm<
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::RowMajor,
        precision,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   2 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   2
//       Threadblock:  16 x  64 x  8
CUTLASS_TEST_L2(SM50_device_qgemm_nn, 16x64x8_8x32x1_2x4_4x8_2x2, {
    using precision = cutlass::Quaternion<float>;
    using ThreadblockShape = cutlass::gemm::GemmShape<16, 64, 8>;
    using WarpShape = cutlass::gemm::GemmShape<8, 32, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = cutlass::gemm::device::Gemm<
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::RowMajor,
        precision,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   2
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   2
//       Threadblock:  32 x  32 x  8
CUTLASS_TEST_L2(SM50_device_qgemm_nn, 32x32x8_16x16x1_4x2_4x8_2x2, {
    using precision = cutlass::Quaternion<float>;
    using ThreadblockShape = cutlass::gemm::GemmShape<32, 32, 8>;
    using WarpShape = cutlass::gemm::GemmShape<16, 16, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = cutlass::gemm::device::Gemm<
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::RowMajor,
        precision,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   2
//       Threadblock:  32 x  64 x  8
CUTLASS_TEST_L0(SM50_device_qgemm_nn, 32x64x8_16x32x1_4x4_4x8_2x2, {
    using precision = cutlass::Quaternion<float>;
    using ThreadblockShape = cutlass::gemm::GemmShape<32, 64, 8>;
    using WarpShape = cutlass::gemm::GemmShape<16, 32, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = cutlass::gemm::device::Gemm<
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::RowMajor,
        precision,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   8 x   4
//     Warps / Block:   2 x   2
//       Threadblock:  64 x  32 x  8
CUTLASS_TEST_L1(SM50_device_qgemm_nn, 64x32x8_32x16x1_4x4_8x4_2x2, {
    using precision = cutlass::Quaternion<float>;
    using ThreadblockShape = cutlass::gemm::GemmShape<64, 32, 8>;
    using WarpShape = cutlass::gemm::GemmShape<32, 16, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = cutlass::gemm::device::Gemm<
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::RowMajor,
        precision,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   2 x   2
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   4
//       Threadblock:  16 x  64 x 16
CUTLASS_TEST_L2(SM50_device_qgemm_nn, 16x64x16_8x16x1_2x2_4x8_2x4, {
    using precision = cutlass::Quaternion<float>;
    using ThreadblockShape = cutlass::gemm::GemmShape<16, 64, 16>;
    using WarpShape = cutlass::gemm::GemmShape<8, 16, 16>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = cutlass::gemm::device::Gemm<
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::RowMajor,
        precision,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   2 x   2
//    Threads / Warp:   8 x   4
//     Warps / Block:   2 x   4
//       Threadblock:  32 x  32 x  8
CUTLASS_TEST_L2(SM50_device_qgemm_nn, 32x32x8_16x8x1_2x2_8x4_2x4, {
    using precision = cutlass::Quaternion<float>;
    using ThreadblockShape = cutlass::gemm::GemmShape<32, 32, 8>;
    using WarpShape = cutlass::gemm::GemmShape<16, 8, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = cutlass::gemm::device::Gemm<
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::RowMajor,
        precision,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   2
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   4
//       Threadblock:  32 x  64 x  8
CUTLASS_TEST_L1(SM50_device_qgemm_nn, 32x64x8_16x16x1_4x2_4x8_2x4, {
    using precision = cutlass::Quaternion<float>;
    using ThreadblockShape = cutlass::gemm::GemmShape<32, 64, 8>;
    using WarpShape = cutlass::gemm::GemmShape<16, 16, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = cutlass::gemm::device::Gemm<
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::RowMajor,
        precision,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   2 x   4
//       Threadblock:  32 x 128 x  8
CUTLASS_TEST_L1(SM50_device_qgemm_nn, 32x128x8_16x32x1_4x4_4x8_2x4, {
    using precision = cutlass::Quaternion<float>;
    using ThreadblockShape = cutlass::gemm::GemmShape<32, 128, 8>;
    using WarpShape = cutlass::gemm::GemmShape<16, 32, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = cutlass::gemm::device::Gemm<
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::RowMajor,
        precision,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   8 x   4
//     Warps / Block:   2 x   4
//       Threadblock:  64 x  64 x  8
CUTLASS_TEST_L1(SM50_device_qgemm_nn, 64x64x8_32x16x1_4x4_8x4_2x4, {
    using precision = cutlass::Quaternion<float>;
    using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 8>;
    using WarpShape = cutlass::gemm::GemmShape<32, 16, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = cutlass::gemm::device::Gemm<
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::RowMajor,
        precision,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   2 x   2
//    Threads / Warp:   4 x   8
//     Warps / Block:   4 x   2
//       Threadblock:  32 x  32 x  8
CUTLASS_TEST_L2(SM50_device_qgemm_nn, 32x32x8_8x16x1_2x2_4x8_4x2, {
    using precision = cutlass::Quaternion<float>;
    using ThreadblockShape = cutlass::gemm::GemmShape<32, 32, 8>;
    using WarpShape = cutlass::gemm::GemmShape<8, 16, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = cutlass::gemm::device::Gemm<
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::RowMajor,
        precision,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   2
//    Threads / Warp:   4 x   8
//     Warps / Block:   4 x   2
//       Threadblock:  64 x  32 x  8
CUTLASS_TEST_L2(SM50_device_qgemm_nn, 64x32x8_16x16x1_4x2_4x8_4x2, {
    using precision = cutlass::Quaternion<float>;
    using ThreadblockShape = cutlass::gemm::GemmShape<64, 32, 8>;
    using WarpShape = cutlass::gemm::GemmShape<16, 16, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = cutlass::gemm::device::Gemm<
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::RowMajor,
        precision,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   4 x   2
//       Threadblock:  64 x  64 x  8
CUTLASS_TEST_L2(SM50_device_qgemm_nn, 64x64x8_16x32x1_4x4_4x8_4x2, {
    using precision = cutlass::Quaternion<float>;
    using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 8>;
    using WarpShape = cutlass::gemm::GemmShape<16, 32, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = cutlass::gemm::device::Gemm<
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::RowMajor,
        precision,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   8 x   4
//     Warps / Block:   4 x   2
//       Threadblock: 128 x  32 x  8
CUTLASS_TEST_L1(SM50_device_qgemm_nn, 128x32x8_32x16x1_4x4_8x4_4x2, {
    using precision = cutlass::Quaternion<float>;
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 32, 8>;
    using WarpShape = cutlass::gemm::GemmShape<32, 16, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = cutlass::gemm::device::Gemm<
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::RowMajor,
        precision,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   2 x   2
//    Threads / Warp:   4 x   8
//     Warps / Block:   4 x   4
//       Threadblock:  32 x  64 x 16
CUTLASS_TEST_L2(SM50_device_qgemm_nn, 32x64x16_8x16x1_2x2_4x8_4x4, {
    using precision = cutlass::Quaternion<float>;
    using ThreadblockShape = cutlass::gemm::GemmShape<32, 64, 16>;
    using WarpShape = cutlass::gemm::GemmShape<8, 16, 16>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = cutlass::gemm::device::Gemm<
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::RowMajor,
        precision,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   2 x   2
//    Threads / Warp:   8 x   4
//     Warps / Block:   4 x   4
//       Threadblock:  64 x  32 x 16
CUTLASS_TEST_L2(SM50_device_qgemm_nn, 64x32x16_16x8x1_2x2_8x4_4x4, {
    using precision = cutlass::Quaternion<float>;
    using ThreadblockShape = cutlass::gemm::GemmShape<64, 32, 16>;
    using WarpShape = cutlass::gemm::GemmShape<16, 8, 16>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = cutlass::gemm::device::Gemm<
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::RowMajor,
        precision,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   2
//    Threads / Warp:   4 x   8
//     Warps / Block:   4 x   4
//       Threadblock:  64 x  64 x  8
CUTLASS_TEST_L2(SM50_device_qgemm_nn, 64x64x8_16x16x1_4x2_4x8_4x4, {
    using precision = cutlass::Quaternion<float>;
    using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 8>;
    using WarpShape = cutlass::gemm::GemmShape<16, 16, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = cutlass::gemm::device::Gemm<
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::RowMajor,
        precision,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   4 x   8
//     Warps / Block:   4 x   4
//       Threadblock:  64 x 128 x  8
CUTLASS_TEST_L1(SM50_device_qgemm_nn, 64x128x8_16x32x1_4x4_4x8_4x4, {
    using precision = cutlass::Quaternion<float>;
    using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 8>;
    using WarpShape = cutlass::gemm::GemmShape<16, 32, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = cutlass::gemm::device::Gemm<
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::RowMajor,
        precision,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   8 x   4
//     Warps / Block:   4 x   4
//       Threadblock: 128 x  64 x  8
CUTLASS_TEST_L1(SM50_device_qgemm_nn, 128x64x8_32x16x1_4x4_8x4_4x4, {
    using precision = cutlass::Quaternion<float>;
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 8>;
    using WarpShape = cutlass::gemm::GemmShape<32, 16, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = cutlass::gemm::device::Gemm<
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::RowMajor,
        precision,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

