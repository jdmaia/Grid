/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: Fujitsu_A64FX_intrin_double.h

    Copyright (C) 2020

Author: Nils Meyer <nils.meyer@ur.de>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

    See the full license in the file "LICENSE" in the top level distribution directory
*************************************************************************************/
/*  END LEGAL */
#define LOAD_CHIMU_A64FXd(x)           LOAD_CHIMU_INTERLEAVED_A64FXd(x)  
#define PREFETCH_CHIMU_L1(A)  
#define PREFETCH_GAUGE_L1(A)  
#define PREFETCH_CHIMU_L2(A)  
#define PREFETCH_GAUGE_L2(A)  
#define PF_GAUGE(A)  
#define PREFETCH1_CHIMU(A)  
#define PREFETCH_CHIMU(A)  
#define LOCK_GAUGE(A)  
#define UNLOCK_GAUGE(A)  
#define MASK_REGS                      DECLARATIONS_A64FXd  
#define COMPLEX_SIGNS(A)  
#define LOAD64(A,B)  
#define SAVE_RESULT(A,B)               RESULT_A64FXd(A)  
#define MULT_2SPIN_DIR_PF(A,B)         MULT_2SPIN_A64FXd(A)  
#define MAYBEPERM(A,perm)              { A ; }  
#define LOAD_CHI(base)                 LOAD_CHI_A64FXd(base)  
#define ZERO_PSI  
#define ADD_RESULT(base,basep)         LOAD_CHIMU_A64FXd(base); ADD_RESULT_INTERNAL_A64FXd; RESULT_A64FXd(base)  
#define XP_PROJMEM(base)               LOAD_CHIMU_A64FXd(base);   XP_PROJ_A64FXd  
#define YP_PROJMEM(base)               LOAD_CHIMU_A64FXd(base);   YP_PROJ_A64FXd  
#define ZP_PROJMEM(base)               LOAD_CHIMU_A64FXd(base);   ZP_PROJ_A64FXd  
#define TP_PROJMEM(base)               LOAD_CHIMU_A64FXd(base);   TP_PROJ_A64FXd  
#define XM_PROJMEM(base)               LOAD_CHIMU_A64FXd(base);   XM_PROJ_A64FXd  
#define YM_PROJMEM(base)               LOAD_CHIMU_A64FXd(base);   YM_PROJ_A64FXd  
#define ZM_PROJMEM(base)               LOAD_CHIMU_A64FXd(base);   ZM_PROJ_A64FXd  
#define TM_PROJMEM(base)               LOAD_CHIMU_A64FXd(base);   TM_PROJ_A64FXd  
#define XP_RECON                       XP_RECON_A64FXd  
#define XM_RECON                       XM_RECON_A64FXd  
#define XM_RECON_ACCUM                 XM_RECON_ACCUM_A64FXd  
#define YM_RECON_ACCUM                 YM_RECON_ACCUM_A64FXd  
#define ZM_RECON_ACCUM                 ZM_RECON_ACCUM_A64FXd  
#define TM_RECON_ACCUM                 TM_RECON_ACCUM_A64FXd  
#define XP_RECON_ACCUM                 XP_RECON_ACCUM_A64FXd  
#define YP_RECON_ACCUM                 YP_RECON_ACCUM_A64FXd  
#define ZP_RECON_ACCUM                 ZP_RECON_ACCUM_A64FXd  
#define TP_RECON_ACCUM                 TP_RECON_ACCUM_A64FXd  
#define PERMUTE_DIR0                   LOAD_TABLE0; if (perm) { PERM0_A64FXd; }  
#define PERMUTE_DIR1                   LOAD_TABLE1; if (perm) { PERM1_A64FXd; }  
#define PERMUTE_DIR2                   LOAD_TABLE2; if (perm) { PERM2_A64FXd; }  
#define PERMUTE_DIR3  
// DECLARATIONS
#define DECLARATIONS_A64FXd  \
    const uint64_t lut[4][8] = { \
        {4, 5, 6, 7, 0, 1, 2, 3}, \
        {2, 3, 0, 1, 6, 7, 4, 5}, \
        {1, 0, 3, 2, 5, 4, 7, 6}, \
        {0, 1, 2, 4, 5, 6, 7, 8} };\
    svfloat64_t result_00;        \
    svfloat64_t result_01;        \
    svfloat64_t result_02;        \
    svfloat64_t result_10;        \
    svfloat64_t result_11;        \
    svfloat64_t result_12;        \
    svfloat64_t result_20;        \
    svfloat64_t result_21;        \
    svfloat64_t result_22;        \
    svfloat64_t result_30;        \
    svfloat64_t result_31;        \
    svfloat64_t result_32;        \
    svfloat64_t Chi_00;        \
    svfloat64_t Chi_01;        \
    svfloat64_t Chi_02;        \
    svfloat64_t Chi_10;        \
    svfloat64_t Chi_11;        \
    svfloat64_t Chi_12;        \
    svfloat64_t UChi_00;        \
    svfloat64_t UChi_01;        \
    svfloat64_t UChi_02;        \
    svfloat64_t UChi_10;        \
    svfloat64_t UChi_11;        \
    svfloat64_t UChi_12;        \
    svfloat64_t U_00;        \
    svfloat64_t U_10;        \
    svfloat64_t U_20;        \
    svfloat64_t U_01;        \
    svfloat64_t U_11;        \
    svfloat64_t U_21;        \
    svbool_t pg1;        \
    pg1 = svptrue_b64();        \
    svuint64_t table0; \
    svfloat64_t zero0;        \
    zero0 = __svzero(zero0); 

#define Chimu_00 Chi_00  
#define Chimu_01 Chi_01  
#define Chimu_02 Chi_02  
#define Chimu_10 Chi_10  
#define Chimu_11 Chi_11  
#define Chimu_12 Chi_12  
#define Chimu_20 U_00  
#define Chimu_21 U_10  
#define Chimu_22 U_20  
#define Chimu_30 U_01  
#define Chimu_31 U_11  
#define Chimu_32 U_21  
// RESULT
#define RESULT_A64FXd(base)  \
{ \
    svstnt1(pg1, (float64_t*)(base + 2 * 3 * 64 + -6 * 64), result_00);  \
    svstnt1(pg1, (float64_t*)(base + 2 * 3 * 64 + -5 * 64), result_01);  \
    svstnt1(pg1, (float64_t*)(base + 2 * 3 * 64 + -4 * 64), result_02);  \
    svstnt1(pg1, (float64_t*)(base + 2 * 3 * 64 + -3 * 64), result_10);  \
    svstnt1(pg1, (float64_t*)(base + 2 * 3 * 64 + -2 * 64), result_11);  \
    svstnt1(pg1, (float64_t*)(base + 2 * 3 * 64 + -1 * 64), result_12);  \
    svstnt1(pg1, (float64_t*)(base + 2 * 3 * 64 + 0 * 64), result_20);  \
    svstnt1(pg1, (float64_t*)(base + 2 * 3 * 64 + 1 * 64), result_21);  \
    svstnt1(pg1, (float64_t*)(base + 2 * 3 * 64 + 2 * 64), result_22);  \
    svstnt1(pg1, (float64_t*)(base + 2 * 3 * 64 + 3 * 64), result_30);  \
    svstnt1(pg1, (float64_t*)(base + 2 * 3 * 64 + 4 * 64), result_31);  \
    svstnt1(pg1, (float64_t*)(base + 2 * 3 * 64 + 5 * 64), result_32);  \
}
// PREFETCH_CHIMU_L2 (prefetch to L2)
#define PREFETCH_CHIMU_L2_INTERNAL_A64FXd(base)  \
{ \
    svprfd(pg1, (int64_t*)(base + 0), SV_PLDL2STRM); \
    svprfd(pg1, (int64_t*)(base + 256), SV_PLDL2STRM); \
    svprfd(pg1, (int64_t*)(base + 512), SV_PLDL2STRM); \
}
// PREFETCH_CHIMU_L1 (prefetch to L1)
#define PREFETCH_CHIMU_L1_INTERNAL_A64FXd(base)  \
{ \
    svprfd(pg1, (int64_t*)(base + 0), SV_PLDL1STRM); \
    svprfd(pg1, (int64_t*)(base + 256), SV_PLDL1STRM); \
    svprfd(pg1, (int64_t*)(base + 512), SV_PLDL1STRM); \
}
// PREFETCH_GAUGE_L2 (prefetch to L2)
#define PREFETCH_GAUGE_L2_INTERNAL_A64FXd(A)  \
{ \
    const auto & ref(U[sUn][A]); uint64_t baseU = (uint64_t)&ref[0][0]; \
    svprfd(pg1, (int64_t*)(baseU + 0), SV_PLDL2STRM); \
    svprfd(pg1, (int64_t*)(baseU + 256), SV_PLDL2STRM); \
    svprfd(pg1, (int64_t*)(baseU + 512), SV_PLDL2STRM); \
}
// PREFETCH_GAUGE_L1 (prefetch to L1)
#define PREFETCH_GAUGE_L1_INTERNAL(A)_A64FXd  \
{ \
    const auto & ref(U[sU][A]); uint64_t baseU = (uint64_t)&ref[0][0]; \
    svprfd(pg1, (int64_t*)(baseU + 0), SV_PLDL1STRM); \
    svprfd(pg1, (int64_t*)(baseU + 256), SV_PLDL1STRM); \
    svprfd(pg1, (int64_t*)(baseU + 512), SV_PLDL1STRM); \
}
// LOAD_CHI
#define LOAD_CHI_A64FXd(base)  \
{ \
    Chi_00 = svld1(pg1, (float64_t*)(base + 0 * 64));  \
    Chi_01 = svld1(pg1, (float64_t*)(base + 1 * 64));  \
    Chi_02 = svld1(pg1, (float64_t*)(base + 2 * 64));  \
    Chi_10 = svld1(pg1, (float64_t*)(base + 3 * 64));  \
    Chi_11 = svld1(pg1, (float64_t*)(base + 4 * 64));  \
    Chi_12 = svld1(pg1, (float64_t*)(base + 5 * 64));  \
}
// LOAD_CHIMU
#define LOAD_CHIMU_INTERLEAVED_A64FXd(base)  \
{ \
    Chimu_00 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + -6 * 64));  \
    Chimu_30 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + 3 * 64));  \
    Chimu_10 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + -3 * 64));  \
    Chimu_20 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + 0 * 64));  \
    Chimu_01 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + -5 * 64));  \
    Chimu_31 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + 4 * 64));  \
    Chimu_11 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + -2 * 64));  \
    Chimu_21 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + 1 * 64));  \
    Chimu_02 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + -4 * 64));  \
    Chimu_32 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + 5 * 64));  \
    Chimu_12 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + -1 * 64));  \
    Chimu_22 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + 2 * 64));  \
}
// LOAD_CHIMU_0213
#define LOAD_CHIMU_0213_A64FXd  \
{ \
    const SiteSpinor & ref(in[offset]); \
    Chimu_00 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + -6 * 64));  \
    Chimu_20 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + 0 * 64));  \
    Chimu_01 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + -5 * 64));  \
    Chimu_21 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + 1 * 64));  \
    Chimu_02 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + -4 * 64));  \
    Chimu_22 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + 2 * 64));  \
    Chimu_10 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + -3 * 64));  \
    Chimu_30 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + 3 * 64));  \
    Chimu_11 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + -2 * 64));  \
    Chimu_31 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + 4 * 64));  \
    Chimu_12 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + -1 * 64));  \
    Chimu_32 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + 5 * 64));  \
}
// LOAD_CHIMU_0312
#define LOAD_CHIMU_0312_A64FXd  \
{ \
    const SiteSpinor & ref(in[offset]); \
    Chimu_00 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + -6 * 64));  \
    Chimu_30 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + 3 * 64));  \
    Chimu_01 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + -5 * 64));  \
    Chimu_31 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + 4 * 64));  \
    Chimu_02 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + -4 * 64));  \
    Chimu_32 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + 5 * 64));  \
    Chimu_10 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + -3 * 64));  \
    Chimu_20 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + 0 * 64));  \
    Chimu_11 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + -2 * 64));  \
    Chimu_21 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + 1 * 64));  \
    Chimu_12 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + -1 * 64));  \
    Chimu_22 = svld1(pg1, (float64_t*)(base + 2 * 3 * 64 + 2 * 64));  \
}
// LOAD_TABLE0
#define LOAD_TABLE0  \
    table0 = svld1(pg1, (uint64_t*)&lut[0]);  

// LOAD_TABLE1
#define LOAD_TABLE1  \
    table0 = svld1(pg1, (uint64_t*)&lut[1]);  

// LOAD_TABLE2
#define LOAD_TABLE2  \
    table0 = svld1(pg1, (uint64_t*)&lut[2]);  

// LOAD_TABLE3
#define LOAD_TABLE3  \
    table0 = svld1(pg1, (uint64_t*)&lut[3]);  

// PERM0
#define PERM0_A64FXd  \
    Chi_00 = svtbl(Chi_00, table0);    \
    Chi_01 = svtbl(Chi_01, table0);    \
    Chi_02 = svtbl(Chi_02, table0);    \
    Chi_10 = svtbl(Chi_10, table0);    \
    Chi_11 = svtbl(Chi_11, table0);    \
    Chi_12 = svtbl(Chi_12, table0);    

// PERM1
#define PERM1_A64FXd  \
    Chi_00 = svtbl(Chi_00, table0);    \
    Chi_01 = svtbl(Chi_01, table0);    \
    Chi_02 = svtbl(Chi_02, table0);    \
    Chi_10 = svtbl(Chi_10, table0);    \
    Chi_11 = svtbl(Chi_11, table0);    \
    Chi_12 = svtbl(Chi_12, table0);    

// PERM2
#define PERM2_A64FXd  \
    Chi_00 = svtbl(Chi_00, table0);    \
    Chi_01 = svtbl(Chi_01, table0);    \
    Chi_02 = svtbl(Chi_02, table0);    \
    Chi_10 = svtbl(Chi_10, table0);    \
    Chi_11 = svtbl(Chi_11, table0);    \
    Chi_12 = svtbl(Chi_12, table0);    

// PERM3
#define PERM3_A64FXd  

// MULT_2SPIN
#define MULT_2SPIN_A64FXd(A)  \
{ \
    const auto & ref(U[sU](A)); uint64_t baseU = (uint64_t)&ref; \
    U_00 = svld1(pg1, (float64_t*)(baseU + 2 * 3 * 64 + -6 * 64));  \
    U_10 = svld1(pg1, (float64_t*)(baseU + 2 * 3 * 64 + -3 * 64));  \
    U_20 = svld1(pg1, (float64_t*)(baseU + 2 * 3 * 64 + 0 * 64));  \
    U_01 = svld1(pg1, (float64_t*)(baseU + 2 * 3 * 64 + -5 * 64));  \
    U_11 = svld1(pg1, (float64_t*)(baseU + 2 * 3 * 64 + -2 * 64));  \
    U_21 = svld1(pg1, (float64_t*)(baseU + 2 * 3 * 64 + 1 * 64));  \
    UChi_00 = svcmla_x(pg1, zero0, U_00, Chi_00, 0); \
    UChi_10 = svcmla_x(pg1, zero0, U_00, Chi_10, 0); \
    UChi_01 = svcmla_x(pg1, zero0, U_10, Chi_00, 0); \
    UChi_11 = svcmla_x(pg1, zero0, U_10, Chi_10, 0); \
    UChi_02 = svcmla_x(pg1, zero0, U_20, Chi_00, 0); \
    UChi_12 = svcmla_x(pg1, zero0, U_20, Chi_10, 0); \
    UChi_00 = svcmla_x(pg1, UChi_00, U_00, Chi_00, 90); \
    UChi_10 = svcmla_x(pg1, UChi_10, U_00, Chi_10, 90); \
    UChi_01 = svcmla_x(pg1, UChi_01, U_10, Chi_00, 90); \
    UChi_11 = svcmla_x(pg1, UChi_11, U_10, Chi_10, 90); \
    UChi_02 = svcmla_x(pg1, UChi_02, U_20, Chi_00, 90); \
    UChi_12 = svcmla_x(pg1, UChi_12, U_20, Chi_10, 90); \
    U_00 = svld1(pg1, (float64_t*)(baseU + 2 * 3 * 64 + -4 * 64));  \
    U_10 = svld1(pg1, (float64_t*)(baseU + 2 * 3 * 64 + -1 * 64));  \
    U_20 = svld1(pg1, (float64_t*)(baseU + 2 * 3 * 64 + 2 * 64));  \
    UChi_00 = svcmla_x(pg1, UChi_00, U_01, Chi_01, 0); \
    UChi_10 = svcmla_x(pg1, UChi_10, U_01, Chi_11, 0); \
    UChi_01 = svcmla_x(pg1, UChi_01, U_11, Chi_01, 0); \
    UChi_11 = svcmla_x(pg1, UChi_11, U_11, Chi_11, 0); \
    UChi_02 = svcmla_x(pg1, UChi_02, U_21, Chi_01, 0); \
    UChi_12 = svcmla_x(pg1, UChi_12, U_21, Chi_11, 0); \
    UChi_00 = svcmla_x(pg1, UChi_00, U_01, Chi_01, 90); \
    UChi_10 = svcmla_x(pg1, UChi_10, U_01, Chi_11, 90); \
    UChi_01 = svcmla_x(pg1, UChi_01, U_11, Chi_01, 90); \
    UChi_11 = svcmla_x(pg1, UChi_11, U_11, Chi_11, 90); \
    UChi_02 = svcmla_x(pg1, UChi_02, U_21, Chi_01, 90); \
    UChi_12 = svcmla_x(pg1, UChi_12, U_21, Chi_11, 90); \
    UChi_00 = svcmla_x(pg1, UChi_00, U_00, Chi_02, 0); \
    UChi_10 = svcmla_x(pg1, UChi_10, U_00, Chi_12, 0); \
    UChi_01 = svcmla_x(pg1, UChi_01, U_10, Chi_02, 0); \
    UChi_11 = svcmla_x(pg1, UChi_11, U_10, Chi_12, 0); \
    UChi_02 = svcmla_x(pg1, UChi_02, U_20, Chi_02, 0); \
    UChi_12 = svcmla_x(pg1, UChi_12, U_20, Chi_12, 0); \
    UChi_00 = svcmla_x(pg1, UChi_00, U_00, Chi_02, 90); \
    UChi_10 = svcmla_x(pg1, UChi_10, U_00, Chi_12, 90); \
    UChi_01 = svcmla_x(pg1, UChi_01, U_10, Chi_02, 90); \
    UChi_11 = svcmla_x(pg1, UChi_11, U_10, Chi_12, 90); \
    UChi_02 = svcmla_x(pg1, UChi_02, U_20, Chi_02, 90); \
    UChi_12 = svcmla_x(pg1, UChi_12, U_20, Chi_12, 90); \
}
// XP_PROJ
#define XP_PROJ_A64FXd  \
{ \
    Chi_00 = svcadd_x(pg1, Chimu_00, Chimu_30, 90);   \
    Chi_01 = svcadd_x(pg1, Chimu_01, Chimu_31, 90);   \
    Chi_02 = svcadd_x(pg1, Chimu_02, Chimu_32, 90);   \
    Chi_10 = svcadd_x(pg1, Chimu_10, Chimu_20, 90);   \
    Chi_11 = svcadd_x(pg1, Chimu_11, Chimu_21, 90);   \
    Chi_12 = svcadd_x(pg1, Chimu_12, Chimu_22, 90);   \
}
// XP_RECON
#define XP_RECON_A64FXd  \
    result_20 = svcadd_x(pg1, zero0, UChi_10, 270);   \
    result_21 = svcadd_x(pg1, zero0, UChi_11, 270);   \
    result_22 = svcadd_x(pg1, zero0, UChi_12, 270);   \
    result_30 = svcadd_x(pg1, zero0, UChi_00, 270);   \
    result_31 = svcadd_x(pg1, zero0, UChi_01, 270);   \
    result_32 = svcadd_x(pg1, zero0, UChi_02, 270);   \
    result_00 = UChi_00;        \
    result_01 = UChi_01;        \
    result_02 = UChi_02;        \
    result_10 = UChi_10;        \
    result_11 = UChi_11;        \
    result_12 = UChi_12;        

// XP_RECON_ACCUM
#define XP_RECON_ACCUM_A64FXd  \
    result_30 = svcadd_x(pg1, result_30, UChi_00, 270);   \
    result_00 = svadd_x(pg1, result_00, UChi_00); \
    result_31 = svcadd_x(pg1, result_31, UChi_01, 270);   \
    result_01 = svadd_x(pg1, result_01, UChi_01); \
    result_32 = svcadd_x(pg1, result_32, UChi_02, 270);   \
    result_02 = svadd_x(pg1, result_02, UChi_02); \
    result_20 = svcadd_x(pg1, result_20, UChi_10, 270);   \
    result_10 = svadd_x(pg1, result_10, UChi_10); \
    result_21 = svcadd_x(pg1, result_21, UChi_11, 270);   \
    result_11 = svadd_x(pg1, result_11, UChi_11); \
    result_22 = svcadd_x(pg1, result_22, UChi_12, 270);   \
    result_12 = svadd_x(pg1, result_12, UChi_12); 

// YP_PROJ
#define YP_PROJ_A64FXd  \
{ \
    Chi_00 = svsub_x(pg1, Chimu_00, Chimu_30);  \
    Chi_01 = svsub_x(pg1, Chimu_01, Chimu_31);  \
    Chi_02 = svsub_x(pg1, Chimu_02, Chimu_32);  \
    Chi_10 = svadd_x(pg1, Chimu_10, Chimu_20);  \
    Chi_11 = svadd_x(pg1, Chimu_11, Chimu_21);  \
    Chi_12 = svadd_x(pg1, Chimu_12, Chimu_22);  \
}
// ZP_PROJ
#define ZP_PROJ_A64FXd  \
{ \
    Chi_00 = svcadd_x(pg1, Chimu_00, Chimu_20, 90);   \
    Chi_01 = svcadd_x(pg1, Chimu_01, Chimu_21, 90);   \
    Chi_02 = svcadd_x(pg1, Chimu_02, Chimu_22, 90);   \
    Chi_10 = svcadd_x(pg1, Chimu_10, Chimu_30, 270);   \
    Chi_11 = svcadd_x(pg1, Chimu_11, Chimu_31, 270);   \
    Chi_12 = svcadd_x(pg1, Chimu_12, Chimu_32, 270);   \
}
// TP_PROJ
#define TP_PROJ_A64FXd  \
{ \
    Chi_00 = svadd_x(pg1, Chimu_00, Chimu_20);  \
    Chi_01 = svadd_x(pg1, Chimu_01, Chimu_21);  \
    Chi_02 = svadd_x(pg1, Chimu_02, Chimu_22);  \
    Chi_10 = svadd_x(pg1, Chimu_10, Chimu_30);  \
    Chi_11 = svadd_x(pg1, Chimu_11, Chimu_31);  \
    Chi_12 = svadd_x(pg1, Chimu_12, Chimu_32);  \
}
// XM_PROJ
#define XM_PROJ_A64FXd  \
{ \
    Chi_00 = svcadd_x(pg1, Chimu_00, Chimu_30, 270);   \
    Chi_01 = svcadd_x(pg1, Chimu_01, Chimu_31, 270);   \
    Chi_02 = svcadd_x(pg1, Chimu_02, Chimu_32, 270);   \
    Chi_10 = svcadd_x(pg1, Chimu_10, Chimu_20, 270);   \
    Chi_11 = svcadd_x(pg1, Chimu_11, Chimu_21, 270);   \
    Chi_12 = svcadd_x(pg1, Chimu_12, Chimu_22, 270);   \
}
// XM_RECON
#define XM_RECON_A64FXd  \
    result_20 = svcadd_x(pg1, zero0, UChi_10, 90);   \
    result_21 = svcadd_x(pg1, zero0, UChi_11, 90);   \
    result_22 = svcadd_x(pg1, zero0, UChi_12, 90);   \
    result_30 = svcadd_x(pg1, zero0, UChi_00, 90);   \
    result_31 = svcadd_x(pg1, zero0, UChi_01, 90);   \
    result_32 = svcadd_x(pg1, zero0, UChi_02, 90);   \
    result_00 = UChi_00;        \
    result_01 = UChi_01;        \
    result_02 = UChi_02;        \
    result_10 = UChi_10;        \
    result_11 = UChi_11;        \
    result_12 = UChi_12;        

// YM_PROJ
#define YM_PROJ_A64FXd  \
{ \
    Chi_00 = svadd_x(pg1, Chimu_00, Chimu_30);  \
    Chi_01 = svadd_x(pg1, Chimu_01, Chimu_31);  \
    Chi_02 = svadd_x(pg1, Chimu_02, Chimu_32);  \
    Chi_10 = svsub_x(pg1, Chimu_10, Chimu_20);  \
    Chi_11 = svsub_x(pg1, Chimu_11, Chimu_21);  \
    Chi_12 = svsub_x(pg1, Chimu_12, Chimu_22);  \
}
// ZM_PROJ
#define ZM_PROJ_A64FXd  \
{ \
    Chi_00 = svcadd_x(pg1, Chimu_00, Chimu_20, 270);   \
    Chi_01 = svcadd_x(pg1, Chimu_01, Chimu_21, 270);   \
    Chi_02 = svcadd_x(pg1, Chimu_02, Chimu_22, 270);   \
    Chi_10 = svcadd_x(pg1, Chimu_10, Chimu_30, 90);   \
    Chi_11 = svcadd_x(pg1, Chimu_11, Chimu_31, 90);   \
    Chi_12 = svcadd_x(pg1, Chimu_12, Chimu_32, 90);   \
}
// TM_PROJ
#define TM_PROJ_A64FXd  \
{ \
    Chi_00 = svsub_x(pg1, Chimu_00, Chimu_20);  \
    Chi_01 = svsub_x(pg1, Chimu_01, Chimu_21);  \
    Chi_02 = svsub_x(pg1, Chimu_02, Chimu_22);  \
    Chi_10 = svsub_x(pg1, Chimu_10, Chimu_30);  \
    Chi_11 = svsub_x(pg1, Chimu_11, Chimu_31);  \
    Chi_12 = svsub_x(pg1, Chimu_12, Chimu_32);  \
}
// XM_RECON_ACCUM
#define XM_RECON_ACCUM_A64FXd  \
    result_30 = svcadd_x(pg1, result_30, UChi_00, 90);   \
    result_31 = svcadd_x(pg1, result_31, UChi_01, 90);   \
    result_32 = svcadd_x(pg1, result_32, UChi_02, 90);   \
    result_20 = svcadd_x(pg1, result_20, UChi_10, 90);   \
    result_21 = svcadd_x(pg1, result_21, UChi_11, 90);   \
    result_22 = svcadd_x(pg1, result_22, UChi_12, 90);   \
    result_00 = svadd_x(pg1, result_00, UChi_00); \
    result_01 = svadd_x(pg1, result_01, UChi_01); \
    result_02 = svadd_x(pg1, result_02, UChi_02); \
    result_10 = svadd_x(pg1, result_10, UChi_10); \
    result_11 = svadd_x(pg1, result_11, UChi_11); \
    result_12 = svadd_x(pg1, result_12, UChi_12); 

// YP_RECON_ACCUM
#define YP_RECON_ACCUM_A64FXd  \
    result_00 = svadd_x(pg1, result_00, UChi_00); \
    result_30 = svsub_x(pg1, result_30, UChi_00); \
    result_01 = svadd_x(pg1, result_01, UChi_01); \
    result_31 = svsub_x(pg1, result_31, UChi_01); \
    result_02 = svadd_x(pg1, result_02, UChi_02); \
    result_32 = svsub_x(pg1, result_32, UChi_02); \
    result_10 = svadd_x(pg1, result_10, UChi_10); \
    result_20 = svadd_x(pg1, result_20, UChi_10); \
    result_11 = svadd_x(pg1, result_11, UChi_11); \
    result_21 = svadd_x(pg1, result_21, UChi_11); \
    result_12 = svadd_x(pg1, result_12, UChi_12); \
    result_22 = svadd_x(pg1, result_22, UChi_12); 

// YM_RECON_ACCUM
#define YM_RECON_ACCUM_A64FXd  \
    result_00 = svadd_x(pg1, result_00, UChi_00); \
    result_30 = svadd_x(pg1, result_30, UChi_00); \
    result_01 = svadd_x(pg1, result_01, UChi_01); \
    result_31 = svadd_x(pg1, result_31, UChi_01); \
    result_02 = svadd_x(pg1, result_02, UChi_02); \
    result_32 = svadd_x(pg1, result_32, UChi_02); \
    result_10 = svadd_x(pg1, result_10, UChi_10); \
    result_20 = svsub_x(pg1, result_20, UChi_10); \
    result_11 = svadd_x(pg1, result_11, UChi_11); \
    result_21 = svsub_x(pg1, result_21, UChi_11); \
    result_12 = svadd_x(pg1, result_12, UChi_12); \
    result_22 = svsub_x(pg1, result_22, UChi_12); 

// ZP_RECON_ACCUM
#define ZP_RECON_ACCUM_A64FXd  \
    result_20 = svcadd_x(pg1, result_20, UChi_00, 270);   \
    result_00 = svadd_x(pg1, result_00, UChi_00); \
    result_21 = svcadd_x(pg1, result_21, UChi_01, 270);   \
    result_01 = svadd_x(pg1, result_01, UChi_01); \
    result_22 = svcadd_x(pg1, result_22, UChi_02, 270);   \
    result_02 = svadd_x(pg1, result_02, UChi_02); \
    result_30 = svcadd_x(pg1, result_30, UChi_10, 90);   \
    result_10 = svadd_x(pg1, result_10, UChi_10); \
    result_31 = svcadd_x(pg1, result_31, UChi_11, 90);   \
    result_11 = svadd_x(pg1, result_11, UChi_11); \
    result_32 = svcadd_x(pg1, result_32, UChi_12, 90);   \
    result_12 = svadd_x(pg1, result_12, UChi_12); 

// ZM_RECON_ACCUM
#define ZM_RECON_ACCUM_A64FXd  \
    result_20 = svcadd_x(pg1, result_20, UChi_00, 90);   \
    result_00 = svadd_x(pg1, result_00, UChi_00); \
    result_21 = svcadd_x(pg1, result_21, UChi_01, 90);   \
    result_01 = svadd_x(pg1, result_01, UChi_01); \
    result_22 = svcadd_x(pg1, result_22, UChi_02, 90);   \
    result_02 = svadd_x(pg1, result_02, UChi_02); \
    result_30 = svcadd_x(pg1, result_30, UChi_10, 270);   \
    result_10 = svadd_x(pg1, result_10, UChi_10); \
    result_31 = svcadd_x(pg1, result_31, UChi_11, 270);   \
    result_11 = svadd_x(pg1, result_11, UChi_11); \
    result_32 = svcadd_x(pg1, result_32, UChi_12, 270);   \
    result_12 = svadd_x(pg1, result_12, UChi_12); 

// TP_RECON_ACCUM
#define TP_RECON_ACCUM_A64FXd  \
    result_00 = svadd_x(pg1, result_00, UChi_00); \
    result_20 = svadd_x(pg1, result_20, UChi_00); \
    result_01 = svadd_x(pg1, result_01, UChi_01); \
    result_21 = svadd_x(pg1, result_21, UChi_01); \
    result_02 = svadd_x(pg1, result_02, UChi_02); \
    result_22 = svadd_x(pg1, result_22, UChi_02); \
    result_10 = svadd_x(pg1, result_10, UChi_10); \
    result_30 = svadd_x(pg1, result_30, UChi_10); \
    result_11 = svadd_x(pg1, result_11, UChi_11); \
    result_31 = svadd_x(pg1, result_31, UChi_11); \
    result_12 = svadd_x(pg1, result_12, UChi_12); \
    result_32 = svadd_x(pg1, result_32, UChi_12); 

// TM_RECON_ACCUM
#define TM_RECON_ACCUM_A64FXd  \
    result_00 = svadd_x(pg1, result_00, UChi_00); \
    result_20 = svsub_x(pg1, result_20, UChi_00); \
    result_01 = svadd_x(pg1, result_01, UChi_01); \
    result_21 = svsub_x(pg1, result_21, UChi_01); \
    result_02 = svadd_x(pg1, result_02, UChi_02); \
    result_22 = svsub_x(pg1, result_22, UChi_02); \
    result_10 = svadd_x(pg1, result_10, UChi_10); \
    result_30 = svsub_x(pg1, result_30, UChi_10); \
    result_11 = svadd_x(pg1, result_11, UChi_11); \
    result_31 = svsub_x(pg1, result_31, UChi_11); \
    result_12 = svadd_x(pg1, result_12, UChi_12); \
    result_32 = svsub_x(pg1, result_32, UChi_12); 

// ZERO_PSI
#define ZERO_PSI_A64FXd  \
    result_00 = __svzero(result_00); \
    result_01 = __svzero(result_01); \
    result_02 = __svzero(result_02); \
    result_10 = __svzero(result_10); \
    result_11 = __svzero(result_11); \
    result_12 = __svzero(result_12); \
    result_20 = __svzero(result_20); \
    result_21 = __svzero(result_21); \
    result_22 = __svzero(result_22); \
    result_30 = __svzero(result_30); \
    result_31 = __svzero(result_31); \
    result_32 = __svzero(result_32); 

// ADD_RESULT_INTERNAL
#define ADD_RESULT_INTERNAL_A64FXd  \
    result_00 = svadd_x(pg1, result_00, Chimu_00); \
    result_01 = svadd_x(pg1, result_01, Chimu_01); \
    result_02 = svadd_x(pg1, result_02, Chimu_02); \
    result_10 = svadd_x(pg1, result_10, Chimu_10); \
    result_11 = svadd_x(pg1, result_11, Chimu_11); \
    result_12 = svadd_x(pg1, result_12, Chimu_12); \
    result_20 = svadd_x(pg1, result_20, Chimu_20); \
    result_21 = svadd_x(pg1, result_21, Chimu_21); \
    result_22 = svadd_x(pg1, result_22, Chimu_22); \
    result_30 = svadd_x(pg1, result_30, Chimu_30); \
    result_31 = svadd_x(pg1, result_31, Chimu_31); \
    result_32 = svadd_x(pg1, result_32, Chimu_32); 

