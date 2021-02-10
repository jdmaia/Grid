   /*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid 

    Source file: ./lib/qcd/action/fermion/WilsonKernelsHand.cc

    Copyright (C) 2015

Author: Peter Boyle <paboyle@ph.ed.ac.uk>
Author: paboyle <paboyle@ph.ed.ac.uk>

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

#pragma once

#include <Grid/qcd/action/fermion/FermionCore.h>


#undef LOAD_CHIMU  
#undef LOAD_CHI 
#undef MULT_2SPIN
#undef PERMUTE_DIR
#undef XP_PROJ  
#undef YP_PROJ  
#undef ZP_PROJ  
#undef TP_PROJ  
#undef XM_PROJ  
#undef YM_PROJ  
#undef ZM_PROJ  
#undef TM_PROJ  
#undef XP_RECON 
#undef XP_RECON_ACCUM 
#undef XM_RECON 
#undef XM_RECON_ACCUM 
#undef YP_RECON_ACCUM 
#undef YM_RECON_ACCUM 
#undef ZP_RECON_ACCUM 
#undef ZM_RECON_ACCUM 
#undef TP_RECON_ACCUM 
#undef TM_RECON_ACCUM 
#undef ZERO_RESULT				 
#undef Chimu_00
#undef Chimu_01
#undef Chimu_02
#undef Chimu_10
#undef Chimu_11
#undef Chimu_12
#undef Chimu_20
#undef Chimu_21
#undef Chimu_22
#undef Chimu_30
#undef Chimu_31
#undef Chimu_32
#undef HAND_STENCIL_LEG
#undef HAND_STENCIL_LEG_INT
#undef HAND_STENCIL_LEG_EXT
#undef HAND_RESULT
#undef HAND_RESULT_INT
#undef HAND_RESULT_EXT

#define REGISTER

#ifdef GRID_SIMT
#define LOAD_CHIMU(ptype)		\
  {const SiteSpinor & ref (in[offset]);	\
    Chimu_00=coalescedReadPermute<ptype>(ref()(0)(0),perm);	\
    Chimu_01=coalescedReadPermute<ptype>(ref()(0)(1),perm);	\
    Chimu_02=coalescedReadPermute<ptype>(ref()(0)(2),perm);	\
    Chimu_10=coalescedReadPermute<ptype>(ref()(1)(0),perm);	\
    Chimu_11=coalescedReadPermute<ptype>(ref()(1)(1),perm);	\
    Chimu_12=coalescedReadPermute<ptype>(ref()(1)(2),perm);	\
    Chimu_20=coalescedReadPermute<ptype>(ref()(2)(0),perm);	\
    Chimu_21=coalescedReadPermute<ptype>(ref()(2)(1),perm);	\
    Chimu_22=coalescedReadPermute<ptype>(ref()(2)(2),perm);	\
    Chimu_30=coalescedReadPermute<ptype>(ref()(3)(0),perm);	\
    Chimu_31=coalescedReadPermute<ptype>(ref()(3)(1),perm);	\
    Chimu_32=coalescedReadPermute<ptype>(ref()(3)(2),perm);	}

#define PERMUTE_DIR(dir) ;
#else
#define LOAD_CHIMU(ptype)		\
  {const SiteSpinor & ref (in[offset]);	\
    Chimu_00=coalescedRead(ref()(0)(0));	\
    Chimu_01=coalescedRead(ref()(0)(1));	\
    Chimu_02=coalescedRead(ref()(0)(2));	\
    Chimu_10=coalescedRead(ref()(1)(0));	\
    Chimu_11=coalescedRead(ref()(1)(1));	\
    Chimu_12=coalescedRead(ref()(1)(2));	\
    Chimu_20=coalescedRead(ref()(2)(0));	\
    Chimu_21=coalescedRead(ref()(2)(1));	\
    Chimu_22=coalescedRead(ref()(2)(2));	\
    Chimu_30=coalescedRead(ref()(3)(0));	\
    Chimu_31=coalescedRead(ref()(3)(1));	\
    Chimu_32=coalescedRead(ref()(3)(2));	}

#define PERMUTE_DIR(dir)			\
  permute##dir(Chi_00,Chi_00);	\
      permute##dir(Chi_01,Chi_01);\
      permute##dir(Chi_02,Chi_02);\
      permute##dir(Chi_10,Chi_10);	\
      permute##dir(Chi_11,Chi_11);\
      permute##dir(Chi_12,Chi_12);
#endif

#define MULT_2SPIN(A)\
  {auto & ref(U[sU](A));					\
  U_00=coalescedRead(ref()(0,0));				\
  U_10=coalescedRead(ref()(1,0));					\
  U_20=coalescedRead(ref()(2,0));					\
  U_01=coalescedRead(ref()(0,1));					\
  U_11=coalescedRead(ref()(1,1));					\
  U_21=coalescedRead(ref()(2,1));					\
    UChi_00 = U_00*Chi_00;					\
    UChi_10 = U_00*Chi_10;					\
    UChi_01 = U_10*Chi_00;					\
    UChi_11 = U_10*Chi_10;					\
    UChi_02 = U_20*Chi_00;					\
    UChi_12 = U_20*Chi_10;					\
    UChi_00+= U_01*Chi_01;					\
    UChi_10+= U_01*Chi_11;					\
    UChi_01+= U_11*Chi_01;					\
    UChi_11+= U_11*Chi_11;					\
    UChi_02+= U_21*Chi_01;					\
    UChi_12+= U_21*Chi_11;					\
    U_00=coalescedRead(ref()(0,2));				\
    U_10=coalescedRead(ref()(1,2));				\
    U_20=coalescedRead(ref()(2,2));				\
    UChi_00+= U_00*Chi_02;					\
    UChi_10+= U_00*Chi_12;					\
    UChi_01+= U_10*Chi_02;					\
    UChi_11+= U_10*Chi_12;					\
    UChi_02+= U_20*Chi_02;					\
    UChi_12+= U_20*Chi_12;}

#define LOAD_CHI				\
  {const SiteHalfSpinor &ref(buf[offset]);	\
    Chi_00 = coalescedRead(ref()(0)(0));	\
    Chi_01 = coalescedRead(ref()(0)(1));	\
    Chi_02 = coalescedRead(ref()(0)(2));	\
    Chi_10 = coalescedRead(ref()(1)(0));	\
    Chi_11 = coalescedRead(ref()(1)(1));	\
    Chi_12 = coalescedRead(ref()(1)(2));}

//      hspin(0)=fspin(0)+timesI(fspin(3));
//      hspin(1)=fspin(1)+timesI(fspin(2));
#define XP_PROJ \
    Chi_00 = Chimu_00+timesI(Chimu_30);\
    Chi_01 = Chimu_01+timesI(Chimu_31);\
    Chi_02 = Chimu_02+timesI(Chimu_32);\
    Chi_10 = Chimu_10+timesI(Chimu_20);\
    Chi_11 = Chimu_11+timesI(Chimu_21);\
    Chi_12 = Chimu_12+timesI(Chimu_22);

#define YP_PROJ \
    Chi_00 = Chimu_00-Chimu_30;\
    Chi_01 = Chimu_01-Chimu_31;\
    Chi_02 = Chimu_02-Chimu_32;\
    Chi_10 = Chimu_10+Chimu_20;\
    Chi_11 = Chimu_11+Chimu_21;\
    Chi_12 = Chimu_12+Chimu_22;

#define ZP_PROJ \
  Chi_00 = Chimu_00+timesI(Chimu_20);		\
  Chi_01 = Chimu_01+timesI(Chimu_21);		\
  Chi_02 = Chimu_02+timesI(Chimu_22);		\
  Chi_10 = Chimu_10-timesI(Chimu_30);		\
  Chi_11 = Chimu_11-timesI(Chimu_31);		\
  Chi_12 = Chimu_12-timesI(Chimu_32);

#define TP_PROJ \
  Chi_00 = Chimu_00+Chimu_20;		\
  Chi_01 = Chimu_01+Chimu_21;		\
  Chi_02 = Chimu_02+Chimu_22;		\
  Chi_10 = Chimu_10+Chimu_30;		\
  Chi_11 = Chimu_11+Chimu_31;		\
  Chi_12 = Chimu_12+Chimu_32;


//      hspin(0)=fspin(0)-timesI(fspin(3));
//      hspin(1)=fspin(1)-timesI(fspin(2));
#define XM_PROJ \
    Chi_00 = Chimu_00-timesI(Chimu_30);\
    Chi_01 = Chimu_01-timesI(Chimu_31);\
    Chi_02 = Chimu_02-timesI(Chimu_32);\
    Chi_10 = Chimu_10-timesI(Chimu_20);\
    Chi_11 = Chimu_11-timesI(Chimu_21);\
    Chi_12 = Chimu_12-timesI(Chimu_22);

#define YM_PROJ \
    Chi_00 = Chimu_00+Chimu_30;\
    Chi_01 = Chimu_01+Chimu_31;\
    Chi_02 = Chimu_02+Chimu_32;\
    Chi_10 = Chimu_10-Chimu_20;\
    Chi_11 = Chimu_11-Chimu_21;\
    Chi_12 = Chimu_12-Chimu_22;

#define ZM_PROJ \
  Chi_00 = Chimu_00-timesI(Chimu_20);		\
  Chi_01 = Chimu_01-timesI(Chimu_21);		\
  Chi_02 = Chimu_02-timesI(Chimu_22);		\
  Chi_10 = Chimu_10+timesI(Chimu_30);		\
  Chi_11 = Chimu_11+timesI(Chimu_31);		\
  Chi_12 = Chimu_12+timesI(Chimu_32);

#define TM_PROJ \
  Chi_00 = Chimu_00-Chimu_20;		\
  Chi_01 = Chimu_01-Chimu_21;		\
  Chi_02 = Chimu_02-Chimu_22;		\
  Chi_10 = Chimu_10-Chimu_30;		\
  Chi_11 = Chimu_11-Chimu_31;		\
  Chi_12 = Chimu_12-Chimu_32;

//      fspin(0)=hspin(0);
//      fspin(1)=hspin(1);
//      fspin(2)=timesMinusI(hspin(1));
//      fspin(3)=timesMinusI(hspin(0));
#define XP_RECON\
  result_00 = UChi_00;\
  result_01 = UChi_01;\
  result_02 = UChi_02;\
  result_10 = UChi_10;\
  result_11 = UChi_11;\
  result_12 = UChi_12;\
  result_20 = timesMinusI(UChi_10);\
  result_21 = timesMinusI(UChi_11);\
  result_22 = timesMinusI(UChi_12);\
  result_30 = timesMinusI(UChi_00);\
  result_31 = timesMinusI(UChi_01);\
  result_32 = timesMinusI(UChi_02);

#define XP_RECON_ACCUM\
  result_00+=UChi_00;\
  result_01+=UChi_01;\
  result_02+=UChi_02;\
  result_10+=UChi_10;\
  result_11+=UChi_11;\
  result_12+=UChi_12;\
  result_20-=timesI(UChi_10);\
  result_21-=timesI(UChi_11);\
  result_22-=timesI(UChi_12);\
  result_30-=timesI(UChi_00);\
  result_31-=timesI(UChi_01);\
  result_32-=timesI(UChi_02);

#define XM_RECON\
  result_00 = UChi_00;\
  result_01 = UChi_01;\
  result_02 = UChi_02;\
  result_10 = UChi_10;\
  result_11 = UChi_11;\
  result_12 = UChi_12;\
  result_20 = timesI(UChi_10);\
  result_21 = timesI(UChi_11);\
  result_22 = timesI(UChi_12);\
  result_30 = timesI(UChi_00);\
  result_31 = timesI(UChi_01);\
  result_32 = timesI(UChi_02);

#define XM_RECON_ACCUM\
  result_00+= UChi_00;\
  result_01+= UChi_01;\
  result_02+= UChi_02;\
  result_10+= UChi_10;\
  result_11+= UChi_11;\
  result_12+= UChi_12;\
  result_20+= timesI(UChi_10);\
  result_21+= timesI(UChi_11);\
  result_22+= timesI(UChi_12);\
  result_30+= timesI(UChi_00);\
  result_31+= timesI(UChi_01);\
  result_32+= timesI(UChi_02);

#define YP_RECON_ACCUM\
  result_00+= UChi_00;\
  result_01+= UChi_01;\
  result_02+= UChi_02;\
  result_10+= UChi_10;\
  result_11+= UChi_11;\
  result_12+= UChi_12;\
  result_20+= UChi_10;\
  result_21+= UChi_11;\
  result_22+= UChi_12;\
  result_30-= UChi_00;\
  result_31-= UChi_01;\
  result_32-= UChi_02;

#define YM_RECON_ACCUM\
  result_00+= UChi_00;\
  result_01+= UChi_01;\
  result_02+= UChi_02;\
  result_10+= UChi_10;\
  result_11+= UChi_11;\
  result_12+= UChi_12;\
  result_20-= UChi_10;\
  result_21-= UChi_11;\
  result_22-= UChi_12;\
  result_30+= UChi_00;\
  result_31+= UChi_01;\
  result_32+= UChi_02;

#define ZP_RECON_ACCUM\
  result_00+= UChi_00;\
  result_01+= UChi_01;\
  result_02+= UChi_02;\
  result_10+= UChi_10;\
  result_11+= UChi_11;\
  result_12+= UChi_12;\
  result_20-= timesI(UChi_00);			\
  result_21-= timesI(UChi_01);			\
  result_22-= timesI(UChi_02);			\
  result_30+= timesI(UChi_10);			\
  result_31+= timesI(UChi_11);			\
  result_32+= timesI(UChi_12);

#define ZM_RECON_ACCUM\
  result_00+= UChi_00;\
  result_01+= UChi_01;\
  result_02+= UChi_02;\
  result_10+= UChi_10;\
  result_11+= UChi_11;\
  result_12+= UChi_12;\
  result_20+= timesI(UChi_00);			\
  result_21+= timesI(UChi_01);			\
  result_22+= timesI(UChi_02);			\
  result_30-= timesI(UChi_10);			\
  result_31-= timesI(UChi_11);			\
  result_32-= timesI(UChi_12);

#define TP_RECON_ACCUM\
  result_00+= UChi_00;\
  result_01+= UChi_01;\
  result_02+= UChi_02;\
  result_10+= UChi_10;\
  result_11+= UChi_11;\
  result_12+= UChi_12;\
  result_20+= UChi_00;			\
  result_21+= UChi_01;			\
  result_22+= UChi_02;			\
  result_30+= UChi_10;			\
  result_31+= UChi_11;			\
  result_32+= UChi_12;

#define TM_RECON_ACCUM\
  result_00+= UChi_00;\
  result_01+= UChi_01;\
  result_02+= UChi_02;\
  result_10+= UChi_10;\
  result_11+= UChi_11;\
  result_12+= UChi_12;\
  result_20-= UChi_00;	\
  result_21-= UChi_01;	\
  result_22-= UChi_02;	\
  result_30-= UChi_10;	\
  result_31-= UChi_11;	\
  result_32-= UChi_12;

#define HAND_STENCIL_LEGA(PROJ,PERM,DIR,RECON)	\
  SE=&st_p[DIR+8*ss];			\
  ptype=st_perm[DIR];			\
  offset = SE->_offset;				\
  local  = SE->_is_local;			\
  perm   = SE->_permute;			\
  if ( local ) {				\
    LOAD_CHIMU(PERM);				\
    PROJ;					\
    if ( perm) {				\
      PERMUTE_DIR(PERM);			\
    }						\
  } else {					\
    LOAD_CHI;					\
  }						\
  MULT_2SPIN(DIR);				\
  RECON;					

#define HAND_STENCIL_LEG(PROJ,PERM,DIR,RECON)	\
  SE=&st_p[DIR+8*ss];			\
  ptype=st_perm[DIR];			\
  offset = SE->_offset;				\
  local  = SE->_is_local;			\
  perm   = SE->_permute;			\
  LOAD_CHIMU(PERM);				\
  PROJ;						\
  MULT_2SPIN(DIR);				\
  RECON;					


#define HAND_STENCIL_LEG_INT(PROJ,PERM,DIR,RECON)	\
  SE=&st_p[DIR+8*ss];					\
  ptype=st_perm[DIR];					\
  offset = SE->_offset;				\
  local  = SE->_is_local;			\
  perm   = SE->_permute;			\
  if ( local ) {				\
    LOAD_CHIMU;					\
    PROJ;					\
    if ( perm) {				\
      PERMUTE_DIR(PERM);			\
    }						\
  } else if ( st.same_node[DIR] ) {		\
    LOAD_CHI;					\
  }						\
  if (local || st.same_node[DIR] ) {		\
    MULT_2SPIN(DIR);				\
    RECON;					\
  }

#define HAND_STENCIL_LEG_EXT(PROJ,PERM,DIR,RECON)	\
  SE=st.GetEntry(ptype,DIR,ss);			\
  offset = SE->_offset;				\
  if((!SE->_is_local)&&(!st.same_node[DIR]) ) {	\
    LOAD_CHI;					\
    MULT_2SPIN(DIR);				\
    RECON;					\
    nmu++;					\
  }

#define HAND_RESULT(ss)				\
  {						\
    SiteSpinor & ref (out[ss]);			\
    coalescedWrite(ref()(0)(0),result_00);		\
    coalescedWrite(ref()(0)(1),result_01);		\
    coalescedWrite(ref()(0)(2),result_02);		\
    coalescedWrite(ref()(1)(0),result_10);		\
    coalescedWrite(ref()(1)(1),result_11);		\
    coalescedWrite(ref()(1)(2),result_12);		\
    coalescedWrite(ref()(2)(0),result_20);		\
    coalescedWrite(ref()(2)(1),result_21);		\
    coalescedWrite(ref()(2)(2),result_22);		\
    coalescedWrite(ref()(3)(0),result_30);		\
    coalescedWrite(ref()(3)(1),result_31);		\
    coalescedWrite(ref()(3)(2),result_32);		\
  }

#define HAND_RESULT_EXT(ss)			\
  if (nmu){					\
    SiteSpinor & ref (out[ss]);		\
    ref()(0)(0)+=result_00;		\
    ref()(0)(1)+=result_01;		\
    ref()(0)(2)+=result_02;		\
    ref()(1)(0)+=result_10;		\
    ref()(1)(1)+=result_11;		\
    ref()(1)(2)+=result_12;		\
    ref()(2)(0)+=result_20;		\
    ref()(2)(1)+=result_21;		\
    ref()(2)(2)+=result_22;		\
    ref()(3)(0)+=result_30;		\
    ref()(3)(1)+=result_31;		\
    ref()(3)(2)+=result_32;		\
  }

#define HAND_DECLARATIONS(Simd)			\
  Simd result_00;				\
  Simd result_01;				\
  Simd result_02;				\
  Simd result_10;				\
  Simd result_11;				\
  Simd result_12;				\
  Simd result_20;				\
  Simd result_21;				\
  Simd result_22;				\
  Simd result_30;				\
  Simd result_31;				\
  Simd result_32;				\
  Simd Chi_00;					\
  Simd Chi_01;					\
  Simd Chi_02;					\
  Simd Chi_10;					\
  Simd Chi_11;					\
  Simd Chi_12;					\
  Simd UChi_00;					\
  Simd UChi_01;					\
  Simd UChi_02;					\
  Simd UChi_10;					\
  Simd UChi_11;					\
  Simd UChi_12;					\
  Simd U_00;					\
  Simd U_10;					\
  Simd U_20;					\
  Simd U_01;					\
  Simd U_11;					\
  Simd U_21;

#define ZERO_RESULT				\
  result_00=Zero();				\
  result_01=Zero();				\
  result_02=Zero();				\
  result_10=Zero();				\
  result_11=Zero();				\
  result_12=Zero();				\
  result_20=Zero();				\
  result_21=Zero();				\
  result_22=Zero();				\
  result_30=Zero();				\
  result_31=Zero();				\
  result_32=Zero();			

#define Chimu_00 Chi_00
#define Chimu_01 Chi_01
#define Chimu_02 Chi_02
#define Chimu_10 Chi_10
#define Chimu_11 Chi_11
#define Chimu_12 Chi_12
#define Chimu_20 UChi_00
#define Chimu_21 UChi_01
#define Chimu_22 UChi_02
#define Chimu_30 UChi_10
#define Chimu_31 UChi_11
#define Chimu_32 UChi_12

NAMESPACE_BEGIN(Grid);

template<class Impl> accelerator_inline void 
WilsonKernels<Impl>::HandDhopSiteSycl(StencilVector st_perm,StencilEntry *st_p, SiteDoubledGaugeField *U,SiteHalfSpinor  *buf,
				      int ss,int sU,const SiteSpinor *in, SiteSpinor *out)
{
// T==0, Z==1, Y==2, Z==3 expect 1,2,2,2 simd layout etc...
  typedef typename Simd::scalar_type S;
  typedef typename Simd::vector_type V;
  typedef iSinglet<Simd> vCplx;
  //  typedef decltype( coalescedRead( vCplx()()() )) Simt;
  typedef decltype( coalescedRead( in[0]()(0)(0) )) Simt;

  HAND_DECLARATIONS(Simt);

  int offset,local,perm, ptype;
  StencilEntry *SE;
  HAND_STENCIL_LEG(XM_PROJ,3,Xp,XM_RECON);
  HAND_STENCIL_LEG(YM_PROJ,2,Yp,YM_RECON_ACCUM);
  HAND_STENCIL_LEG(ZM_PROJ,1,Zp,ZM_RECON_ACCUM);
  HAND_STENCIL_LEG(TM_PROJ,0,Tp,TM_RECON_ACCUM);
  HAND_STENCIL_LEG(XP_PROJ,3,Xm,XP_RECON_ACCUM);
  HAND_STENCIL_LEG(YP_PROJ,2,Ym,YP_RECON_ACCUM);
  HAND_STENCIL_LEG(ZP_PROJ,1,Zm,ZP_RECON_ACCUM);
  HAND_STENCIL_LEG(TP_PROJ,0,Tm,TP_RECON_ACCUM);
  HAND_RESULT(ss);
}

////////////// Wilson ; uses this implementation /////////////////////

NAMESPACE_END(Grid);
#undef LOAD_CHIMU  
#undef LOAD_CHI 
#undef MULT_2SPIN
#undef PERMUTE_DIR
#undef XP_PROJ  
#undef YP_PROJ  
#undef ZP_PROJ  
#undef TP_PROJ  
#undef XM_PROJ  
#undef YM_PROJ  
#undef ZM_PROJ  
#undef TM_PROJ  
#undef XP_RECON 
#undef XP_RECON_ACCUM 
#undef XM_RECON 
#undef XM_RECON_ACCUM 
#undef YP_RECON_ACCUM 
#undef YM_RECON_ACCUM 
#undef ZP_RECON_ACCUM 
#undef ZM_RECON_ACCUM 
#undef TP_RECON_ACCUM 
#undef TM_RECON_ACCUM 
#undef ZERO_RESULT				 
#undef Chimu_00
#undef Chimu_01
#undef Chimu_02
#undef Chimu_10
#undef Chimu_11
#undef Chimu_12
#undef Chimu_20
#undef Chimu_21
#undef Chimu_22
#undef Chimu_30
#undef Chimu_31
#undef Chimu_32
#undef HAND_STENCIL_LEG
#undef HAND_STENCIL_LEG_INT
#undef HAND_STENCIL_LEG_EXT
#undef HAND_RESULT
#undef HAND_RESULT_INT
#undef HAND_RESULT_EXT
#undef HAND_DECLARATIONS