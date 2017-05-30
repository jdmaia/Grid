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
#include <Grid/qcd/action/fermion/FermionCore.h>

#define REGISTER

#define LOAD_CHIMU \
  {const SiteSpinor & ref (in._odata[offset]);	\
    Chimu_00=ref()(0)(0);\
    Chimu_01=ref()(0)(1);\
    Chimu_02=ref()(0)(2);\
    Chimu_10=ref()(1)(0);\
    Chimu_11=ref()(1)(1);\
    Chimu_12=ref()(1)(2);\
    Chimu_20=ref()(2)(0);\
    Chimu_21=ref()(2)(1);\
    Chimu_22=ref()(2)(2);\
    Chimu_30=ref()(3)(0);\
    Chimu_31=ref()(3)(1);\
    Chimu_32=ref()(3)(2);}

#define LOAD_CHI\
  {const SiteHalfSpinor &ref(buf[offset]);	\
    Chi_00 = ref()(0)(0);\
    Chi_01 = ref()(0)(1);\
    Chi_02 = ref()(0)(2);\
    Chi_10 = ref()(1)(0);\
    Chi_11 = ref()(1)(1);\
    Chi_12 = ref()(1)(2);}

// To splat or not to splat depends on the implementation
#define MULT_2SPIN(A)\
  {auto & ref(U._odata[sU](A));			\
   Impl::loadLinkElement(U_00,ref()(0,0));	\
   Impl::loadLinkElement(U_10,ref()(1,0));	\
   Impl::loadLinkElement(U_20,ref()(2,0));	\
   Impl::loadLinkElement(U_01,ref()(0,1));	\
   Impl::loadLinkElement(U_11,ref()(1,1));	\
   Impl::loadLinkElement(U_21,ref()(2,1));	\
    UChi_00 = U_00*Chi_00;\
    UChi_10 = U_00*Chi_10;\
    UChi_01 = U_10*Chi_00;\
    UChi_11 = U_10*Chi_10;\
    UChi_02 = U_20*Chi_00;\
    UChi_12 = U_20*Chi_10;\
    UChi_00+= U_01*Chi_01;\
    UChi_10+= U_01*Chi_11;\
    UChi_01+= U_11*Chi_01;\
    UChi_11+= U_11*Chi_11;\
    UChi_02+= U_21*Chi_01;\
    UChi_12+= U_21*Chi_11;\
    Impl::loadLinkElement(U_00,ref()(0,2));	\
    Impl::loadLinkElement(U_10,ref()(1,2));	\
    Impl::loadLinkElement(U_20,ref()(2,2));	\
    UChi_00+= U_00*Chi_02;\
    UChi_10+= U_00*Chi_12;\
    UChi_01+= U_10*Chi_02;\
    UChi_11+= U_10*Chi_12;\
    UChi_02+= U_20*Chi_02;\
    UChi_12+= U_20*Chi_12;}


#define PERMUTE_DIR(dir)			\
      permute##dir(Chi_00,Chi_00);\
      permute##dir(Chi_01,Chi_01);\
      permute##dir(Chi_02,Chi_02);\
      permute##dir(Chi_10,Chi_10);\
      permute##dir(Chi_11,Chi_11);\
      permute##dir(Chi_12,Chi_12);

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

#define HAND_STENCIL_LEG(PROJ,PERM,DIR,RECON)	\
  SE=st.GetEntry(ptype,DIR,ss);			\
  offset = SE->_offset;				\
  local  = SE->_is_local;			\
  perm   = SE->_permute;			\
  if ( local ) {				\
    LOAD_CHIMU;					\
    PROJ;					\
    if ( perm) {				\
      PERMUTE_DIR(PERM);			\
    }						\
  } else {					\
    LOAD_CHI;					\
  }						\
  MULT_2SPIN(DIR);				\
  RECON;					

#define HAND_STENCIL_LEG_INT(PROJ,PERM,DIR,RECON)	\
  SE=st.GetEntry(ptype,DIR,ss);			\
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
    SiteSpinor & ref (out._odata[ss]);		\
    vstream(ref()(0)(0),result_00);		\
    vstream(ref()(0)(1),result_01);		\
    vstream(ref()(0)(2),result_02);		\
    vstream(ref()(1)(0),result_10);		\
    vstream(ref()(1)(1),result_11);		\
    vstream(ref()(1)(2),result_12);		\
    vstream(ref()(2)(0),result_20);		\
    vstream(ref()(2)(1),result_21);		\
    vstream(ref()(2)(2),result_22);		\
    vstream(ref()(3)(0),result_30);		\
    vstream(ref()(3)(1),result_31);		\
    vstream(ref()(3)(2),result_32);		\
  }

#define HAND_RESULT_EXT(ss)			\
  if (nmu){					\
    SiteSpinor & ref (out._odata[ss]);		\
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


#define HAND_DECLARATIONS(a)			\
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
  result_00=zero;				\
  result_01=zero;				\
  result_02=zero;				\
  result_10=zero;				\
  result_11=zero;				\
  result_12=zero;				\
  result_20=zero;				\
  result_21=zero;				\
  result_22=zero;				\
  result_30=zero;				\
  result_31=zero;				\
  result_32=zero;			

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

namespace Grid {
namespace QCD {

template<class Impl> void 
WilsonKernels<Impl>::HandDhopSite(StencilImpl &st,LebesgueOrder &lo,DoubledGaugeField &U,SiteHalfSpinor  *buf,
					  int ss,int sU,const FermionField &in, FermionField &out)
{
// T==0, Z==1, Y==2, Z==3 expect 1,2,2,2 simd layout etc...
  typedef typename Simd::scalar_type S;
  typedef typename Simd::vector_type V;

  HAND_DECLARATIONS(ignore);

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

template<class Impl>
void WilsonKernels<Impl>::HandDhopSiteDag(StencilImpl &st,LebesgueOrder &lo,DoubledGaugeField &U,SiteHalfSpinor *buf,
						  int ss,int sU,const FermionField &in, FermionField &out)
{
  typedef typename Simd::scalar_type S;
  typedef typename Simd::vector_type V;

  HAND_DECLARATIONS(ignore);

  StencilEntry *SE;
  int offset,local,perm, ptype;
  
  HAND_STENCIL_LEG(XP_PROJ,3,Xp,XP_RECON);
  HAND_STENCIL_LEG(YP_PROJ,2,Yp,YP_RECON_ACCUM);
  HAND_STENCIL_LEG(ZP_PROJ,1,Zp,ZP_RECON_ACCUM);
  HAND_STENCIL_LEG(TP_PROJ,0,Tp,TP_RECON_ACCUM);
  HAND_STENCIL_LEG(XM_PROJ,3,Xm,XM_RECON_ACCUM);
  HAND_STENCIL_LEG(YM_PROJ,2,Ym,YM_RECON_ACCUM);
  HAND_STENCIL_LEG(ZM_PROJ,1,Zm,ZM_RECON_ACCUM);
  HAND_STENCIL_LEG(TM_PROJ,0,Tm,TM_RECON_ACCUM);
  HAND_RESULT(ss);
}

template<class Impl> void 
WilsonKernels<Impl>::HandDhopSiteInt(StencilImpl &st,LebesgueOrder &lo,DoubledGaugeField &U,SiteHalfSpinor  *buf,
					  int ss,int sU,const FermionField &in, FermionField &out)
{
// T==0, Z==1, Y==2, Z==3 expect 1,2,2,2 simd layout etc...
  typedef typename Simd::scalar_type S;
  typedef typename Simd::vector_type V;

  HAND_DECLARATIONS(ignore);

  int offset,local,perm, ptype;
  StencilEntry *SE;
  ZERO_RESULT;
  HAND_STENCIL_LEG_INT(XM_PROJ,3,Xp,XM_RECON_ACCUM);
  HAND_STENCIL_LEG_INT(YM_PROJ,2,Yp,YM_RECON_ACCUM);
  HAND_STENCIL_LEG_INT(ZM_PROJ,1,Zp,ZM_RECON_ACCUM);
  HAND_STENCIL_LEG_INT(TM_PROJ,0,Tp,TM_RECON_ACCUM);
  HAND_STENCIL_LEG_INT(XP_PROJ,3,Xm,XP_RECON_ACCUM);
  HAND_STENCIL_LEG_INT(YP_PROJ,2,Ym,YP_RECON_ACCUM);
  HAND_STENCIL_LEG_INT(ZP_PROJ,1,Zm,ZP_RECON_ACCUM);
  HAND_STENCIL_LEG_INT(TP_PROJ,0,Tm,TP_RECON_ACCUM);
  HAND_RESULT(ss);
}

template<class Impl>
void WilsonKernels<Impl>::HandDhopSiteDagInt(StencilImpl &st,LebesgueOrder &lo,DoubledGaugeField &U,SiteHalfSpinor *buf,
						  int ss,int sU,const FermionField &in, FermionField &out)
{
  typedef typename Simd::scalar_type S;
  typedef typename Simd::vector_type V;

  HAND_DECLARATIONS(ignore);

  StencilEntry *SE;
  int offset,local,perm, ptype;
  ZERO_RESULT;
  HAND_STENCIL_LEG_INT(XP_PROJ,3,Xp,XP_RECON_ACCUM);
  HAND_STENCIL_LEG_INT(YP_PROJ,2,Yp,YP_RECON_ACCUM);
  HAND_STENCIL_LEG_INT(ZP_PROJ,1,Zp,ZP_RECON_ACCUM);
  HAND_STENCIL_LEG_INT(TP_PROJ,0,Tp,TP_RECON_ACCUM);
  HAND_STENCIL_LEG_INT(XM_PROJ,3,Xm,XM_RECON_ACCUM);
  HAND_STENCIL_LEG_INT(YM_PROJ,2,Ym,YM_RECON_ACCUM);
  HAND_STENCIL_LEG_INT(ZM_PROJ,1,Zm,ZM_RECON_ACCUM);
  HAND_STENCIL_LEG_INT(TM_PROJ,0,Tm,TM_RECON_ACCUM);
  HAND_RESULT(ss);
}

template<class Impl> void 
WilsonKernels<Impl>::HandDhopSiteExt(StencilImpl &st,LebesgueOrder &lo,DoubledGaugeField &U,SiteHalfSpinor  *buf,
					  int ss,int sU,const FermionField &in, FermionField &out)
{
// T==0, Z==1, Y==2, Z==3 expect 1,2,2,2 simd layout etc...
  typedef typename Simd::scalar_type S;
  typedef typename Simd::vector_type V;

  HAND_DECLARATIONS(ignore);

  int offset,local,perm, ptype;
  StencilEntry *SE;
  int nmu=0;
  ZERO_RESULT;
  HAND_STENCIL_LEG_EXT(XM_PROJ,3,Xp,XM_RECON_ACCUM);
  HAND_STENCIL_LEG_EXT(YM_PROJ,2,Yp,YM_RECON_ACCUM);
  HAND_STENCIL_LEG_EXT(ZM_PROJ,1,Zp,ZM_RECON_ACCUM);
  HAND_STENCIL_LEG_EXT(TM_PROJ,0,Tp,TM_RECON_ACCUM);
  HAND_STENCIL_LEG_EXT(XP_PROJ,3,Xm,XP_RECON_ACCUM);
  HAND_STENCIL_LEG_EXT(YP_PROJ,2,Ym,YP_RECON_ACCUM);
  HAND_STENCIL_LEG_EXT(ZP_PROJ,1,Zm,ZP_RECON_ACCUM);
  HAND_STENCIL_LEG_EXT(TP_PROJ,0,Tm,TP_RECON_ACCUM);
  HAND_RESULT_EXT(ss);
}

template<class Impl>
void WilsonKernels<Impl>::HandDhopSiteDagExt(StencilImpl &st,LebesgueOrder &lo,DoubledGaugeField &U,SiteHalfSpinor *buf,
						  int ss,int sU,const FermionField &in, FermionField &out)
{
  typedef typename Simd::scalar_type S;
  typedef typename Simd::vector_type V;

  HAND_DECLARATIONS(ignore);

  StencilEntry *SE;
  int offset,local,perm, ptype;
  int nmu=0;
  ZERO_RESULT;
  HAND_STENCIL_LEG_EXT(XP_PROJ,3,Xp,XP_RECON_ACCUM);
  HAND_STENCIL_LEG_EXT(YP_PROJ,2,Yp,YP_RECON_ACCUM);
  HAND_STENCIL_LEG_EXT(ZP_PROJ,1,Zp,ZP_RECON_ACCUM);
  HAND_STENCIL_LEG_EXT(TP_PROJ,0,Tp,TP_RECON_ACCUM);
  HAND_STENCIL_LEG_EXT(XM_PROJ,3,Xm,XM_RECON_ACCUM);
  HAND_STENCIL_LEG_EXT(YM_PROJ,2,Ym,YM_RECON_ACCUM);
  HAND_STENCIL_LEG_EXT(ZM_PROJ,1,Zm,ZM_RECON_ACCUM);
  HAND_STENCIL_LEG_EXT(TM_PROJ,0,Tm,TM_RECON_ACCUM);
  HAND_RESULT_EXT(ss);
}

  ////////////////////////////////////////////////
  // Specialise Gparity to simple implementation
  ////////////////////////////////////////////////
#define HAND_SPECIALISE_EMPTY(IMPL)					\
  template<> void							\
  WilsonKernels<IMPL>::HandDhopSite(StencilImpl &st,			\
				    LebesgueOrder &lo,			\
				    DoubledGaugeField &U,		\
				    SiteHalfSpinor *buf,		\
				    int sF,int sU,			\
				    const FermionField &in,		\
				    FermionField &out){ assert(0); }	\
  template<> void							\
  WilsonKernels<IMPL>::HandDhopSiteDag(StencilImpl &st,			\
				    LebesgueOrder &lo,			\
				    DoubledGaugeField &U,		\
				    SiteHalfSpinor *buf,		\
				    int sF,int sU,			\
				    const FermionField &in,		\
				    FermionField &out){ assert(0); }	\
  template<> void							\
  WilsonKernels<IMPL>::HandDhopSiteInt(StencilImpl &st,			\
				    LebesgueOrder &lo,			\
				    DoubledGaugeField &U,		\
				    SiteHalfSpinor *buf,		\
				    int sF,int sU,			\
				    const FermionField &in,		\
				    FermionField &out){ assert(0); }	\
  template<> void							\
  WilsonKernels<IMPL>::HandDhopSiteExt(StencilImpl &st,			\
				    LebesgueOrder &lo,			\
				    DoubledGaugeField &U,		\
				    SiteHalfSpinor *buf,		\
				    int sF,int sU,			\
				    const FermionField &in,		\
				    FermionField &out){ assert(0); }	\
  template<> void							\
  WilsonKernels<IMPL>::HandDhopSiteDagInt(StencilImpl &st,	       	\
				    LebesgueOrder &lo,			\
				    DoubledGaugeField &U,		\
				    SiteHalfSpinor *buf,		\
				    int sF,int sU,			\
				    const FermionField &in,		\
				    FermionField &out){ assert(0); }	\
  template<> void							\
  WilsonKernels<IMPL>::HandDhopSiteDagExt(StencilImpl &st,	       	\
				    LebesgueOrder &lo,			\
				    DoubledGaugeField &U,		\
				    SiteHalfSpinor *buf,		\
				    int sF,int sU,			\
				    const FermionField &in,		\
				    FermionField &out){ assert(0); }	\

  HAND_SPECIALISE_EMPTY(GparityWilsonImplF);
  HAND_SPECIALISE_EMPTY(GparityWilsonImplD);
  HAND_SPECIALISE_EMPTY(GparityWilsonImplFH);
  HAND_SPECIALISE_EMPTY(GparityWilsonImplDF);

////////////// Wilson ; uses this implementation /////////////////////

#define INSTANTIATE_THEM(A) \
template void WilsonKernels<A>::HandDhopSite(StencilImpl &st,LebesgueOrder &lo,DoubledGaugeField &U,SiteHalfSpinor *buf,\
					     int ss,int sU,const FermionField &in, FermionField &out); \
template void WilsonKernels<A>::HandDhopSiteDag(StencilImpl &st,LebesgueOrder &lo,DoubledGaugeField &U,SiteHalfSpinor *buf, \
						int ss,int sU,const FermionField &in, FermionField &out);\
template void WilsonKernels<A>::HandDhopSiteInt(StencilImpl &st,LebesgueOrder &lo,DoubledGaugeField &U,SiteHalfSpinor *buf,\
						int ss,int sU,const FermionField &in, FermionField &out); \
template void WilsonKernels<A>::HandDhopSiteDagInt(StencilImpl &st,LebesgueOrder &lo,DoubledGaugeField &U,SiteHalfSpinor *buf, \
						   int ss,int sU,const FermionField &in, FermionField &out); \
template void WilsonKernels<A>::HandDhopSiteExt(StencilImpl &st,LebesgueOrder &lo,DoubledGaugeField &U,SiteHalfSpinor *buf,\
						int ss,int sU,const FermionField &in, FermionField &out); \
template void WilsonKernels<A>::HandDhopSiteDagExt(StencilImpl &st,LebesgueOrder &lo,DoubledGaugeField &U,SiteHalfSpinor *buf, \
						   int ss,int sU,const FermionField &in, FermionField &out); 

INSTANTIATE_THEM(WilsonImplF);
INSTANTIATE_THEM(WilsonImplD);
INSTANTIATE_THEM(ZWilsonImplF);
INSTANTIATE_THEM(ZWilsonImplD);
INSTANTIATE_THEM(GparityWilsonImplF);
INSTANTIATE_THEM(GparityWilsonImplD);
INSTANTIATE_THEM(DomainWallVec5dImplF);
INSTANTIATE_THEM(DomainWallVec5dImplD);
INSTANTIATE_THEM(ZDomainWallVec5dImplF);
INSTANTIATE_THEM(ZDomainWallVec5dImplD);
INSTANTIATE_THEM(WilsonImplFH);
INSTANTIATE_THEM(WilsonImplDF);
INSTANTIATE_THEM(ZWilsonImplFH);
INSTANTIATE_THEM(ZWilsonImplDF);
INSTANTIATE_THEM(GparityWilsonImplFH);
INSTANTIATE_THEM(GparityWilsonImplDF);
INSTANTIATE_THEM(DomainWallVec5dImplFH);
INSTANTIATE_THEM(DomainWallVec5dImplDF);
INSTANTIATE_THEM(ZDomainWallVec5dImplFH);
INSTANTIATE_THEM(ZDomainWallVec5dImplDF);

}}