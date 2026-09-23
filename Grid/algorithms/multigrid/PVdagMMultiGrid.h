/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./Grid/algorithms/multigrid/PVdagMMultiGrid.h

    Copyright (C) 2026

Author: Peter Boyle <pboyle@bnl.gov>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    See the full license in the file "LICENSE" in the top level distribution
    directory
*************************************************************************************/
/*  END LEGAL */
#pragma once

#include <Grid/algorithms/multigrid/DenseCoarseMatrix.h>
#include <Grid/algorithms/multigrid/PVdagMOperators.h>
#include <Grid/algorithms/multigrid/MrhsMultiGrid.h>

NAMESPACE_BEGIN(Grid);

//////////////////////////////////////////////////////////////////////////////////////
// The three-level mrhs PVdagM multigrid, as objects:
//
//   MGCoarseGrids              the derived coarse grids (owned here, since
//                              conformability is pointer identity and someone
//                              must hold them; declare it BEFORE anything that
//                              borrows from it)
//   PVdagMMultiGridCoarsening  ALL the coarsening information for one gauge
//                              configuration: the raw near-null basis, the
//                              transfer operators it induces, the Galerkin
//                              coarse operator at each level, and the dense
//                              bottom inverse.  Everything here is a function
//                              of the gauge field -- the state HMC must
//                              rebuild or maintain as U evolves -- and
//                              everything is solver-independent.
//   PVdagMMultiGridSolver      the solve chain composed on a borrowed
//                              coarsening: smoothers, coarse Krylov, V-cycle,
//                              outer mrhs PGCR.
//
// Scope discipline: grids outlive the coarsening outlives the solver.
//////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// Subspace I/O: bare-vector scidac records.  Loaded vectors are RAW --
// deliberately NOT re-orthogonalised: CoarsenOperator block
// orthonormalises in place, and projecting a block-orthonormal vector
// onto its own block-orthonormalised aggregation gives e_k with the
// near-null content silently gone.  GramGuard below catches that.
//////////////////////////////////////////////////////////////////////
template <class Field>
void saveSubspace(std::vector<Field> &subspace, std::string const fname){
#ifdef HAVE_LIME
  Grid::emptyUserRecord record;
  Grid::ScidacWriter SW(subspace[0].Grid()->IsBoss());
  SW.open(fname);
  for (int k = 0; k < (int)subspace.size(); k++) {
    SW.writeScidacFieldRecord(subspace[k], record);
  }
  SW.close();
#endif
}
template <class Field>
void loadSubspace(std::vector<Field> &subspace, std::string const fname){
#ifdef HAVE_LIME
  Grid::emptyUserRecord record;
  Grid::ScidacReader SR;
  SR.open(fname);
  for (int k = 0; k < (int)subspace.size(); k++) {
    SR.readScidacFieldRecord(subspace[k], record);
  }
  SR.close();
#endif
}

//////////////////////////////////////////////////////////////////////
// ||<v|v> - I||_F over a set of coarse vectors.  Small means the raw
// near-null content survived the projection.
//////////////////////////////////////////////////////////////////////
template<class CoarseField>
RealD GramDefect(std::vector<CoarseField> &v)
{
  RealD s2=0.0;
  for(int i=0;i<(int)v.size();i++){
    for(int j=0;j<(int)v.size();j++){
      ComplexD sij=TensorRemove(innerProduct(v[i],v[j]));
      ComplexD d=sij-(i==j?ComplexD(1.0):ComplexD(0.0));
      s2+=real(d)*real(d)+imag(d)*imag(d);
    }
  }
  return std::sqrt(s2);
}

// On a leak every image collapses to the block unit e_k, the Gram becomes
// N*I, and the defect lands at (N-1)*sqrt(nbasis) -- orders above the ~0.2
// of a content-preserving projection.  Trip well below that so a mis-set
// threshold costs a log line rather than the run.
template<class CoarseField>
void GramGuard(const std::string &name,std::vector<CoarseField> &v,GridBase *grid)
{
  RealD defect = GramDefect(v);
  RealD N      = (RealD)grid->gSites();
  RealD leak   = (N-1.0)*std::sqrt((RealD)v.size());
  RealD trip   = std::sqrt(N);
  std::cout << GridLogMessage << "GUARD: ||<"<<name<<"|"<<name<<"> - I||_F = " << defect
            << "   (e_k leak would be " << leak << ", trip at " << trip << ")" << std::endl;
  GRID_ASSERT( defect < trip );
}

//////////////////////////////////////////////////////////////////////
// The derived grids of the three-level chain, owned in one place.
// The coarse space is UNVECTORISED (sComplex scalar, simd {1,..,1});
// the 5D/6D grids are built directly so the SIMD layout is ours.
// rhs/batch is dim 0 of the 6D grids, undistributed -- no divisibility
// constraint on nrhs.
//////////////////////////////////////////////////////////////////////
class MGCoarseGrids {
public:
  GridCartesian *FGrid;                     // borrowed
  int Ls;
  int batch;
  Coordinate clatt;                         // 4d coarse lattice
  Coordinate cclatt;                        // 4d coarse-coarse lattice
  Coordinate c5simd, c5mpi;                 // 5D coarse simd/mpi
  Coordinate cmsimd, cmmpi;                 // 6D coarse simd/mpi
  // owned:
  GridCartesian *Coarse5d;
  GridCartesian *CoarseBatch;               // 6D at the coarsening batch
  GridCartesian *CoarseCoarse5d;
  GridCartesian *CoarseCoarseBatch;

  MGCoarseGrids(GridCartesian *_FGrid, const MGSetupParams &P)
    : FGrid(_FGrid)
  {
    Coordinate fdims = FGrid->FullDimensions();      // {Ls, x,y,z,t}
    Coordinate fmpi  = FGrid->_processors;
    GRID_ASSERT( fdims.size() == 5 );
    Ls    = fdims[0];
    batch = P.CoarsenBatch;

    clatt.resize(4); cclatt.resize(4);
    for(int d=0;d<4;d++){
      GRID_ASSERT( fdims[d+1] % P.Block[d] == 0 );
      clatt[d] = fdims[d+1] / P.Block[d];
    }
    for(int d=0;d<4;d++){
      GRID_ASSERT( clatt[d] % P.Block2[d] == 0 );
      cclatt[d] = clatt[d] / P.Block2[d];
    }
    std::cout << GridLogMessage << "MGCoarseGrids: Block  " << P.Block  << "  coarse lattice        " << clatt  << std::endl;
    std::cout << GridLogMessage << "MGCoarseGrids: Block2 " << P.Block2 << "  coarse-coarse lattice " << cclatt << std::endl;

    Coordinate c5latt({1,clatt[0],clatt[1],clatt[2],clatt[3]});
    c5simd = Coordinate({1,1,1,1,1});
    c5mpi  = Coordinate({1,fmpi[1],fmpi[2],fmpi[3],fmpi[4]});
    Coarse5d = new GridCartesian(c5latt,c5simd,c5mpi);

    cmsimd = Coordinate({1,1,1,1,1,1});
    cmmpi  = Coordinate({1,1,fmpi[1],fmpi[2],fmpi[3],fmpi[4]});
    Coordinate cblatt({batch,1,clatt[0],clatt[1],clatt[2],clatt[3]});
    CoarseBatch = new GridCartesian(cblatt,cmsimd,cmmpi);

    Coordinate cc5latt({1,cclatt[0],cclatt[1],cclatt[2],cclatt[3]});
    CoarseCoarse5d = new GridCartesian(cc5latt,c5simd,c5mpi);

    Coordinate ccblatt({batch,1,cclatt[0],cclatt[1],cclatt[2],cclatt[3]});
    CoarseCoarseBatch = new GridCartesian(ccblatt,cmsimd,cmmpi);
  }
  ~MGCoarseGrids()
  {
    delete CoarseCoarseBatch;
    delete CoarseCoarse5d;
    delete CoarseBatch;
    delete Coarse5d;
  }
};

//////////////////////////////////////////////////////////////////////
// ALL the coarsening information for one gauge configuration.
//
// Owns: the RAW near-null bases (fine and coarse -- retained so the
// coarse operators can be REBUILT on a changed gauge field with a
// fixed basis, the cheap HMC maintenance step; DiscardBasis() frees
// them for valence use), the two coarsened operators, the two block
// projectors, the dense bottom inverse (raw owning pointer: null in
// the constructor, allocated by BuildDenseBottom, deleted here), and
// the transient per-Nrhs solve grids (created by SetNrhs, which owns
// the ReleaseGrid -> delete -> new -> SetGrid ordering in ONE place).
//
// Borrows: the grid bundle.  Declare MGCoarseGrids first.
//////////////////////////////////////////////////////////////////////
template<class Fobj,class CComplex,int nbasis>
class PVdagMMultiGridCoarsening {
public:
  typedef Lattice<Fobj>                                                 FineField;
  typedef MultiGeneralCoarsenedOperatorV2<Fobj,CComplex,nbasis>         CoarseOperator;
  typedef typename CoarseOperator::CoarseVector                         CoarseVector;
  typedef typename CoarseVector::vector_object                          CoarseSiteObj;
  typedef iScalar<CComplex>                                             CComplex2;  // coarsening deepens the nest by one iScalar
  typedef MultiGeneralCoarsenedOperatorV2<CoarseSiteObj,CComplex2,nbasis> CoarseCoarseOperator;
  typedef typename CoarseCoarseOperator::CoarseVector                   CoarseCoarseVector;
  typedef DenseCoarseMatrix<CComplex2,nbasis>                           DenseBottom;

  MGCoarseGrids                     &Grids;      // borrowed
  MGSetupParams                      Params;
  NextToNearestStencilGeometry5D     geom;
  NextToNearestStencilGeometry5D     geom2;
  CoarseOperator                     CoarseOpPV;
  CoarseCoarseOperator               CoarseOpL2;
  MultiRHSBlockProject<FineField>    MrhsProjector;
  MultiRHSBlockProject<CoarseVector> MrhsProjectorL2;
  DenseBottom                       *DenseCC;
  std::vector<FineField>             rawNull;    // RAW fine near-null basis
  std::vector<CoarseVector>          rawPsi;     // RAW coarse near-null basis
  GridCartesian                     *CMrhs;      // transient solve grids, owned
  GridCartesian                     *CCMrhs;
  int                                nrhs;

  PVdagMMultiGridCoarsening(MGCoarseGrids &_Grids, const MGSetupParams &P)
    : Grids(_Grids),
      Params(P),
      geom (_Grids.Coarse5d),
      geom2(_Grids.CoarseCoarse5d),
      CoarseOpPV(geom ,_Grids.Coarse5d),
      CoarseOpL2(geom2,_Grids.CoarseCoarse5d),
      DenseCC(nullptr),
      CMrhs(nullptr),
      CCMrhs(nullptr),
      nrhs(-1)
  {};

  ~PVdagMMultiGridCoarsening()
  {
    if ( DenseCC ) delete DenseCC;
    CoarseOpPV.ReleaseGrid();     // borrowers let go before their grids die
    CoarseOpL2.ReleaseGrid();
    if ( CMrhs  ) delete CMrhs;
    if ( CCMrhs ) delete CCMrhs;
  }

  ////////////////////////////////////////////////////////////////////
  // The RAW fine basis: load from Params.SubspaceFile if it exists,
  // else GCR inverse iteration from noise (and save if a file name
  // was given).  The Aggregation is scaffolding for CreateSubspaceGCR
  // only -- that runs entirely on the fine grid, so the coarse grid
  // it holds is never dereferenced.
  ////////////////////////////////////////////////////////////////////
  void GetSubspace(GridParallelRNG &RNG, LinearOperatorBase<FineField> &FineOp)
  {
    uint64_t file_exists=0;
    if ( Params.SubspaceFile.length() ) {
      if ( Grids.FGrid->IsBoss() ){ std::ifstream f(Params.SubspaceFile); file_exists=f.good()?1:0; }
      Grids.FGrid->GlobalSum(file_exists);
    }
    rawNull.clear();
    rawNull.reserve(nbasis);
    for(int k=0;k<nbasis;k++) rawNull.push_back(FineField(Grids.FGrid));
    if ( file_exists ){
      std::cout << GridLogMessage << "PVdagMMultiGridCoarsening: loading subspace "
                << Params.SubspaceFile << " (kept RAW)" << std::endl;
      loadSubspace(rawNull, Params.SubspaceFile);
    } else {
      std::cout << GridLogMessage << "PVdagMMultiGridCoarsening: GCR subspace generation" << std::endl;
      Aggregation<Fobj,CComplex,nbasis> Agg(Grids.Coarse5d,Grids.FGrid,0);
      Agg.CreateSubspaceGCR(RNG,FineOp,nbasis);
      for(int k=0;k<nbasis;k++)
	rawNull[k]=Agg.subspace[k];
      if ( Params.SubspaceFile.length() )
	saveSubspace(rawNull, Params.SubspaceFile);
    }
  }

  ////////////////////////////////////////////////////////////////////
  // Both coarsenings from the retained RAW basis.  Calling this again
  // on a CHANGED fine operator is the fixed-basis rebuild: for the
  // PVdagM coarsening the coarse operator with a fixed basis is exact
  // in the changed links.  CoarsenOperator block-orthonormalises its
  // input in place, so working copies are taken and the RAW vectors
  // survive for the next rebuild.
  //
  // The fine operator must be the PVdagM wrapper (it needs
  // SloppyComms: the coarsening builds the PRECONDITIONER, so its
  // halos follow the FineSloppyComms policy, restored EXACT on exit).
  ////////////////////////////////////////////////////////////////////
  template<class PVdagMOp>
  void Coarsen(PVdagMOp &FineOp)
  {
    GRID_ASSERT( rawNull.size() == nbasis );

    // L1: working copy of the raw fine basis, orthonormalised in place
    std::vector<FineField> sub(rawNull);

    CoarseOpPV.SetGrid(Grids.CoarseBatch);

    std::cout << GridLogMessage << "PVdagMMultiGridCoarsening: L1 CoarsenOperator, batch "
              << Grids.batch << std::endl;

    FineOp.SloppyComms(Params.FineSloppyComms);
    CoarseOpPV.CoarsenOperator(FineOp,sub,Grids.Coarse5d,Grids.batch);
    FineOp.SloppyComms(0);

    // Transfer operators from the orthonormalised basis; then the
    // Galerkin images of the RAW basis define the L2 null space.
    MrhsProjector.Allocate(nbasis,Grids.FGrid,Grids.Coarse5d);
    MrhsProjector.ImportBasis(sub);
    sub.clear(); sub.shrink_to_fit();

    std::vector<CoarseVector> psi(nbasis,Grids.Coarse5d);
    MrhsProjector.blockProject(rawNull,psi);
    GramGuard("psi_coarse",psi,Grids.Coarse5d);

    rawPsi.clear();
    rawPsi.reserve(nbasis);
    for(int k=0;k<nbasis;k++) rawPsi.push_back(psi[k]);

    // L2: the V2 L1 operator is natively multiRHS at the batch grid
    CoarseOpL2.SetGrid(Grids.CoarseCoarseBatch);
    NonHermitianLinearOperator<CoarseOperator,CoarseVector> LinOpCoarse(CoarseOpPV);
    std::cout << GridLogMessage << "PVdagMMultiGridCoarsening: L2 CoarsenOperator, batch "
              << Grids.batch << std::endl;
    CoarseOpL2.CoarsenOperator(LinOpCoarse,Grids.CoarseBatch,psi,Grids.CoarseCoarse5d);

    MrhsProjectorL2.Allocate(nbasis,Grids.Coarse5d,Grids.CoarseCoarse5d);
    MrhsProjectorL2.ImportBasis(psi);              // now block orthonormal
    {
      std::vector<CoarseCoarseVector> psi_cc(nbasis,Grids.CoarseCoarse5d);
      MrhsProjectorL2.blockProject(rawPsi,psi_cc); // RAW vectors in
      GramGuard("psi_cc",psi_cc,Grids.CoarseCoarse5d);
    }
    nrhs = -1;                                     // both operators left at the batch grids
  }

  ////////////////////////////////////////////////////////////////////
  // Dense bottom: import the L2 matrix elements, invert (2D
  // block-cyclic, fp64), then the check Import cannot run itself --
  // the mrhs operator applies only on a D+1 grid, so drive it at
  // Nrhs 1 through a slice: ||A Ainv x - x||/||x||.
  ////////////////////////////////////////////////////////////////////
  void BuildDenseBottom(void)
  {
    if ( DenseCC ) delete DenseCC;
    std::cout << GridLogMessage << "PVdagMMultiGridCoarsening: L3 dense bottom import" << std::endl;
    DenseCC = new DenseBottom(Grids.CoarseCoarse5d);
    DenseCC->Import(CoarseOpL2);

    Coordinate cc1latt({1,1,Grids.cclatt[0],Grids.cclatt[1],Grids.cclatt[2],Grids.cclatt[3]});
    GridCartesian *CoarseCoarseOne = new GridCartesian(cc1latt,Grids.cmsimd,Grids.cmmpi);
    CoarseOpL2.SetGrid(CoarseCoarseOne);

    CoarseCoarseVector x(Grids.CoarseCoarse5d),y(Grids.CoarseCoarse5d),z(Grids.CoarseCoarse5d);
    GridParallelRNG dRNG(Grids.CoarseCoarse5d); dRNG.SeedFixedIntegers(std::vector<int>({11,12,13,14}));
    random(dRNG,x);

    (*DenseCC)(x,y);                    // y = Ainv x

    CoarseCoarseVector y1(CoarseCoarseOne),z1(CoarseCoarseOne);
    InsertSliceFast(y,y1,0,0);
    CoarseOpL2.M(y1,z1);                // z = A y
    ExtractSliceFast(z,z1,0,0);

    z = z - x;
    RealD rel = std::sqrt(norm2(z)/norm2(x));
    std::cout << GridLogMessage << "PVdagMMultiGridCoarsening: L3 dense ||A Ainv x - x||/||x|| = "
              << rel << std::endl;
    GRID_ASSERT( rel < 1.0e-2 );

    CoarseOpL2.ReleaseGrid();           // let go before the grid dies
    delete CoarseCoarseOne;
    nrhs = -1;
  }

  ////////////////////////////////////////////////////////////////////
  // Retarget both operators to a solve Nrhs.  The matrix elements are
  // Nrhs independent and survive; only the D+1 grids and BLAS buffers
  // change.  The ordering discipline (borrowers release BEFORE their
  // grids are destroyed) lives here and nowhere else.
  ////////////////////////////////////////////////////////////////////
  void SetNrhs(int nr)
  {
    if ( nr == nrhs ) return;
    CoarseOpPV.ReleaseGrid();
    CoarseOpL2.ReleaseGrid();
    if ( CMrhs  ) { delete CMrhs;  CMrhs =nullptr; }
    if ( CCMrhs ) { delete CCMrhs; CCMrhs=nullptr; }
    Coordinate cml ({nr,1,Grids.clatt[0], Grids.clatt[1], Grids.clatt[2], Grids.clatt[3]});
    Coordinate ccml({nr,1,Grids.cclatt[0],Grids.cclatt[1],Grids.cclatt[2],Grids.cclatt[3]});
    CMrhs  = new GridCartesian(cml, Grids.cmsimd,Grids.cmmpi);
    CCMrhs = new GridCartesian(ccml,Grids.cmsimd,Grids.cmmpi);
    CoarseOpPV.SetGrid(CMrhs);
    CoarseOpL2.SetGrid(CCMrhs);
    nrhs = nr;
    std::cout << GridLogMessage << "PVdagMMultiGridCoarsening: operators at Nrhs " << nr << std::endl;
  }

  ////////////////////////////////////////////////////////////////////
  // Free the retained RAW bases (valence use: coarsening is final for
  // this configuration and the fine basis is ~GB-scale host memory).
  ////////////////////////////////////////////////////////////////////
  void DiscardBasis(void)
  {
    rawNull.clear(); rawNull.shrink_to_fit();
    rawPsi.clear();  rawPsi.shrink_to_fit();
  }
};

//////////////////////////////////////////////////////////////////////
// The solve chain on a borrowed coarsening: adaptive PGCR smoothers
// at both levels (the one correct route for this non-Hermitian
// chain), dense bottom, mrhs V-cycle, outer shared-coefficient mrhs
// PGCR.  Construction retargets the coarsening to this Nrhs.
//
// Halo policy: the whole V-cycle is preconditioner and runs with
// sloppy halos when FineSloppyComms is set; the outer Krylov is EXACT
// (its applications define what "converged" means), asserted by the
// FINAL true-residual report in Solve.
//////////////////////////////////////////////////////////////////////
template<class Matrix,class Coarsening>
class PVdagMMultiGridSolver {
public:
  typedef typename Coarsening::FineField            FineField;
  typedef typename Coarsening::CoarseOperator       CoarseOperator;
  typedef typename Coarsening::CoarseVector         CoarseVector;
  typedef typename Coarsening::CoarseCoarseOperator CoarseCoarseOperator;
  typedef typename Coarsening::CoarseCoarseVector   CoarseCoarseVector;
  typedef typename Coarsening::DenseBottom          DenseBottom;
  typedef PrecGeneralisedConjugateResidualNonHermitian<FineField>    FineSmoother_t;
  typedef PrecGeneralisedConjugateResidualNonHermitian<CoarseVector> CoarseKrylov_t;

  Coarsening            &C;
  PVdagMMultiGridParams  Params;
  int                    nrhs;
  int                    _regrid;    // FIRST member-like init: SetNrhs before members capture grids

  PVdagMLinearOperator<Matrix,FineField>        PVdagM;
  ShiftedPVdagMLinearOperator<Matrix,FineField> ShiftedPVdagM;
  TrivialPrecon<FineField>                      simple_fine;
  TrivialPrecon<CoarseVector>                   simpleC;
  NonHermitianLinearOperator<CoarseOperator,CoarseVector> LinOpC;
  ShiftedLinearOperator<CoarseVector>           ShiftedC;
  MrhsDenseCCSolve<DenseBottom,CoarseCoarseVector> ccSolve;
  CoarseKrylov_t                                CoarseSmootherGCR;
  MrhsCoarseThreeLevelPrec<CoarseVector,CoarseCoarseVector> L2to3Precon;
  CoarseKrylov_t                                L2PGCR;
  FineSmoother_t                                SmootherGCR;
  MrhsTwoLevelMG<FineField,CoarseVector,FineSmoother_t> ThreeLevelPrecon;
  MrhsPGCRNonHermitian<FineField>               L1PGCR;

  PVdagMMultiGridSolver(Matrix &Ddwf, Matrix &Dpv,
                        Coarsening &_C, const PVdagMMultiGridParams &P, int nr)
    : C(_C), Params(P), nrhs(nr),
      _regrid((C.SetNrhs(nr),0)),              // members below capture C.CMrhs/C.CCMrhs
      PVdagM(Ddwf,Dpv),
      ShiftedPVdagM(P.FineSmoother.Shift,Ddwf,Dpv),
      LinOpC(C.CoarseOpPV),
      ShiftedC(P.CoarseSmoother.Shift,LinOpC),
      ccSolve(*C.DenseCC,nr),
      // Coarse smoother: one preconditioned restart of nstep GCR steps.
      CoarseSmootherGCR(0.01,1,ShiftedC,simpleC,P.CoarseSmoother.Mmax,P.CoarseSmoother.Nstep),
      L2to3Precon(LinOpC,CoarseSmootherGCR,C.MrhsProjectorL2,ccSolve,
                  C.Grids.Coarse5d,C.Grids.CoarseCoarse5d,C.CCMrhs,nr),
      // Coarse Krylov: Order/16 restarts of 16 steps.
      L2PGCR(P.CoarseSolver.Tol,P.CoarseSolver.Order/16,LinOpC,L2to3Precon,P.CoarseSolver.Mmax,16),
      // Fine smoother: one restart of nstep GCR steps, tolerance 0 = fixed work.
      SmootherGCR(0.0,1,ShiftedPVdagM,simple_fine,P.FineSmoother.Mmax,P.FineSmoother.Nstep),
      ThreeLevelPrecon(PVdagM,SmootherGCR,C.MrhsProjector,L2PGCR,C.Grids.Coarse5d,C.CMrhs),
      L1PGCR(P.Outer.Tol,P.Outer.MaxIterations,PVdagM,ThreeLevelPrecon,P.Outer.Mmax,P.Outer.Nstep)
  {
    GRID_ASSERT( C.DenseCC != nullptr );
    
    CoarseSmootherGCR.Level(2);
    CoarseSmootherGCR.Name("Csmoother");
    CoarseSmootherGCR.SetZeroGuess(1);

    L2PGCR.Level(2);
    L2PGCR.Name("Couter");
    L2PGCR.SetZeroGuess(1);
    
    SmootherGCR.Level(1);
    SmootherGCR.Name("Fsmoother");
    SmootherGCR.SetZeroGuess(1);

    L1PGCR.Level(1);
    L1PGCR.Name("Fouter");
    L1PGCR.SetZeroGuess(1);
    
    // The V-cycle is preconditioner: sloppy halos inside, exact restored on exit.
    ThreeLevelPrecon.SetSloppy    = [this](int s){ PVdagM.SloppyComms(s); };
    ThreeLevelPrecon.SloppyComms  = Params.Setup.FineSloppyComms;
    PVdagM.SloppyComms(0);
    std::cout << GridLogMessage << "PVdagMMultiGridSolver: Nrhs " << nr
              << ", fine halo policy: preconditioner+coarsening "
              << (Params.Setup.FineSloppyComms ? "SLOPPY (fp32 wire)" : "exact")
              << ", outer Krylov EXACT" << std::endl;
  }

  void Solve(std::vector<FineField> &src, std::vector<FineField> &sol)
  {
    GRID_ASSERT( (int)src.size() == nrhs );
    // Start the solve from a clean device: evict setup-era Lattice copies
    // and release the allocation caches' held blocks, so the LRU cap
    // applies to the solve's own working set (OOM policy, Frontier).
    MemoryManager::EvictAll();
    MemoryManager::DropCache();

    GridStopWatch w; w.Start();
    L1PGCR(src,sol);
    w.Stop();
    std::cout << GridLogMessage << "PVdagMMultiGridSolver: Nrhs "<<nrhs<<" total " << w.Elapsed()
              << "  (per RHS: " << w.useconds()/1.0e6/nrhs << " s)" << std::endl;

    // The outer operator is exact by policy; assert the state rather than
    // trust it -- a preconditioner that failed to restore would surface here.
    PVdagM.SloppyComms(0);
    {
      FineField Ax(src[0].Grid());
      RealD worst=0.0;
      for(int r=0;r<nrhs;r++){
	PVdagM.Op(sol[r],Ax);
	Ax=Ax-src[r];
        RealD rn=std::sqrt(norm2(Ax)/norm2(src[r]));
        std::cout << GridLogMessage << "FINAL Nrhs "<<nrhs<<": rhs["<<r<<"] true residual = " << rn << std::endl;
        worst=std::max(worst,rn);
      }
      std::cout << GridLogMessage << "FINAL Nrhs "<<nrhs<<": worst-case residual = " << worst
                << "   (exact-halo verification)" << std::endl;
    }
  }
};

NAMESPACE_END(Grid);
