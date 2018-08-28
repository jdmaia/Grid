#ifndef A2A_Vectors_hpp_
#define A2A_Vectors_hpp_

#include <Grid/Hadrons/Global.hpp>
#include <Grid/Hadrons/Environment.hpp>
#include <Grid/Hadrons/Solver.hpp>

BEGIN_HADRONS_NAMESPACE

////////////////////////////////
// A2A Modes
////////////////////////////////

template <class Field, class Matrix, class Solver>
class A2AModesSchurDiagTwo
{
  private:
    const std::vector<Field> *evec;
    const std::vector<RealD> *eval;
    Matrix &action;
    Solver &solver;
    std::vector<Field> w_high_5d, v_high_5d, w_high_4d, v_high_4d;
    const int Nl, Nh;
    const bool return_5d;

  public:

  int getNl (void ) {return Nl;}
  int getNh (void ) {return Nh;}
  int getN  (void ) {return Nh+Nl;}

    A2AModesSchurDiagTwo(const std::vector<Field> *_evec, const std::vector<RealD> *_eval,
                         Matrix &_action,
                         Solver &_solver,
                         std::vector<Field> _w_high_5d, std::vector<Field> _v_high_5d,
                         std::vector<Field> _w_high_4d, std::vector<Field> _v_high_4d,
                         const int _Nl, const int _Nh,
                         const bool _return_5d)
                        : evec(_evec), eval(_eval),
                        action(_action),
                        solver(_solver),
                        w_high_5d(_w_high_5d), v_high_5d(_v_high_5d),
                        w_high_4d(_w_high_4d), v_high_4d(_v_high_4d),
                        Nl(_Nl), Nh(_Nh),
                        return_5d(_return_5d){};

    void high_modes(Field &source_5d, Field &w_source_5d, Field &source_4d, int i)
    {
        int i5d;
        LOG(Message) << "A2A high modes for i = " << i << std::endl;
        i5d = 0;
        if (return_5d) i5d = i;
        this->high_mode_v(action, solver, source_5d, v_high_5d[i5d], v_high_4d[i]);
        this->high_mode_w(w_source_5d, source_4d, w_high_5d[i5d], w_high_4d[i]);
    }

    void return_v(int i, Field &vout_5d, Field &vout_4d)
    {
        if (i < Nl)
        {
            this->low_mode_v(action, evec->at(i), eval->at(i), vout_5d, vout_4d);
        }
        else
        {
            vout_4d = v_high_4d[i - Nl];
            if (!(return_5d)) i = Nl;
            vout_5d = v_high_5d[i - Nl];
        }
    }
    void return_w(int i, Field &wout_5d, Field &wout_4d)
    {
        if (i < Nl)
        {
            this->low_mode_w(action, evec->at(i), eval->at(i), wout_5d, wout_4d);
        }
        else
        {
            wout_4d = w_high_4d[i - Nl];
            if (!(return_5d)) i = Nl;
            wout_5d = w_high_5d[i - Nl];
        }
    }

    void low_mode_v(Matrix &action, const Field &evec, const RealD &eval, Field &vout_5d, Field &vout_4d)
    {
        GridBase *grid = action.RedBlackGrid();
        Field src_o(grid);
        Field sol_e(grid);
        Field sol_o(grid);
        Field tmp(grid);

        src_o = evec;
        src_o.checkerboard = Odd;
        pickCheckerboard(Even, sol_e, vout_5d);
        pickCheckerboard(Odd, sol_o, vout_5d);

        /////////////////////////////////////////////////////
        // v_ie = -(1/eval_i) * MeeInv Meo MooInv evec_i
        /////////////////////////////////////////////////////
        action.MooeeInv(src_o, tmp);
        assert(tmp.checkerboard == Odd);
        action.Meooe(tmp, sol_e);
        assert(sol_e.checkerboard == Even);
        action.MooeeInv(sol_e, tmp);
        assert(tmp.checkerboard == Even);
        sol_e = (-1.0 / eval) * tmp;
        assert(sol_e.checkerboard == Even);

        /////////////////////////////////////////////////////
        // v_io = (1/eval_i) * MooInv evec_i
        /////////////////////////////////////////////////////
        action.MooeeInv(src_o, tmp);
        assert(tmp.checkerboard == Odd);
        sol_o = (1.0 / eval) * tmp;
        assert(sol_o.checkerboard == Odd);

        setCheckerboard(vout_5d, sol_e);
        assert(sol_e.checkerboard == Even);
        setCheckerboard(vout_5d, sol_o);
        assert(sol_o.checkerboard == Odd);

        action.ExportPhysicalFermionSolution(vout_5d, vout_4d);
    }

    void low_mode_w(Matrix &action, const Field &evec, const RealD &eval, Field &wout_5d, Field &wout_4d)
    {
        GridBase *grid = action.RedBlackGrid();
        SchurDiagTwoOperator<Matrix, Field> _HermOpEO(action);

        Field src_o(grid);
        Field sol_e(grid);
        Field sol_o(grid);
        Field tmp(grid);

        GridBase *fgrid = action.Grid();
        Field tmp_wout(fgrid);

        src_o = evec;
        src_o.checkerboard = Odd;
        pickCheckerboard(Even, sol_e, tmp_wout);
        pickCheckerboard(Odd, sol_o, tmp_wout);

        /////////////////////////////////////////////////////
        // w_ie = - MeeInvDag MoeDag Doo evec_i
        /////////////////////////////////////////////////////
        _HermOpEO.Mpc(src_o, tmp);
        assert(tmp.checkerboard == Odd);
        action.MeooeDag(tmp, sol_e);
        assert(sol_e.checkerboard == Even);
        action.MooeeInvDag(sol_e, tmp);
        assert(tmp.checkerboard == Even);
        sol_e = (-1.0) * tmp;

        /////////////////////////////////////////////////////
        // w_io = Doo evec_i
        /////////////////////////////////////////////////////
        _HermOpEO.Mpc(src_o, sol_o);
        assert(sol_o.checkerboard == Odd);

        setCheckerboard(tmp_wout, sol_e);
        assert(sol_e.checkerboard == Even);
        setCheckerboard(tmp_wout, sol_o);
        assert(sol_o.checkerboard == Odd);

        action.DminusDag(tmp_wout, wout_5d);

        action.ExportPhysicalFermionSource(wout_5d, wout_4d);
    }

    void high_mode_v(Matrix &action, Solver &solver, const Field &source, Field &vout_5d, Field &vout_4d)
    {
        GridBase *fgrid = action.Grid();
        solver(vout_5d, source); // Note: solver is solver(out, in)
        action.ExportPhysicalFermionSolution(vout_5d, vout_4d);
    }

    void high_mode_w(const Field &w_source_5d, const Field &source_4d, Field &wout_5d, Field &wout_4d)
    {
        wout_5d = w_source_5d;
        wout_4d = source_4d;
    }
};

// TODO: A2A for coarse eigenvectors

// template <class FineField, class CoarseField, class Matrix, class Solver>
// class A2ALMSchurDiagTwoCoarse : public A2AModesSchurDiagTwo<FineField, Matrix, Solver>
// {
//   private:
//     const std::vector<FineField> &subspace;
//     const std::vector<CoarseField> &evec_coarse;
//     const std::vector<RealD> &eval_coarse;
//     Matrix &action;

//   public:
//     A2ALMSchurDiagTwoCoarse(const std::vector<FineField> &_subspace, const std::vector<CoarseField> &_evec_coarse, const std::vector<RealD> &_eval_coarse, Matrix &_action)
//         : subspace(_subspace), evec_coarse(_evec_coarse), eval_coarse(_eval_coarse), action(_action){};

//     void operator()(int i, FineField &vout, FineField &wout)
//     {
//         FineField prom_evec(subspace[0]._grid);
//         blockPromote(evec_coarse[i], prom_evec, subspace);
//         this->low_mode_v(action, prom_evec, eval_coarse[i], vout);
//         this->low_mode_w(action, prom_evec, eval_coarse[i], wout);
//     }
// };

END_HADRONS_NAMESPACE

#endif // A2A_Vectors_hpp_