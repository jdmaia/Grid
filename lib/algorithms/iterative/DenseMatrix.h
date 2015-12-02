#ifndef GRID_DENSE_MATRIX_H
#define GRID_DENSE_MATRIX_H

namespace Grid {
    /////////////////////////////////////////////////////////////
    // Matrix untils
    /////////////////////////////////////////////////////////////

template<class T> using DenseVector = std::vector<T>;
template<class T> using DenseMatrix = DenseVector<DenseVector<T> >;

template<class T> void Size(DenseVector<T> & vec, int &N) 
{ 
  N= vec.size();
}
template<class T> void Size(DenseMatrix<T> & mat, int &N,int &M) 
{ 
  N= mat.size();
  M= mat[0].size();
}

template<class T> void SizeSquare(DenseMatrix<T> & mat, int &N) 
{ 
  int M; Size(mat,N,M);
  assert(N==M);
}

template<class T> void Resize(DenseMatrix<T > & mat, int N, int M) { 
  mat.resize(N);
  for(int i=0;i<N;i++){
    mat[i].resize(M);
  }
}
template<class T> void Fill(DenseMatrix<T> & mat, T&val) { 
  int N,M;
  Size(mat,N,M);
  for(int i=0;i<N;i++){
  for(int j=0;j<M;j++){
    mat[i][j] = val;
  }}
}

/** Transpose of a matrix **/
template<class T> DenseMatrix<T> Transpose(DenseMatrix<T> & mat){
  int N,M;
  Size(mat,N,M);
  DenseMatrix<T> C; Resize(C,M,N);
  for(int i=0;i<M;i++){
  for(int j=0;j<N;j++){
    C[i][j] = mat[j][i];
  }} 
  return C;
}
/** Set DenseMatrix to unit matrix **/
template<class T> void Unity(DenseMatrix<T> &A){
  int N;  SizeSquare(A,N);
  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      if ( i==j ) A[i][j] = 1;
      else        A[i][j] = 0;
    } 
  } 
}

/** Add C * I to matrix **/
template<class T>
void PlusUnit(DenseMatrix<T> & A,T c){
  int dim;  SizeSquare(A,dim);
  for(int i=0;i<dim;i++){A[i][i] = A[i][i] + c;} 
}

/** return the Hermitian conjugate of matrix **/
template<class T>
DenseMatrix<T> HermitianConj(DenseMatrix<T> &mat){

  int dim; SizeSquare(mat,dim);

  DenseMatrix<T> C; Resize(C,dim,dim);

  for(int i=0;i<dim;i++){
    for(int j=0;j<dim;j++){
      C[i][j] = conj(mat[j][i]);
    } 
  } 
  return C;
}
/**Get a square submatrix**/
template <class T>
DenseMatrix<T> GetSubMtx(DenseMatrix<T> &A,int row_st, int row_end, int col_st, int col_end)
{
  DenseMatrix<T> H; Resize(H,row_end - row_st,col_end-col_st);

  for(int i = row_st; i<row_end; i++){
  for(int j = col_st; j<col_end; j++){
    H[i-row_st][j-col_st]=A[i][j];
  }}
  return H;
}

}

#include <algorithms/iterative/Householder.h>
#include <algorithms/iterative/Francis.h>

#endif

